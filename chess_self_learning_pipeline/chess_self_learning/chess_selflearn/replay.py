from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import h5py
import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info


POLICY_SENTINEL = np.uint16(65535)


@dataclass(slots=True)
class TrainingSample:
    state: np.ndarray
    policy_indices: np.ndarray
    policy_probabilities: np.ndarray
    value: int
    ply: int
    game_id: int


@dataclass(slots=True)
class FinishedGame:
    samples: list[TrainingSample]
    result_white: int
    moves_uci: list[str]
    termination: str


class ReplayWriter:
    def __init__(
        self,
        path: str | Path,
        max_policy_moves: int,
        generation: int,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_policy_moves = max_policy_moves
        self.generation = generation
        self.file = h5py.File(self.path, "w")
        self.count = 0
        self.game_count = 0

        self.datasets = {
            "states": self.file.create_dataset(
                "states",
                shape=(0, 34, 8, 8),
                maxshape=(None, 34, 8, 8),
                dtype=np.uint8,
                chunks=(256, 34, 8, 8),
                compression="lzf",
                shuffle=True,
            ),
            "policy_indices": self.file.create_dataset(
                "policy_indices",
                shape=(0, max_policy_moves),
                maxshape=(None, max_policy_moves),
                dtype=np.uint16,
                chunks=(512, max_policy_moves),
                compression="lzf",
            ),
            "policy_probabilities": self.file.create_dataset(
                "policy_probabilities",
                shape=(0, max_policy_moves),
                maxshape=(None, max_policy_moves),
                dtype=np.float16,
                chunks=(512, max_policy_moves),
                compression="lzf",
            ),
            "policy_count": self.file.create_dataset(
                "policy_count",
                shape=(0,),
                maxshape=(None,),
                dtype=np.uint16,
                chunks=(4096,),
                compression="lzf",
            ),
            "values": self.file.create_dataset(
                "values",
                shape=(0,),
                maxshape=(None,),
                dtype=np.int8,
                chunks=(4096,),
                compression="lzf",
            ),
            "plies": self.file.create_dataset(
                "plies",
                shape=(0,),
                maxshape=(None,),
                dtype=np.uint16,
                chunks=(4096,),
                compression="lzf",
            ),
            "game_ids": self.file.create_dataset(
                "game_ids",
                shape=(0,),
                maxshape=(None,),
                dtype=np.uint64,
                chunks=(4096,),
                compression="lzf",
            ),
        }
        self.file.attrs["format_version"] = 1
        self.file.attrs["generation"] = generation
        self.file.attrs["max_policy_moves"] = max_policy_moves
        self.file.attrs["action_space_size"] = 4672

    def append_games(self, games: list[FinishedGame]) -> None:
        samples = [sample for game in games for sample in game.samples]
        if not samples:
            return

        count = len(samples)
        start = self.count
        end = start + count
        for dataset in self.datasets.values():
            dataset.resize((end, *dataset.shape[1:]))

        states = np.stack([sample.state for sample in samples])
        indices = np.full(
            (count, self.max_policy_moves),
            POLICY_SENTINEL,
            dtype=np.uint16,
        )
        probabilities = np.zeros(
            (count, self.max_policy_moves),
            dtype=np.float16,
        )
        policy_counts = np.empty(count, dtype=np.uint16)

        for row, sample in enumerate(samples):
            policy_count = len(sample.policy_indices)
            if policy_count > self.max_policy_moves:
                raise ValueError(
                    f"Policy contains {policy_count} moves; maximum is "
                    f"{self.max_policy_moves}"
                )
            indices[row, :policy_count] = sample.policy_indices
            probabilities[row, :policy_count] = sample.policy_probabilities
            policy_counts[row] = policy_count

        self.datasets["states"][start:end] = states
        self.datasets["policy_indices"][start:end] = indices
        self.datasets["policy_probabilities"][start:end] = probabilities
        self.datasets["policy_count"][start:end] = policy_counts
        self.datasets["values"][start:end] = np.fromiter(
            (sample.value for sample in samples),
            dtype=np.int8,
            count=count,
        )
        self.datasets["plies"][start:end] = np.fromiter(
            (sample.ply for sample in samples),
            dtype=np.uint16,
            count=count,
        )
        self.datasets["game_ids"][start:end] = np.fromiter(
            (sample.game_id for sample in samples),
            dtype=np.uint64,
            count=count,
        )

        self.count = end
        self.game_count += len(games)
        self.file.attrs["positions"] = self.count
        self.file.attrs["games"] = self.game_count
        self.file.flush()

    def close(self) -> None:
        self.file.attrs["positions"] = self.count
        self.file.attrs["games"] = self.game_count
        self.file.flush()
        self.file.close()

    def __enter__(self) -> "ReplayWriter":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def update_replay_manifest(
    run_root: str | Path,
    *,
    iteration: int,
    replay_path: str | Path,
    games: int,
    positions: int,
    generation: int,
) -> None:
    root = Path(run_root)
    path = root / "replay_manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
    else:
        manifest = {"format_version": 1, "iterations": []}

    manifest["iterations"] = [
        entry
        for entry in manifest["iterations"]
        if int(entry["iteration"]) != iteration
    ]
    manifest["iterations"].append(
        {
            "iteration": iteration,
            "generation": generation,
            "path": str(Path(replay_path).resolve()),
            "games": games,
            "positions": positions,
        }
    )
    manifest["iterations"].sort(key=lambda entry: int(entry["iteration"]))

    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def replay_files(run_root: str | Path, keep_iterations: int) -> list[Path]:
    path = Path(run_root) / "replay_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing replay manifest: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    entries = manifest["iterations"][-keep_iterations:]
    files = [Path(entry["path"]) for entry in entries]
    missing = [file for file in files if not file.exists()]
    if missing:
        raise FileNotFoundError(f"Missing replay files: {missing}")
    return files


class ReplayIterableDataset(IterableDataset):
    def __init__(
        self,
        files: list[Path],
        read_chunk_size: int,
        seed: int,
    ) -> None:
        super().__init__()
        self.files = files
        self.read_chunk_size = read_chunk_size
        self.seed = seed

    def __iter__(self) -> Iterator[tuple[torch.Tensor, ...]]:
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        workers = 1 if worker is None else worker.num_workers
        worker_seed = self.seed if worker is None else self.seed ^ int(worker.seed)
        rng = np.random.default_rng(worker_seed + worker_id * 997)

        files = self.files[worker_id::workers]
        file_order = list(range(len(files)))
        rng.shuffle(file_order)

        for file_index in file_order:
            with h5py.File(files[file_index], "r", swmr=True) as handle:
                count = int(handle["states"].shape[0])
                starts = list(range(0, count, self.read_chunk_size))
                rng.shuffle(starts)

                for start in starts:
                    end = min(start + self.read_chunk_size, count)
                    states = handle["states"][start:end]
                    indices = handle["policy_indices"][start:end]
                    probabilities = handle["policy_probabilities"][start:end]
                    policy_count = handle["policy_count"][start:end]
                    values = handle["values"][start:end]
                    order = rng.permutation(len(states))

                    for row in order:
                        yield (
                            torch.from_numpy(states[row]),
                            torch.from_numpy(indices[row].astype(np.int64)),
                            torch.from_numpy(probabilities[row].astype(np.float32)),
                            torch.tensor(int(policy_count[row]), dtype=torch.long),
                            torch.tensor(float(values[row]), dtype=torch.float32),
                        )


class BootstrapIterableDataset(IterableDataset):
    def __init__(
        self,
        dataset_root: str | Path,
        read_chunk_size: int,
        seed: int,
    ) -> None:
        super().__init__()
        self.root = Path(dataset_root)
        self.read_chunk_size = read_chunk_size
        self.seed = seed
        manifest = json.loads(
            (self.root / "manifest.json").read_text(encoding="utf-8")
        )
        self.shards = list(manifest["shards"])

    def __iter__(self) -> Iterator[tuple[torch.Tensor, ...]]:
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        workers = 1 if worker is None else worker.num_workers
        worker_seed = self.seed if worker is None else self.seed ^ int(worker.seed)
        rng = np.random.default_rng(worker_seed + worker_id * 991)

        shards = self.shards[worker_id::workers]
        order = list(range(len(shards)))
        rng.shuffle(order)

        for shard_index in order:
            path = self.root / str(shards[shard_index]["filename"])
            with h5py.File(path, "r", swmr=True) as handle:
                count = int(handle["states"].shape[0])
                starts = list(range(0, count, self.read_chunk_size))
                rng.shuffle(starts)
                for start in starts:
                    end = min(start + self.read_chunk_size, count)
                    states = handle["states"][start:end]
                    actions = handle["actions"][start:end]
                    values = handle["values"][start:end]
                    rows = rng.permutation(len(states))
                    for row in rows:
                        yield (
                            torch.from_numpy(states[row]),
                            torch.tensor(int(actions[row]), dtype=torch.long),
                            torch.tensor(float(values[row]), dtype=torch.float32),
                        )
