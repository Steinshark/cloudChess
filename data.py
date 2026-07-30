from __future__ import annotations
import json
from pathlib import Path
from typing import Iterator, Literal
import h5py
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import IterableDataset, get_worker_info

SplitName = Literal["train", "validation", "test"]

def stable_game_bucket(game_ids: np.ndarray, seed: int) -> np.ndarray:
    x = game_ids.astype(np.uint64, copy=True)
    x ^= np.uint64(seed)
    x ^= x >> np.uint64(30)
    x *= np.uint64(0xBF58476D1CE4E5B9)
    x ^= x >> np.uint64(27)
    x *= np.uint64(0x94D049BB133111EB)
    x ^= x >> np.uint64(31)
    return (x % np.uint64(100)).astype(np.uint8)

def split_mask(
    game_ids: np.ndarray,
    split: SplitName,
    train_percent: int,
    validation_percent: int,
    seed: int,
) -> np.ndarray:
    buckets = stable_game_bucket(game_ids, seed)
    validation_end = train_percent + validation_percent
    if split == "train":
        return buckets < train_percent
    if split == "validation":
        return (buckets >= train_percent) & (buckets < validation_end)
    if split == "test":
        return buckets >= validation_end
    raise ValueError(split)

class H5ChessIterableDataset(IterableDataset):
    def __init__(
        self,
        dataset_root: str | Path,
        split: SplitName,
        train_split_percent: int,
        validation_split_percent: int,
        split_seed: int,
        read_chunk_size: int = 8192,
        shuffle_buffer_chunks: int = 4,
        base_seed: int = 1337,
        shuffle: bool = True,
    ) -> None:
        super().__init__()
        self.root = Path(dataset_root)
        self.split = split
        self.train_split_percent = train_split_percent
        self.validation_split_percent = validation_split_percent
        self.split_seed = split_seed
        self.read_chunk_size = read_chunk_size
        self.shuffle_buffer_chunks = max(1, shuffle_buffer_chunks)
        self.base_seed = base_seed
        self.shuffle = shuffle
        with (self.root / "manifest.json").open("r", encoding="utf-8") as handle:
            self.manifest = json.load(handle)
        self.shards = list(self.manifest["shards"])
        if tuple(self.manifest["state_shape"]) != (34, 8, 8):
            raise ValueError(f"Unexpected state shape: {self.manifest['state_shape']}")
        if int(self.manifest["action_space_size"]) != 4672:
            raise ValueError("Unexpected action space")

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor, Tensor]]:
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        worker_count = 1 if worker is None else worker.num_workers
        rng = np.random.default_rng(self.base_seed + worker_id * 1009)

        shard_indices = list(range(len(self.shards)))[worker_id::worker_count]
        if self.shuffle:
            rng.shuffle(shard_indices)

        buffer: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

        def emit(arrays):
            states, actions, values = arrays
            if self.shuffle:
                order = rng.permutation(len(states))
                states, actions, values = states[order], actions[order], values[order]
            for i in range(len(states)):
                yield (
                    torch.from_numpy(states[i]),
                    torch.tensor(int(actions[i]), dtype=torch.long),
                    torch.tensor(float(values[i]), dtype=torch.float32),
                )

        for shard_index in shard_indices:
            shard_path = self.root / str(self.shards[shard_index]["filename"])
            with h5py.File(shard_path, "r", swmr=True) as handle:
                count = int(handle["states"].shape[0])
                starts = list(range(0, count, self.read_chunk_size))
                if self.shuffle:
                    rng.shuffle(starts)
                for start in starts:
                    end = min(start + self.read_chunk_size, count)
                    game_ids = handle["game_ids"][start:end]
                    keep = split_mask(
                        game_ids,
                        self.split,
                        self.train_split_percent,
                        self.validation_split_percent,
                        self.split_seed,
                    )
                    if not np.any(keep):
                        continue
                    arrays = (
                        handle["states"][start:end][keep],
                        handle["actions"][start:end][keep],
                        handle["values"][start:end][keep],
                    )
                    buffer.append(arrays)
                    if len(buffer) >= self.shuffle_buffer_chunks:
                        index = int(rng.integers(0, len(buffer))) if self.shuffle else 0
                        yield from emit(buffer.pop(index))

        while buffer:
            index = int(rng.integers(0, len(buffer))) if self.shuffle else 0
            yield from emit(buffer.pop(index))
