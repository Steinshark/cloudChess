from __future__ import annotations
import argparse, json
from pathlib import Path
import h5py, numpy as np
from .data import split_mask

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--sample-shards", type=int, default=3)
    args = parser.parse_args()
    manifest = json.loads(
        (args.dataset_root / "manifest.json").read_text(encoding="utf-8")
    )
    print(json.dumps({
        "total_positions": manifest["total_positions"],
        "total_games": manifest["total_games"],
        "state_shape": manifest["state_shape"],
        "action_space_size": manifest["action_space_size"],
        "shards": len(manifest["shards"]),
    }, indent=2))
    counts = {"train": 0, "validation": 0, "test": 0}
    for shard in manifest["shards"][:args.sample_shards]:
        with h5py.File(args.dataset_root / shard["filename"], "r") as f:
            ids = f["game_ids"][:]
            for split in counts:
                counts[split] += int(split_mask(ids, split, 98, 1, 7349).sum())
            assert int(f["actions"][:].max()) < 4672
            assert set(np.unique(f["values"][:])).issubset({-1, 0, 1})
    print("Sampled split counts:", counts)

if __name__ == "__main__":
    main()
