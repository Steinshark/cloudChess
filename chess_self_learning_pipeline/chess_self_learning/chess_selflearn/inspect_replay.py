from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = args.run_root / "replay_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    total_games = 0
    total_positions = 0

    for entry in manifest["iterations"]:
        path = Path(entry["path"])
        with h5py.File(path, "r") as handle:
            positions = int(handle["states"].shape[0])
            counts = handle["policy_count"][:]
            values = handle["values"][:]
            assert handle["states"].shape[1:] == (34, 8, 8)
            assert int(counts.max(initial=0)) <= int(handle.attrs["max_policy_moves"])
            assert set(np.unique(values)).issubset({-1, 0, 1})
            assert np.all(handle["policy_indices"][:] < 65536)
            total_games += int(entry["games"])
            total_positions += positions
            print(
                f"iteration={entry['iteration']} games={entry['games']} "
                f"positions={positions} mean_policy_moves={counts.mean():.1f}"
            )

    print(f"Total games: {total_games:,}")
    print(f"Total positions: {total_positions:,}")


if __name__ == "__main__":
    main()
