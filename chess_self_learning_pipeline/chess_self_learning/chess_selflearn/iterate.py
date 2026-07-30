from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .arena import evaluate_candidate
from .checkpoint import atomic_copy
from .config import AppConfig, load_config
from .self_play import generate_self_play
from .train_candidate import train_candidate


def initialize_run(config: AppConfig) -> Path:
    run_root = Path(config.run.root)
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "iterations").mkdir(exist_ok=True)
    champion = run_root / "champion.pt"
    state_path = run_root / "run_state.json"
    if not champion.exists():
        initial = Path(config.run.initial_checkpoint)
        if not initial.exists():
            raise FileNotFoundError(f"Initial checkpoint does not exist: {initial}")
        atomic_copy(initial, champion)
        state = {
            "champion_checkpoint": str(champion.resolve()),
            "source_checkpoint": str(initial.resolve()),
            "last_completed_iteration": 0,
            "promotions": 0,
        }
        write_json_atomic(state_path, state)
    elif not state_path.exists():
        write_json_atomic(
            state_path,
            {
                "champion_checkpoint": str(champion.resolve()),
                "source_checkpoint": "unknown_existing_champion",
                "last_completed_iteration": 0,
                "promotions": 0,
            },
        )
    return champion


def write_json_atomic(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def next_iteration(run_root: Path) -> int:
    state_path = run_root / "run_state.json"
    if not state_path.exists():
        return 1
    state = json.loads(state_path.read_text(encoding="utf-8"))
    return int(state.get("last_completed_iteration", 0)) + 1


def update_history(
    run_root: Path,
    iteration: int,
    selfplay: dict[str, object],
    candidate: dict[str, object],
    arena: dict[str, object],
) -> None:
    history_path = run_root / "history.json"
    if history_path.exists():
        history = json.loads(history_path.read_text(encoding="utf-8"))
    else:
        history = {"iterations": []}
    history["iterations"] = [
        entry
        for entry in history["iterations"]
        if int(entry["iteration"]) != iteration
    ]
    history["iterations"].append(
        {
            "iteration": iteration,
            "selfplay": selfplay,
            "candidate": candidate,
            "arena": arena,
        }
    )
    history["iterations"].sort(key=lambda entry: int(entry["iteration"]))
    write_json_atomic(history_path, history)


def run_iterations(config: AppConfig, count: int) -> None:
    champion = initialize_run(config)
    run_root = Path(config.run.root)
    iteration = next_iteration(run_root)

    for _ in range(count):
        print(f"\n=== Self-learning iteration {iteration} ===")
        selfplay_summary = generate_self_play(
            config,
            iteration=iteration,
            checkpoint_path=champion,
        )
        candidate_summary = train_candidate(
            config,
            iteration=iteration,
            champion_checkpoint=champion,
        )
        candidate_path = Path(str(candidate_summary["candidate_path"]))
        arena_summary = evaluate_candidate(
            config,
            iteration=iteration,
            champion_checkpoint=champion,
            candidate_checkpoint=candidate_path,
        )

        if bool(arena_summary["promoted"]):
            atomic_copy(candidate_path, champion)
            print(
                f"Promoted generation {arena_summary['candidate_generation']} "
                f"with score {arena_summary['score']:.3f}."
            )
        else:
            print(
                f"Candidate rejected at score {arena_summary['score']:.3f}; "
                f"champion remains generation "
                f"{arena_summary['champion_generation']}."
            )

        update_history(
            run_root,
            iteration,
            selfplay_summary,
            candidate_summary,
            arena_summary,
        )
        state_path = run_root / "run_state.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state["last_completed_iteration"] = iteration
        if bool(arena_summary["promoted"]):
            state["promotions"] = int(state.get("promotions", 0)) + 1
        state["last_arena_score"] = arena_summary["score"]
        write_json_atomic(state_path, state)
        iteration += 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("selflearn_config.yaml"))
    parser.add_argument("--iterations", type=int, default=1)
    args = parser.parse_args()
    if args.iterations <= 0:
        raise ValueError("--iterations must be positive")
    run_iterations(load_config(args.config), args.iterations)


if __name__ == "__main__":
    main()
