from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .config import StatsConfig


@dataclass(slots=True)
class RunReference:
    id: str
    name: str
    path: Path
    kind: str

    def public(self) -> dict[str, str]:
        return {"id": self.id, "name": self.name, "path": str(self.path), "kind": self.kind}


def _id(path: Path, kind: str) -> str:
    return hashlib.sha256(f"{kind}:{path.resolve()}".encode("utf-8")).hexdigest()[:20]


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
                if isinstance(value, dict):
                    records.append(value)
            except json.JSONDecodeError:
                continue
    return records


def _downsample(points: list[dict[str, Any]], maximum: int) -> list[dict[str, Any]]:
    if len(points) <= maximum:
        return points
    stride = max(1, len(points) // maximum)
    sampled = points[::stride]
    if sampled[-1] != points[-1]:
        sampled.append(points[-1])
    return sampled[: maximum - 1] + [points[-1]] if len(sampled) > maximum else sampled


def _unique_existing(paths: Iterable[str]) -> list[Path]:
    result: dict[str, Path] = {}
    for text in paths:
        path = Path(text).expanduser()
        if path.exists():
            result[str(path.resolve())] = path
    return list(result.values())


class StatsService:
    def __init__(self, config: StatsConfig) -> None:
        self.config = config

    def bootstrap_runs(self) -> list[RunReference]:
        found: dict[str, RunReference] = {}
        for root in _unique_existing(self.config.bootstrap_roots):
            candidates: set[Path] = set()
            if (root / "tensorboard").is_dir():
                candidates.add(root)
            if root.is_dir():
                for event in root.glob("**/tensorboard/events.out.tfevents.*"):
                    candidates.add(event.parent.parent)
            for run in candidates:
                ref = RunReference(
                    id=_id(run, "bootstrap"),
                    name=f"Bootstrap · {run.name}",
                    path=run,
                    kind="bootstrap",
                )
                found[ref.id] = ref
        return sorted(found.values(), key=lambda item: item.name.lower())

    def self_learning_runs(self) -> list[RunReference]:
        found: dict[str, RunReference] = {}
        for root in _unique_existing(self.config.self_learning_roots):
            candidates: set[Path] = set()
            if (
                (root / "history.json").exists()
                or (root / "run_state.json").exists()
                or (root / "iterations").is_dir()
            ):
                candidates.add(root)
            if root.is_dir():
                for history in root.glob("**/history.json"):
                    candidates.add(history.parent)
            for run in candidates:
                ref = RunReference(
                    id=_id(run, "self-learning"),
                    name=f"Self-learning · {run.name}",
                    path=run,
                    kind="self-learning",
                )
                found[ref.id] = ref
        return sorted(found.values(), key=lambda item: item.name.lower())

    def overview(self) -> dict[str, Any]:
        bootstrap = self.bootstrap_runs()
        selflearn = self.self_learning_runs()
        return {
            "bootstrap_runs": [run.public() for run in bootstrap],
            "self_learning_runs": [run.public() for run in selflearn],
        }

    def _resolve(self, run_id: str, kind: str) -> RunReference:
        runs = self.bootstrap_runs() if kind == "bootstrap" else self.self_learning_runs()
        for run in runs:
            if run.id == run_id:
                return run
        raise KeyError(f"Unknown {kind} run: {run_id}")

    def bootstrap_detail(self, run_id: str) -> dict[str, Any]:
        run = self._resolve(run_id, "bootstrap")
        try:
            from tensorboard.backend.event_processing.event_accumulator import (
                EventAccumulator,
            )
        except ImportError as exc:
            raise RuntimeError(
                "TensorBoard is required to read bootstrap scalar logs"
            ) from exc
        event_dir = run.path / "tensorboard"
        accumulator = EventAccumulator(str(event_dir), size_guidance={"scalars": 0})
        accumulator.Reload()
        series: dict[str, list[dict[str, Any]]] = {}
        for tag in accumulator.Tags().get("scalars", []):
            points = [
                {"step": int(event.step), "value": float(event.value), "wall_time": event.wall_time}
                for event in accumulator.Scalars(tag)
            ]
            series[tag] = _downsample(points, self.config.max_points_per_series)
        latest = {
            tag: values[-1] if values else None
            for tag, values in series.items()
        }
        metadata: dict[str, Any] = {}
        latest_path = run.path / "latest.json"
        if latest_path.exists():
            metadata["latest_checkpoint"] = _read_json(latest_path, {})
        return {"run": run.public(), "series": series, "latest": latest, "metadata": metadata}

    def self_learning_detail(self, run_id: str) -> dict[str, Any]:
        run = self._resolve(run_id, "self-learning")
        history = _read_json(run.path / "history.json", {"iterations": []}) or {"iterations": []}
        state = _read_json(run.path / "run_state.json", {}) or {}
        history_iterations = history.get("iterations", []) if isinstance(history, dict) else []
        entries: dict[int, dict[str, Any]] = {
            int(entry.get("iteration", 0)): dict(entry)
            for entry in history_iterations
            if int(entry.get("iteration", 0)) > 0
        }

        iterations_root = run.path / "iterations"
        if iterations_root.is_dir():
            for directory in iterations_root.glob("iteration_*"):
                if not directory.is_dir():
                    continue
                try:
                    iteration = int(directory.name.rsplit("_", 1)[1])
                except (IndexError, ValueError):
                    continue
                entry = entries.setdefault(iteration, {"iteration": iteration})
                entry.setdefault("selfplay", _read_json(directory / "selfplay_summary.json", {}) or {})
                entry.setdefault("candidate", _read_json(directory / "candidate_summary.json", {}) or {})
                entry.setdefault("arena", _read_json(directory / "arena_summary.json", {}) or {})

        summaries: list[dict[str, Any]] = []
        for iteration in sorted(entries):
            entry = entries[iteration]
            selfplay = entry.get("selfplay", {}) or {}
            arena = entry.get("arena", {}) or {}
            candidate = entry.get("candidate", {}) or {}
            search = selfplay.get("search", {}) or {}
            evaluator = selfplay.get("evaluator", {}) or {}
            metrics_path = (
                run.path
                / "iterations"
                / f"iteration_{iteration:06d}"
                / "candidate_metrics.jsonl"
            )
            candidate_metrics = _read_jsonl(metrics_path)
            latest_candidate_metric = candidate_metrics[-1] if candidate_metrics else None
            games = int(selfplay.get("games", 0) or 0)
            positions = int(selfplay.get("positions", 0) or 0)
            arena_complete = arena.get("score") is not None
            candidate_complete = bool(candidate)
            selfplay_complete = bool(selfplay)
            status = (
                "complete"
                if arena_complete
                else "candidate training"
                if candidate_metrics or candidate_complete
                else "self-play complete"
                if selfplay_complete
                else "in progress"
            )
            summaries.append(
                {
                    "iteration": iteration,
                    "status": status,
                    "generation": selfplay.get("generation"),
                    "games": games,
                    "positions": positions,
                    "positions_per_game": positions / games if games else 0.0,
                    "positions_per_second": float(selfplay.get("positions_per_second", 0.0) or 0.0),
                    "elapsed_seconds": float(selfplay.get("elapsed_seconds", 0.0) or 0.0),
                    "simulations_per_second": float(search.get("simulations_per_second", 0.0) or 0.0),
                    "mean_evaluation_batch": float(search.get("mean_evaluation_batch", 0.0) or 0.0),
                    "mean_unique_batch": float(evaluator.get("mean_unique_batch", 0.0) or 0.0),
                    "duplicate_fraction": float(evaluator.get("duplicate_fraction", 0.0) or 0.0),
                    "results": selfplay.get("results", {}),
                    "terminations": selfplay.get("terminations", {}),
                    "candidate_generation": candidate.get("candidate_generation"),
                    "candidate_step": (
                        latest_candidate_metric.get("step")
                        if latest_candidate_metric
                        else candidate.get("steps")
                    ),
                    "latest_candidate_loss": (
                        latest_candidate_metric.get("loss")
                        if latest_candidate_metric
                        else None
                    ),
                    "arena_games": arena.get("games", 0),
                    "wins": arena.get("wins", 0),
                    "draws": arena.get("draws", 0),
                    "losses": arena.get("losses", 0),
                    "arena_score": arena.get("score"),
                    "promotion_threshold": arena.get("promotion_threshold"),
                    "promoted": (
                        bool(arena.get("promoted"))
                        if arena_complete
                        else None
                    ),
                }
            )
        return {"run": run.public(), "state": state, "iterations": summaries}

    def iteration_detail(self, run_id: str, iteration: int) -> dict[str, Any]:
        run = self._resolve(run_id, "self-learning")
        root = run.path / "iterations" / f"iteration_{iteration:06d}"
        if not root.exists():
            raise KeyError(f"Iteration {iteration} does not exist")
        return {
            "run": run.public(),
            "iteration": iteration,
            "candidate_metrics": _downsample(
                _read_jsonl(root / "candidate_metrics.jsonl"),
                self.config.max_points_per_series,
            ),
            "selfplay": _read_json(root / "selfplay_summary.json", {}),
            "candidate": _read_json(root / "candidate_summary.json", {}),
            "arena": _read_json(root / "arena_summary.json", {}),
        }
