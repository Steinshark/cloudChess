from __future__ import annotations

import gc
import hashlib
import threading
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from chess_selflearn.checkpoint import checkpoint_model_config, load_model
from chess_selflearn.evaluator import EvaluatorConfig, NeuralEvaluator

from .config import ModelScanConfig


@dataclass(slots=True)
class ModelDescriptor:
    id: str
    name: str
    path: str
    relative_path: str
    family: str
    size_bytes: int
    modified_at: str
    residual_blocks: int | None = None
    channels: int | None = None
    generation: int | None = None
    step: int | None = None
    best_validation_loss: float | None = None
    inspect_error: str | None = None

    def public(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class LoadedModel:
    descriptor: ModelDescriptor
    evaluator: NeuralEvaluator
    payload: dict[str, Any]
    loaded_at: float


def _model_id(path: Path) -> str:
    return hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:20]


def _iso_timestamp(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def _family(path: Path) -> str:
    lowered = [part.lower() for part in path.parts]
    if "iterations" in lowered or path.name.lower() == "champion.pt":
        return "self-learning"
    if path.name.lower() == "best.pt" or any("bootstrap" in p for p in lowered):
        return "bootstrap"
    return "checkpoint"


def _friendly_name(path: Path, root: Path, payload: dict[str, Any]) -> str:
    family = _family(path)
    generation = payload.get("generation")
    step = payload.get("step")
    try:
        relative = path.relative_to(root)
    except ValueError:
        relative = path.name
    parts = list(relative.parts) if isinstance(relative, Path) else [str(relative)]
    label = " / ".join(parts[-3:])
    suffixes: list[str] = []
    if generation is not None:
        suffixes.append(f"gen {generation}")
    if step is not None:
        suffixes.append(f"step {int(step):,}")
    suffix = f" · {', '.join(suffixes)}" if suffixes else ""
    return f"{family.title()} · {label}{suffix}"


def _safe_load_metadata(path: Path) -> dict[str, Any]:
    kwargs = {"map_location": "cpu", "weights_only": False}
    try:
        return torch.load(path, mmap=True, **kwargs)
    except (TypeError, RuntimeError, ValueError):
        return torch.load(path, **kwargs)


class ModelRegistry:
    def __init__(self, config: ModelScanConfig) -> None:
        self.config = config
        self._lock = threading.RLock()
        self._models: dict[str, ModelDescriptor] = {}
        self._inspection_cache: dict[str, tuple[int, int, ModelDescriptor]] = {}
        self._last_scan = 0.0

    def _candidate_paths(self) -> list[tuple[Path, Path]]:
        found: dict[str, tuple[Path, Path]] = {}
        for root_text in self.config.roots:
            root = Path(root_text).expanduser()
            if not root.exists():
                continue
            if root.is_file() and root.suffix.lower() == ".pt":
                found[str(root.resolve())] = (root, root.parent)
                continue
            for pattern in self.config.patterns:
                for path in root.glob(pattern):
                    if not path.is_file() or path.suffix.lower() != ".pt":
                        continue
                    lowered = path.name.lower()
                    if any(token.lower() in lowered for token in self.config.exclude_name_contains):
                        continue
                    found[str(path.resolve())] = (path, root)
        return sorted(found.values(), key=lambda pair: str(pair[0]).lower())

    def refresh(self, force: bool = False) -> list[ModelDescriptor]:
        with self._lock:
            now = time.monotonic()
            if (
                not force
                and self._models
                and now - self._last_scan < self.config.refresh_seconds
            ):
                return self.list()

            descriptors: dict[str, ModelDescriptor] = {}
            live_paths: set[str] = set()
            for path, root in self._candidate_paths():
                stat = path.stat()
                cache_key = str(path.resolve())
                live_paths.add(cache_key)
                stamp = (stat.st_mtime_ns, stat.st_size)
                cached = self._inspection_cache.get(cache_key)
                if cached is not None and cached[:2] == stamp:
                    descriptors[cached[2].id] = cached[2]
                    continue

                model_id = _model_id(path)
                try:
                    payload = _safe_load_metadata(path)
                    if "model" not in payload:
                        raise ValueError("Checkpoint has no 'model' state dictionary")
                    model_config = checkpoint_model_config(payload)
                    try:
                        relative = str(path.relative_to(root))
                    except ValueError:
                        relative = path.name
                    descriptor = ModelDescriptor(
                        id=model_id,
                        name=_friendly_name(path, root, payload),
                        path=str(path.resolve()),
                        relative_path=relative,
                        family=_family(path),
                        size_bytes=stat.st_size,
                        modified_at=_iso_timestamp(stat.st_mtime),
                        residual_blocks=model_config.residual_blocks,
                        channels=model_config.channels,
                        generation=(
                            int(payload["generation"])
                            if payload.get("generation") is not None
                            else None
                        ),
                        step=(
                            int(payload["step"])
                            if payload.get("step") is not None
                            else None
                        ),
                        best_validation_loss=(
                            float(payload["best_validation_loss"])
                            if payload.get("best_validation_loss") is not None
                            else None
                        ),
                    )
                    del payload
                    gc.collect()
                except Exception as exc:  # keep bad files visible but unselectable
                    descriptor = ModelDescriptor(
                        id=model_id,
                        name=f"Unreadable · {path.name}",
                        path=str(path.resolve()),
                        relative_path=path.name,
                        family=_family(path),
                        size_bytes=stat.st_size,
                        modified_at=_iso_timestamp(stat.st_mtime),
                        inspect_error=f"{type(exc).__name__}: {exc}",
                    )
                descriptors[descriptor.id] = descriptor
                self._inspection_cache[cache_key] = (stamp[0], stamp[1], descriptor)
            self._inspection_cache = {
                key: value for key, value in self._inspection_cache.items() if key in live_paths
            }
            self._models = descriptors
            self._last_scan = now
            return self.list()

    def list(self) -> list[ModelDescriptor]:
        with self._lock:
            return sorted(
                self._models.values(),
                key=lambda item: (
                    item.inspect_error is not None,
                    item.family,
                    item.name.lower(),
                ),
            )

    def get(self, model_id: str) -> ModelDescriptor:
        self.refresh()
        with self._lock:
            descriptor = self._models.get(model_id)
            if descriptor is None:
                raise KeyError(f"Unknown model id: {model_id}")
            if descriptor.inspect_error:
                raise RuntimeError(
                    f"Checkpoint cannot be loaded: {descriptor.inspect_error}"
                )
            return descriptor


class ModelCache:
    def __init__(self, registry: ModelRegistry, config: ModelScanConfig) -> None:
        self.registry = registry
        self.config = config
        self._lock = threading.RLock()
        self._loaded: OrderedDict[str, LoadedModel] = OrderedDict()
        self.device = self._resolve_device(config.device)

    @staticmethod
    def _resolve_device(requested: str) -> torch.device:
        if requested == "cpu":
            return torch.device("cpu")
        if requested == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("models.device is cuda but CUDA is unavailable")
            return torch.device("cuda")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get(self, model_id: str) -> LoadedModel:
        with self._lock:
            cached = self._loaded.pop(model_id, None)
            if cached is not None:
                self._loaded[model_id] = cached
                return cached

            descriptor = self.registry.get(model_id)
            model, full_payload = load_model(
                descriptor.path,
                self.device,
                channels_last=self.config.channels_last,
            )
            # Training checkpoints can contain multiple copies of the model in
            # optimizer/scaler state. Keep only small provenance metadata after
            # the weights have been loaded onto the selected device.
            payload = {
                key: value
                for key, value in full_payload.items()
                if key not in {"model", "optimizer", "scheduler", "scaler", "rng_state"}
            }
            del full_payload
            gc.collect()
            precision = self.config.precision if self.device.type == "cuda" else "fp32"
            evaluator = NeuralEvaluator(
                model,
                self.device,
                EvaluatorConfig(
                    precision=precision,
                    channels_last=self.config.channels_last,
                    max_batch_size=1,
                ),
            )
            loaded = LoadedModel(
                descriptor=descriptor,
                evaluator=evaluator,
                payload=payload,
                loaded_at=time.time(),
            )
            self._loaded[model_id] = loaded
            self._evict_if_needed()
            return loaded

    def _evict_if_needed(self) -> None:
        while len(self._loaded) > self.config.max_loaded_models:
            _, removed = self._loaded.popitem(last=False)
            del removed
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "device": str(self.device),
                "loaded_model_ids": list(self._loaded),
                "max_loaded_models": self.config.max_loaded_models,
            }
