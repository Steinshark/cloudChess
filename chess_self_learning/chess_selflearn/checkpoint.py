from __future__ import annotations

import os
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .model import ChessPolicyValueNet, ModelConfig


def checkpoint_model_config(payload: dict[str, Any]) -> ModelConfig:
    raw = payload.get("config", {}).get("model", {})
    allowed = set(ModelConfig.__dataclass_fields__)
    filtered = {key: value for key, value in raw.items() if key in allowed}
    return ModelConfig(**filtered)


def load_model(
    path: str | Path,
    device: torch.device,
    channels_last: bool = True,
) -> tuple[ChessPolicyValueNet, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint_model_config(payload)
    model = ChessPolicyValueNet(config)
    model.load_state_dict(payload["model"])
    model.to(device)
    if channels_last:
        model.to(memory_format=torch.channels_last)
    model.eval()
    return model, payload


def save_model_checkpoint(
    path: str | Path,
    model: nn.Module,
    model_config: ModelConfig,
    *,
    generation: int,
    source_checkpoint: str,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: object | None = None,
    step: int = 0,
    metadata: dict[str, Any] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    original = getattr(model, "_orig_mod", model)
    payload: dict[str, Any] = {
        "format_version": 2,
        "generation": generation,
        "step": step,
        "model": original.state_dict(),
        "config": {"model": asdict(model_config)},
        "source_checkpoint": source_checkpoint,
        "metadata": metadata or {},
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None and hasattr(scheduler, "state_dict"):
        payload["scheduler"] = scheduler.state_dict()

    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return path


def atomic_copy(source: str | Path, destination: str | Path) -> Path:
    source = Path(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)
    return destination
