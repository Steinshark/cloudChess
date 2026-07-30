from __future__ import annotations
import json, os, random
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch import nn

def unwrap(model: nn.Module) -> nn.Module:
    return getattr(model, "_orig_mod", model)

def rng_state() -> dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state

def save_checkpoint(
    output_dir,
    step,
    model,
    optimizer,
    scheduler,
    scaler,
    best_validation_loss,
    config,
    is_best=False,
):
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "model": unwrap(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "best_validation_loss": best_validation_loss,
        "config": config,
        "rng_state": rng_state(),
    }
    path = directory / f"checkpoint_{step:08d}.pt"
    temp = path.with_suffix(".tmp")
    torch.save(payload, temp)
    os.replace(temp, path)
    (directory / "latest.json").write_text(
        json.dumps({"checkpoint": path.name, "step": step}, indent=2),
        encoding="utf-8",
    )
    if is_best:
        temp_best = directory / "best.tmp"
        torch.save(payload, temp_best)
        os.replace(temp_best, directory / "best.pt")
    return path

def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    unwrap(model).load_state_dict(payload["model"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None and payload.get("scaler") is not None:
        scaler.load_state_dict(payload["scaler"])
    return payload

def prune(output_dir, keep_last: int):
    paths = sorted(Path(output_dir).glob("checkpoint_*.pt"))
    for path in paths[:-keep_last]:
        path.unlink(missing_ok=True)
