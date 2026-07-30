from __future__ import annotations
import argparse
from dataclasses import asdict
from pathlib import Path
import torch
from .checkpoint import load_checkpoint
from .config import load_config
from .model import ChessPolicyValueNet, ModelConfig
from .train import create_loader, validate

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--batches", type=int)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.batches:
        cfg.training.validation_batches = args.batches
    device = torch.device("cuda")
    model = ChessPolicyValueNet(ModelConfig(**asdict(cfg.model))).to(device)
    load_checkpoint(args.checkpoint, model)
    loader = create_loader(cfg, args.split, False)
    print(validate(model, loader, device, cfg))

if __name__ == "__main__":
    main()
