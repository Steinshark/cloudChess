from __future__ import annotations
import argparse, contextlib, json, math, random, time
from dataclasses import asdict
from pathlib import Path
from typing import Iterator
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .checkpoint import load_checkpoint, prune, save_checkpoint
from .config import AppConfig, load_config
from .data import H5ChessIterableDataset
from .model import ChessPolicyValueNet, ModelConfig

def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def create_loader(cfg: AppConfig, split: str, shuffle: bool):
    dataset = H5ChessIterableDataset(
        cfg.data.dataset_root,
        split,
        cfg.data.train_split_percent,
        cfg.data.validation_split_percent,
        cfg.data.split_seed,
        cfg.data.read_chunk_size,
        cfg.data.shuffle_buffer_chunks,
        cfg.seed + (0 if split == "train" else 10000),
        shuffle,
    )
    kwargs = dict(
        dataset=dataset,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        drop_last=(split == "train"),
    )
    if cfg.data.num_workers > 0:
        kwargs["prefetch_factor"] = cfg.data.prefetch_factor
        kwargs["persistent_workers"] = False
    return DataLoader(**kwargs)

def endless(loader) -> Iterator:
    while True:
        found = False
        for batch in loader:
            found = True
            yield batch
        if not found:
            raise RuntimeError("Dataset split produced no samples")

def autocast_ctx(device, precision):
    if device.type != "cuda" or precision == "fp32":
        return contextlib.nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast("cuda", dtype=dtype)

def move_batch(batch, device, channels_last):
    states, actions, values = batch
    states = states.to(device, dtype=torch.float32, non_blocking=True).div_(255.0)
    if channels_last:
        states = states.contiguous(memory_format=torch.channels_last)
    return (
        states,
        actions.to(device, dtype=torch.long, non_blocking=True),
        values.to(device, dtype=torch.float32, non_blocking=True),
    )

def losses(policy, value_pred, actions, values, cfg):
    policy_loss = F.cross_entropy(
        policy, actions, label_smoothing=cfg.training.label_smoothing
    )
    value_loss = F.smooth_l1_loss(value_pred, values)
    total = (
        cfg.training.policy_loss_weight * policy_loss
        + cfg.training.value_loss_weight * value_loss
    )
    return total, policy_loss, value_loss

def make_scheduler(optimizer, cfg):
    warmup = cfg.training.warmup_steps
    total = cfg.training.max_steps
    min_ratio = cfg.training.minimum_learning_rate / cfg.training.learning_rate
    def scale(step):
        if warmup and step < warmup:
            return max(1e-8, (step + 1) / warmup)
        progress = min(1.0, max(0.0, (step - warmup) / max(1, total - warmup)))
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, scale)

@torch.inference_mode()
def validate(model, loader, device, cfg):
    model.eval()
    totals = dict(loss=0.0, policy=0.0, value=0.0, top1=0, top3=0, top5=0, count=0)
    for batch_index, batch in enumerate(loader):
        if batch_index >= cfg.training.validation_batches:
            break
        states, actions, values = move_batch(
            batch, device, cfg.training.channels_last
        )
        with autocast_ctx(device, cfg.training.precision):
            policy, value_pred = model(states)
            total, policy_loss, value_loss = losses(
                policy, value_pred, actions, values, cfg
            )
        n = states.shape[0]
        top = policy.topk(5, dim=1).indices.eq(actions[:, None])
        totals["loss"] += float(total) * n
        totals["policy"] += float(policy_loss) * n
        totals["value"] += float(value_loss) * n
        totals["top1"] += int(top[:, :1].any(1).sum())
        totals["top3"] += int(top[:, :3].any(1).sum())
        totals["top5"] += int(top.any(1).sum())
        totals["count"] += n
    model.train()
    n = max(1, totals.pop("count"))
    return {key: value / n for key, value in totals.items()}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_all(cfg.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA-enabled PyTorch is required")
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    out = Path(cfg.training.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model = ChessPolicyValueNet(ModelConfig(**asdict(cfg.model))).to(device)
    if cfg.training.channels_last:
        model = model.to(memory_format=torch.channels_last)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        betas=(cfg.training.beta1, cfg.training.beta2),
        weight_decay=cfg.training.weight_decay,
        fused=True,
    )
    scheduler = make_scheduler(optimizer, cfg)
    scaler = torch.amp.GradScaler("cuda") if cfg.training.precision == "fp16" else None

    start_step = 0
    best = float("inf")
    resume = args.resume or (Path(cfg.training.resume_from) if cfg.training.resume_from else None)
    if resume:
        if resume.is_dir():
            latest = json.loads((resume / "latest.json").read_text(encoding="utf-8"))
            resume = resume / latest["checkpoint"]
        payload = load_checkpoint(resume, model, optimizer, scheduler, scaler)
        start_step = int(payload["step"])
        best = float(payload.get("best_validation_loss", best))
        print(f"Resumed at step {start_step:,}")

    if cfg.training.compile_model:
        model = torch.compile(model)

    train_loader = create_loader(cfg, "train", True)
    val_loader = create_loader(cfg, "validation", False)
    batches = endless(train_loader)
    writer = SummaryWriter(out / "tensorboard")
    optimizer.zero_grad(set_to_none=True)
    model.train()

    progress = tqdm(
        range(start_step + 1, cfg.training.max_steps + 1),
        initial=start_step,
        total=cfg.training.max_steps,
        dynamic_ncols=True,
    )
    running_loss = 0.0
    running_top1 = 0
    running_count = 0
    started = time.perf_counter()

    for step in progress:
        for _ in range(cfg.training.gradient_accumulation_steps):
            states, actions, values = move_batch(
                next(batches), device, cfg.training.channels_last
            )
            with autocast_ctx(device, cfg.training.precision):
                policy, value_pred = model(states)
                total, policy_loss, value_loss = losses(
                    policy, value_pred, actions, values, cfg
                )
                scaled = total / cfg.training.gradient_accumulation_steps
            if scaler:
                scaler.scale(scaled).backward()
            else:
                scaled.backward()

        if scaler:
            scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), cfg.training.gradient_clip_norm
        )
        if scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        n = states.shape[0]
        running_loss += float(total.detach()) * n
        running_top1 += int(policy.detach().argmax(1).eq(actions).sum())
        running_count += n

        if step % cfg.training.log_every_steps == 0:
            elapsed = time.perf_counter() - started
            avg_loss = running_loss / max(1, running_count)
            top1 = running_top1 / max(1, running_count)
            sps = running_count / max(elapsed, 1e-9)
            lr = optimizer.param_groups[0]["lr"]
            progress.set_postfix(loss=f"{avg_loss:.4f}", top1=f"{top1:.3f}", sps=f"{sps:,.0f}")
            writer.add_scalar("train/loss", avg_loss, step)
            writer.add_scalar("train/top1", top1, step)
            writer.add_scalar("train/lr", lr, step)
            writer.add_scalar("train/gradient_norm", float(grad_norm), step)
            writer.add_scalar("train/samples_per_second", sps, step)
            running_loss = 0.0
            running_top1 = 0
            running_count = 0
            started = time.perf_counter()

        val_metrics = None
        if step % cfg.training.validate_every_steps == 0:
            val_metrics = validate(model, val_loader, device, cfg)
            print("\nValidation:", val_metrics)
            for name, value in val_metrics.items():
                writer.add_scalar(f"validation/{name}", value, step)

        if (
            step % cfg.training.checkpoint_every_steps == 0
            or step == cfg.training.max_steps
        ):
            is_best = val_metrics is not None and val_metrics["loss"] < best
            if is_best:
                best = val_metrics["loss"]
            path = save_checkpoint(
                out, step, model, optimizer, scheduler, scaler,
                best, asdict(cfg), is_best
            )
            prune(out, cfg.training.keep_last_checkpoints)
            print(f"\nSaved {path}")

        writer.flush()

    writer.close()

if __name__ == "__main__":
    main()
