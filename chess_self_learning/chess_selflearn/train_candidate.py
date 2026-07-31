from __future__ import annotations

import argparse
import contextlib
import json
import math
import random
import time
from pathlib import Path
from typing import Iterator

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .checkpoint import load_model, save_model_checkpoint
from .config import AppConfig, load_config
from .losses import sparse_policy_loss
from .replay import (
    BootstrapIterableDataset,
    ReplayIterableDataset,
    replay_files,
)


def autocast_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return contextlib.nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def create_loader(dataset, config: AppConfig) -> DataLoader:
    kwargs: dict[str, object] = {
        "dataset": dataset,
        "batch_size": config.training.batch_size,
        "num_workers": config.training.num_workers,
        "pin_memory": True,
        "drop_last": True,
    }
    if config.training.num_workers > 0:
        kwargs["prefetch_factor"] = 2
        kwargs["persistent_workers"] = False
    return DataLoader(**kwargs)


def endless(loader: DataLoader) -> Iterator:
    while True:
        found = False
        for batch in loader:
            found = True
            yield batch
        if not found:
            raise RuntimeError("A training loader yielded no batches")


def move_states(
    states: Tensor,
    device: torch.device,
    channels_last: bool,
) -> Tensor:
    states = states.to(
        device,
        dtype=torch.float32,
        non_blocking=True,
    ).div_(255.0)
    if channels_last:
        states = states.contiguous(memory_format=torch.channels_last)
    return states


def create_scheduler(optimizer: AdamW, config: AppConfig) -> LambdaLR:
    total = config.training.steps_per_iteration
    warmup = config.training.warmup_steps
    minimum_ratio = (
        config.training.minimum_learning_rate
        / config.training.learning_rate
    )

    def multiplier(step: int) -> float:
        if warmup and step < warmup:
            return max(1e-8, (step + 1) / warmup)
        progress = min(
            1.0,
            max(0.0, (step - warmup) / max(1, total - warmup)),
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return minimum_ratio + (1.0 - minimum_ratio) * cosine

    return LambdaLR(optimizer, multiplier)


def train_candidate(
    config: AppConfig,
    *,
    iteration: int,
    champion_checkpoint: str | Path,
) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA-enabled PyTorch is required")

    run_root = Path(config.run.root)
    iteration_root = run_root / "iterations" / f"iteration_{iteration:06d}"
    iteration_root.mkdir(parents=True, exist_ok=True)
    candidate_path = iteration_root / "candidate.pt"

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    model, champion_payload = load_model(
        champion_checkpoint,
        device,
        channels_last=config.training.channels_last,
    )
    model.train()
    model_config = model.config
    champion_generation = int(champion_payload.get("generation", 0))
    candidate_generation = champion_generation + 1

    replay_dataset = ReplayIterableDataset(
        replay_files(
            run_root,
            keep_iterations=config.training.replay_keep_iterations,
        ),
        read_chunk_size=config.training.read_chunk_size,
        seed=config.seed + iteration * 17,
    )
    replay_iterator = endless(create_loader(replay_dataset, config))

    bootstrap_iterator = None
    if (
        config.training.bootstrap_dataset_root
        and config.training.bootstrap_mix_ratio > 0.0
    ):
        bootstrap_dataset = BootstrapIterableDataset(
            config.training.bootstrap_dataset_root,
            read_chunk_size=config.training.read_chunk_size,
            seed=config.seed + iteration * 19,
        )
        bootstrap_iterator = endless(create_loader(bootstrap_dataset, config))

    optimizer = AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        fused=True,
    )
    scheduler = create_scheduler(optimizer, config)
    scaler = (
        torch.amp.GradScaler("cuda")
        if config.training.precision == "fp16"
        else None
    )
    rng = random.Random(config.seed + iteration * 23)
    writer = SummaryWriter(iteration_root / "candidate_tensorboard")
    metrics_path = iteration_root / "candidate_metrics.jsonl"

    started = time.perf_counter()
    running = {
        "loss": 0.0,
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "top1": 0.0,
        "samples": 0,
        "selfplay_batches": 0,
        "bootstrap_batches": 0,
    }

    progress = tqdm(
        range(1, config.training.steps_per_iteration + 1),
        desc="Candidate training",
        dynamic_ncols=True,
    )

    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        for step in progress:
            use_bootstrap = (
                bootstrap_iterator is not None
                and rng.random() < config.training.bootstrap_mix_ratio
            )

            optimizer.zero_grad(set_to_none=True)

            if use_bootstrap:
                states, actions, values = next(bootstrap_iterator)
                states = move_states(
                    states,
                    device,
                    config.training.channels_last,
                )
                actions = actions.to(device, non_blocking=True)
                values = values.to(device, non_blocking=True)

                with autocast_context(device, config.training.precision):
                    logits, predicted_values = model(states)
                    policy_loss = F.cross_entropy(logits, actions)
                    value_loss = F.smooth_l1_loss(predicted_values, values)
                    loss = (
                        policy_loss
                        + config.training.value_loss_weight * value_loss
                    )
                target_actions = actions
                running["bootstrap_batches"] += 1
            else:
                (
                    states,
                    policy_indices,
                    policy_probabilities,
                    policy_counts,
                    values,
                ) = next(replay_iterator)
                states = move_states(
                    states,
                    device,
                    config.training.channels_last,
                )
                values = values.to(device, non_blocking=True)

                with autocast_context(device, config.training.precision):
                    logits, predicted_values = model(states)
                    policy_loss = sparse_policy_loss(
                        logits,
                        policy_indices,
                        policy_probabilities,
                        policy_counts,
                    )
                    value_loss = F.smooth_l1_loss(predicted_values, values)
                    loss = (
                        policy_loss
                        + config.training.value_loss_weight * value_loss
                    )

                best_slots = policy_probabilities.argmax(dim=1)
                target_actions = policy_indices.gather(
                    1,
                    best_slots[:, None],
                ).squeeze(1).to(device, non_blocking=True)
                running["selfplay_batches"] += 1

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()

            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.training.gradient_clip_norm,
            )

            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()

            batch_size = states.shape[0]
            top1 = logits.detach().argmax(dim=1).eq(target_actions).float().sum()
            running["loss"] += float(loss.detach()) * batch_size
            running["policy_loss"] += float(policy_loss.detach()) * batch_size
            running["value_loss"] += float(value_loss.detach()) * batch_size
            running["top1"] += float(top1)
            running["samples"] += batch_size

            if step % 50 == 0:
                samples = max(1, int(running["samples"]))
                elapsed = max(time.perf_counter() - started, 1e-9)
                record = {
                    "step": step,
                    "loss": running["loss"] / samples,
                    "policy_loss": running["policy_loss"] / samples,
                    "value_loss": running["value_loss"] / samples,
                    "top1": running["top1"] / samples,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "gradient_norm": float(gradient_norm),
                    "samples_per_second": samples / elapsed,
                    "selfplay_batches": running["selfplay_batches"],
                    "bootstrap_batches": running["bootstrap_batches"],
                }
                progress.set_postfix(
                    loss=f"{record['loss']:.4f}",
                    top1=f"{record['top1']:.3f}",
                    sps=f"{record['samples_per_second']:,.0f}",
                )
                for key, value in record.items():
                    if isinstance(value, (int, float)) and key != "step":
                        writer.add_scalar(f"train/{key}", value, step)
                metrics_file.write(json.dumps(record) + "\n")
                metrics_file.flush()
                running = {
                    "loss": 0.0,
                    "policy_loss": 0.0,
                    "value_loss": 0.0,
                    "top1": 0.0,
                    "samples": 0,
                    "selfplay_batches": 0,
                    "bootstrap_batches": 0,
                }
                started = time.perf_counter()

            if step % config.training.checkpoint_every_steps == 0:
                save_model_checkpoint(
                    iteration_root / f"candidate_step_{step:06d}.pt",
                    model,
                    model_config,
                    generation=candidate_generation,
                    source_checkpoint=str(Path(champion_checkpoint).resolve()),
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step=step,
                    metadata={"iteration": iteration},
                )

    writer.close()
    save_model_checkpoint(
        candidate_path,
        model,
        model_config,
        generation=candidate_generation,
        source_checkpoint=str(Path(champion_checkpoint).resolve()),
        optimizer=optimizer,
        scheduler=scheduler,
        step=config.training.steps_per_iteration,
        metadata={"iteration": iteration},
    )

    summary = {
        "iteration": iteration,
        "champion_generation": champion_generation,
        "candidate_generation": candidate_generation,
        "steps": config.training.steps_per_iteration,
        "candidate_path": str(candidate_path.resolve()),
    }
    (iteration_root / "candidate_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("selflearn_config.yaml"))
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--champion", type=Path, required=True)
    args = parser.parse_args()

    summary = train_candidate(
        load_config(args.config),
        iteration=args.iteration,
        champion_checkpoint=args.champion,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
