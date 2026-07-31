from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class RunConfig:
    root: str
    initial_checkpoint: str


@dataclass(slots=True)
class SelfPlayConfig:
    games_per_iteration: int = 256
    concurrent_games: int = 96
    simulations: int = 256
    max_game_plies: int = 300
    temperature_moves: int = 20
    temperature: float = 1.0
    c_puct_init: float = 1.25
    c_puct_base: float = 19652.0
    dirichlet_alpha: float = 0.30
    dirichlet_epsilon: float = 0.25
    inference_batch_size: int = 128
    leaves_per_tree: int = 4
    virtual_loss: float = 1.0
    precision: str = "fp16"
    channels_last: bool = True
    max_policy_moves: int = 256


@dataclass(slots=True)
class TrainingConfig:
    steps_per_iteration: int = 5000
    batch_size: int = 512
    num_workers: int = 2
    read_chunk_size: int = 4096
    replay_keep_iterations: int = 8
    bootstrap_dataset_root: str | None = None
    bootstrap_mix_ratio: float = 0.25
    learning_rate: float = 2e-4
    minimum_learning_rate: float = 1e-5
    warmup_steps: int = 250
    weight_decay: float = 1e-4
    value_loss_weight: float = 1.0
    gradient_clip_norm: float = 1.0
    precision: str = "fp16"
    channels_last: bool = True
    checkpoint_every_steps: int = 1000


@dataclass(slots=True)
class ArenaConfig:
    games: int = 40
    concurrent_games: int = 20
    simulations: int = 400
    inference_batch_size: int = 64
    leaves_per_tree: int = 4
    virtual_loss: float = 1.0
    max_game_plies: int = 300
    promotion_score: float = 0.55
    precision: str = "fp16"
    channels_last: bool = True


@dataclass(slots=True)
class AppConfig:
    seed: int
    run: RunConfig
    self_play: SelfPlayConfig
    training: TrainingConfig
    arena: ArenaConfig


def load_config(path: str | Path) -> AppConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)

    config = AppConfig(
        seed=int(raw.get("seed", 1337)),
        run=RunConfig(**raw["run"]),
        self_play=SelfPlayConfig(**raw["self_play"]),
        training=TrainingConfig(**raw["training"]),
        arena=ArenaConfig(**raw["arena"]),
    )
    validate_config(config)
    return config


def validate_config(config: AppConfig) -> None:
    if config.self_play.games_per_iteration <= 0:
        raise ValueError("games_per_iteration must be positive")
    if config.self_play.concurrent_games <= 0:
        raise ValueError("concurrent_games must be positive")
    if config.self_play.simulations <= 0:
        raise ValueError("self-play simulations must be positive")
    if config.self_play.inference_batch_size <= 0:
        raise ValueError("inference_batch_size must be positive")
    if config.self_play.leaves_per_tree <= 0:
        raise ValueError("leaves_per_tree must be positive")
    if config.self_play.virtual_loss < 0.0:
        raise ValueError("virtual_loss cannot be negative")
    if config.self_play.max_policy_moves < 218:
        raise ValueError("max_policy_moves must be at least 218")
    if not 0.0 <= config.training.bootstrap_mix_ratio <= 1.0:
        raise ValueError("bootstrap_mix_ratio must be in [0, 1]")
    if config.arena.inference_batch_size <= 0:
        raise ValueError("arena inference_batch_size must be positive")
    if config.arena.leaves_per_tree <= 0:
        raise ValueError("arena leaves_per_tree must be positive")
    if config.arena.virtual_loss < 0.0:
        raise ValueError("arena virtual_loss cannot be negative")
    if config.arena.games <= 0 or config.arena.games % 2:
        raise ValueError("arena games must be a positive even number")
    if not 0.5 <= config.arena.promotion_score <= 1.0:
        raise ValueError("promotion_score must be in [0.5, 1.0]")
    for precision in (
        config.self_play.precision,
        config.training.precision,
        config.arena.precision,
    ):
        if precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError(f"Unsupported precision: {precision}")
