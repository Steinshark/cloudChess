from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import yaml

@dataclass(slots=True)
class DataConfig:
    dataset_root: str
    train_split_percent: int = 98
    validation_split_percent: int = 1
    split_seed: int = 7349
    read_chunk_size: int = 8192
    shuffle_buffer_chunks: int = 4
    num_workers: int = 2
    prefetch_factor: int = 2
    pin_memory: bool = True

@dataclass(slots=True)
class ModelConfig:
    input_planes: int = 34
    channels: int = 128
    residual_blocks: int = 8
    policy_planes: int = 73
    value_hidden: int = 256
    normalization: str = "batchnorm"
    activation: str = "relu"

@dataclass(slots=True)
class TrainingConfig:
    output_dir: str
    batch_size: int = 512
    gradient_accumulation_steps: int = 1
    max_steps: int = 100000
    learning_rate: float = 1e-3
    minimum_learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.999
    warmup_steps: int = 2000
    gradient_clip_norm: float = 1.0
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 1.0
    label_smoothing: float = 0.02
    precision: str = "fp16"
    channels_last: bool = True
    compile_model: bool = False
    log_every_steps: int = 50
    validate_every_steps: int = 2000
    checkpoint_every_steps: int = 2000
    validation_batches: int = 200
    resume_from: str | None = None
    keep_last_checkpoints: int = 5

@dataclass(slots=True)
class AppConfig:
    seed: int
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig

def load_config(path: str | Path) -> AppConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)
    cfg = AppConfig(
        seed=int(raw.get("seed", 1337)),
        data=DataConfig(**raw["data"]),
        model=ModelConfig(**raw["model"]),
        training=TrainingConfig(**raw["training"]),
    )
    if cfg.data.train_split_percent + cfg.data.validation_split_percent >= 100:
        raise ValueError("Train + validation must be below 100")
    if cfg.model.policy_planes != 73:
        raise ValueError("This pipeline expects 73 policy planes")
    if cfg.training.precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError("precision must be fp32, fp16, or bf16")
    return cfg
