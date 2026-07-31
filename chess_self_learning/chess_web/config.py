from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False


@dataclass(slots=True)
class ModelScanConfig:
    roots: list[str] = field(default_factory=list)
    patterns: list[str] = field(default_factory=lambda: ["**/*.pt"])
    exclude_name_contains: list[str] = field(
        default_factory=lambda: [".tmp", "optimizer_only"]
    )
    device: str = "auto"
    precision: str = "fp16"
    channels_last: bool = True
    max_loaded_models: int = 2
    refresh_seconds: int = 30


@dataclass(slots=True)
class GameConfig:
    default_simulations: int = 128
    min_simulations: int = 1
    max_simulations: int = 800
    max_concurrent_ai: int = 1
    max_games: int = 128
    game_ttl_minutes: int = 360
    c_puct_init: float = 1.25
    c_puct_base: float = 19652.0


@dataclass(slots=True)
class StatsConfig:
    bootstrap_roots: list[str] = field(default_factory=list)
    self_learning_roots: list[str] = field(default_factory=list)
    max_points_per_series: int = 1200


@dataclass(slots=True)
class WebConfig:
    server: ServerConfig
    models: ModelScanConfig
    games: GameConfig
    stats: StatsConfig


def _section(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Configuration section {key!r} must be a mapping")
    return value


def load_web_config(path: str | Path) -> WebConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Web config does not exist: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("Top-level web configuration must be a mapping")

    config = WebConfig(
        server=ServerConfig(**_section(raw, "server")),
        models=ModelScanConfig(**_section(raw, "models")),
        games=GameConfig(**_section(raw, "games")),
        stats=StatsConfig(**_section(raw, "stats")),
    )
    validate_web_config(config)
    return config


def validate_web_config(config: WebConfig) -> None:
    if not 1 <= config.server.port <= 65535:
        raise ValueError("server.port must be in [1, 65535]")
    if config.models.precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError("models.precision must be fp32, fp16, or bf16")
    if config.models.device not in {"auto", "cuda", "cpu"}:
        raise ValueError("models.device must be auto, cuda, or cpu")
    if config.models.max_loaded_models <= 0:
        raise ValueError("models.max_loaded_models must be positive")
    if config.games.min_simulations <= 0:
        raise ValueError("games.min_simulations must be positive")
    if config.games.max_simulations < config.games.min_simulations:
        raise ValueError("games.max_simulations must be >= games.min_simulations")
    if not (
        config.games.min_simulations
        <= config.games.default_simulations
        <= config.games.max_simulations
    ):
        raise ValueError("games.default_simulations is outside the allowed range")
    if config.games.max_concurrent_ai <= 0:
        raise ValueError("games.max_concurrent_ai must be positive")
    if config.stats.max_points_per_series < 50:
        raise ValueError("stats.max_points_per_series must be at least 50")
