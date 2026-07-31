from __future__ import annotations

import argparse
import socket
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import WebConfig, load_web_config
from .games import GameError, GameManager
from .models import ModelCache, ModelRegistry
from .stats import StatsService


class NewGameRequest(BaseModel):
    model_id: str
    human_color: str = Field(pattern="^(white|black)$")
    simulations: int


class MoveRequest(BaseModel):
    uci: str = Field(min_length=4, max_length=5)


def create_app(config: WebConfig) -> FastAPI:
    registry = ModelRegistry(config.models)
    cache = ModelCache(registry, config.models)
    games = GameManager(config.games, registry, cache)
    stats = StatsService(config.stats)
    static_root = Path(__file__).with_name("static")

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        registry.refresh(force=True)
        yield

    app = FastAPI(
        title="Neural Chess LAN",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.registry = registry
    app.state.cache = cache
    app.state.games = games
    app.state.stats = stats

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {
            "ok": True,
            "models": len(registry.refresh()),
            "cache": cache.status(),
        }

    @app.get("/api/models")
    def list_models(refresh: bool = False) -> dict[str, Any]:
        models = registry.refresh(force=refresh)
        return {
            "models": [item.public() for item in models],
            "cache": cache.status(),
            "simulation_limits": {
                "minimum": config.games.min_simulations,
                "maximum": config.games.max_simulations,
                "default": config.games.default_simulations,
            },
        }

    @app.post("/api/games")
    async def create_game(request: NewGameRequest) -> dict[str, Any]:
        try:
            return await games.create(
                request.model_id,
                request.human_color,
                request.simulations,
            )
        except (KeyError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/games/{game_id}")
    def get_game(game_id: str) -> dict[str, Any]:
        try:
            return games.public_state(games.get(game_id))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/games/{game_id}/moves")
    async def play_move(game_id: str, request: MoveRequest) -> dict[str, Any]:
        try:
            return await games.play_human_move(game_id, request.uci)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except GameError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"AI move failed: {type(exc).__name__}: {exc}",
            ) from exc

    @app.delete("/api/games/{game_id}", status_code=204)
    def delete_game(game_id: str) -> None:
        games.delete(game_id)

    @app.get("/api/stats/overview")
    def stats_overview() -> dict[str, Any]:
        return stats.overview()

    @app.get("/api/stats/bootstrap/{run_id}")
    def bootstrap_stats(run_id: str) -> dict[str, Any]:
        try:
            return stats.bootstrap_detail(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Could not read TensorBoard logs: {type(exc).__name__}: {exc}",
            ) from exc

    @app.get("/api/stats/self-learning/{run_id}")
    def self_learning_stats(run_id: str) -> dict[str, Any]:
        try:
            return stats.self_learning_detail(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/stats/self-learning/{run_id}/iterations/{iteration}")
    def iteration_stats(run_id: str, iteration: int) -> dict[str, Any]:
        try:
            return stats.iteration_detail(run_id, iteration)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/")
    def play_page() -> FileResponse:
        return FileResponse(static_root / "play.html")

    @app.get("/stats")
    def stats_page() -> FileResponse:
        return FileResponse(static_root / "stats.html")

    app.mount("/static", StaticFiles(directory=static_root), name="static")
    return app


def _local_ip() -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except OSError:
        return "<this-computer-ip>"
    finally:
        sock.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve checkpointed chess models over your LAN")
    parser.add_argument("--config", type=Path, default=Path("web_config.yaml"))
    args = parser.parse_args()
    config = load_web_config(args.config)
    print(f"Local: http://127.0.0.1:{config.server.port}")
    print(f"LAN:   http://{_local_ip()}:{config.server.port}")
    uvicorn.run(
        create_app(config),
        host=config.server.host,
        port=config.server.port,
        reload=config.server.reload,
    )


if __name__ == "__main__":
    main()
