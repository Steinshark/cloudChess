from __future__ import annotations

import asyncio
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import bulletchess
import chess
import numpy as np

from chess_selflearn.encoding import initial_repetition_counts
from chess_selflearn.mcts import SearchConfig, SearchTree, run_batched_search

from .config import GameConfig
from .models import ModelCache, ModelDescriptor, ModelRegistry


class GameError(RuntimeError):
    pass


@dataclass(slots=True)
class MoveRecord:
    uci: str
    san: str
    color: str


@dataclass(slots=True)
class GameSession:
    id: str
    model_id: str
    human_color: chess.Color
    simulations: int
    board: chess.Board = field(default_factory=chess.Board)
    moves: list[MoveRecord] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    last_ai: dict[str, Any] | None = None
    lock: threading.RLock = field(default_factory=threading.RLock)


class GameManager:
    def __init__(
        self,
        config: GameConfig,
        registry: ModelRegistry,
        cache: ModelCache,
    ) -> None:
        self.config = config
        self.registry = registry
        self.cache = cache
        self._games: dict[str, GameSession] = {}
        self._games_lock = threading.RLock()
        self._ai_semaphore = asyncio.Semaphore(config.max_concurrent_ai)

    def _cleanup(self) -> None:
        cutoff = time.time() - self.config.game_ttl_minutes * 60
        with self._games_lock:
            expired = [
                game_id
                for game_id, game in self._games.items()
                if game.updated_at < cutoff
            ]
            for game_id in expired:
                self._games.pop(game_id, None)
            if len(self._games) > self.config.max_games:
                ordered = sorted(self._games.values(), key=lambda item: item.updated_at)
                for game in ordered[: len(self._games) - self.config.max_games]:
                    self._games.pop(game.id, None)

    def get(self, game_id: str) -> GameSession:
        self._cleanup()
        with self._games_lock:
            game = self._games.get(game_id)
        if game is None:
            raise KeyError(f"Unknown or expired game: {game_id}")
        return game

    async def create(
        self,
        model_id: str,
        human_color: str,
        simulations: int,
    ) -> dict[str, Any]:
        descriptor = self.registry.get(model_id)
        color = chess.WHITE if human_color.lower() == "white" else chess.BLACK
        simulations = max(
            self.config.min_simulations,
            min(self.config.max_simulations, int(simulations)),
        )
        game = GameSession(
            id=uuid.uuid4().hex,
            model_id=descriptor.id,
            human_color=color,
            simulations=simulations,
        )
        with self._games_lock:
            self._games[game.id] = game
        if game.board.turn != game.human_color:
            await self._make_ai_move(game)
        return self.public_state(game)

    async def play_human_move(self, game_id: str, uci: str) -> dict[str, Any]:
        game = self.get(game_id)
        with game.lock:
            if game.board.is_game_over(claim_draw=True):
                raise GameError("This game is already over")
            if game.board.turn != game.human_color:
                raise GameError("It is not the human player's turn")
            try:
                move = chess.Move.from_uci(uci.lower())
            except ValueError as exc:
                raise GameError(f"Invalid UCI move: {uci}") from exc
            if move not in game.board.legal_moves:
                raise GameError(f"Illegal move: {uci}")
            self._push(game, move)

        if not game.board.is_game_over(claim_draw=True):
            await self._make_ai_move(game)
        return self.public_state(game)

    def delete(self, game_id: str) -> None:
        with self._games_lock:
            self._games.pop(game_id, None)

    def _push(self, game: GameSession, move: chess.Move) -> None:
        san = game.board.san(move)
        color = "white" if game.board.turn == chess.WHITE else "black"
        game.board.push(move)
        game.moves.append(MoveRecord(uci=move.uci(), san=san, color=color))
        game.updated_at = time.time()

    async def _make_ai_move(self, game: GameSession) -> None:
        async with self._ai_semaphore:
            await asyncio.to_thread(self._make_ai_move_sync, game)

    def _make_ai_move_sync(self, game: GameSession) -> None:
        with game.lock:
            if game.board.is_game_over(claim_draw=True):
                return
            if game.board.turn == game.human_color:
                return
            started = time.perf_counter()
            loaded = self.cache.get(game.model_id)
            bullet_board = bulletchess.Board()
            for record in game.moves:
                bullet_move = bulletchess.Move.from_uci(record.uci)
                if bullet_move is None or bullet_move not in bullet_board.legal_moves():
                    raise GameError(
                        f"Could not rebuild game for MCTS at move {record.uci}"
                    )
                bullet_board.apply(bullet_move)

            tree = SearchTree(
                bullet_board,
                initial_repetition_counts(bullet_board),
                SearchConfig(
                    simulations=game.simulations,
                    c_puct_init=self.config.c_puct_init,
                    c_puct_base=self.config.c_puct_base,
                    dirichlet_alpha=0.30,
                    dirichlet_epsilon=0.0,
                ),
                np.random.default_rng(),
            )
            run_batched_search([tree], loaded.evaluator, simulations=game.simulations)
            _, bullet_move = tree.select_move(
                ply=len(game.moves),
                temperature_moves=0,
                temperature=0.0,
            )
            uci = bullet_move.uci()
            move = chess.Move.from_uci(uci)
            if move not in game.board.legal_moves:
                raise GameError(
                    f"Model search returned illegal move {uci} for {game.board.fen()}"
                )
            root_value = float(tree.root.mean_value)
            self._push(game, move)
            game.last_ai = {
                "move": uci,
                "san": game.moves[-1].san,
                "simulations": game.simulations,
                "root_value": root_value,
                "elapsed_seconds": time.perf_counter() - started,
                "model_id": game.model_id,
            }

    @staticmethod
    def _result(board: chess.Board) -> tuple[str | None, str | None]:
        if not board.is_game_over(claim_draw=True):
            return None, None
        result = board.result(claim_draw=True)
        outcome = board.outcome(claim_draw=True)
        termination = outcome.termination.name.lower().replace("_", " ") if outcome else "unknown"
        return result, termination

    def public_state(self, game: GameSession) -> dict[str, Any]:
        with game.lock:
            descriptor: ModelDescriptor = self.registry.get(game.model_id)
            result, termination = self._result(game.board)
            legal_moves = [move.uci() for move in game.board.legal_moves]
            return {
                "id": game.id,
                "model": descriptor.public(),
                "human_color": "white" if game.human_color else "black",
                "simulations": game.simulations,
                "fen": game.board.fen(),
                "turn": "white" if game.board.turn else "black",
                "human_to_move": (
                    game.board.turn == game.human_color
                    and not game.board.is_game_over(claim_draw=True)
                ),
                "game_over": game.board.is_game_over(claim_draw=True),
                "in_check": game.board.is_check(),
                "result": result,
                "termination": termination,
                "legal_moves": legal_moves,
                "moves": [
                    {"ply": index + 1, "uci": move.uci, "san": move.san, "color": move.color}
                    for index, move in enumerate(game.moves)
                ],
                "last_move": game.moves[-1].uci if game.moves else None,
                "last_ai": game.last_ai,
                "created_at": datetime.fromtimestamp(
                    game.created_at, tz=timezone.utc
                ).isoformat(),
                "updated_at": datetime.fromtimestamp(
                    game.updated_at, tz=timezone.utc
                ).isoformat(),
            }
