from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import bulletchess
import numpy as np
import torch
from tqdm import tqdm

from .checkpoint import load_model
from .config import AppConfig, load_config
from .encoding import initial_repetition_counts
from .evaluator import EvaluatorConfig, NeuralEvaluator
from .mcts import SearchConfig, SearchTree, run_batched_search
from .openings import DEFAULT_OPENINGS


@dataclass(slots=True)
class ArenaGame:
    game_id: int
    candidate_color: object
    board: bulletchess.Board
    repetitions: Counter[str]
    moves_uci: list[str]
    finished: bool = False
    result_white: int = 0
    termination: str = ""
    tree: SearchTree | None = None

    @property
    def candidate_to_move(self) -> bool:
        return self.board.turn == self.candidate_color

    def apply_tree_move(self) -> None:
        if self.tree is None:
            raise RuntimeError("Arena game has no searched tree")
        action, move = self.tree.select_move(
            ply=len(self.moves_uci),
            temperature_moves=0,
            temperature=0.0,
        )
        self.moves_uci.append(move.uci())
        self.tree.advance(action, move)
        self.board = self.tree.board
        self.repetitions = self.tree.repetitions
        self.tree = None

    def check_finished(self, max_game_plies: int) -> None:
        if self.board in bulletchess.CHECKMATE:
            self.finished = True
            self.termination = "checkmate"
            self.result_white = -1 if self.board.turn == bulletchess.WHITE else 1
        elif self.board in bulletchess.DRAW:
            self.finished = True
            self.termination = "draw"
            self.result_white = 0
        elif len(self.moves_uci) >= max_game_plies:
            self.finished = True
            self.termination = "ply_limit"
            self.result_white = 0

    def candidate_result(self) -> int:
        return (
            self.result_white
            if self.candidate_color == bulletchess.WHITE
            else -self.result_white
        )


def board_from_opening(opening: tuple[str, ...]) -> bulletchess.Board:
    board = bulletchess.Board()
    for uci in opening:
        move = bulletchess.Move.from_uci(uci)
        if move is None or move not in board.legal_moves():
            raise ValueError(f"Illegal configured opening move {uci} in {board.fen()}")
        board.apply(move)
    return board


def make_arena_games(config: AppConfig) -> list[ArenaGame]:
    games: list[ArenaGame] = []
    pairs = config.arena.games // 2
    for pair in range(pairs):
        opening = DEFAULT_OPENINGS[pair % len(DEFAULT_OPENINGS)]
        for candidate_color in (bulletchess.WHITE, bulletchess.BLACK):
            board = board_from_opening(opening)
            games.append(
                ArenaGame(
                    game_id=len(games),
                    candidate_color=candidate_color,
                    board=board,
                    repetitions=initial_repetition_counts(board),
                    moves_uci=list(opening),
                )
            )
    return games


def evaluate_candidate(
    config: AppConfig,
    *,
    iteration: int,
    champion_checkpoint: str | Path,
    candidate_checkpoint: str | Path,
) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA-enabled PyTorch is required")

    device = torch.device("cuda")
    champion_model, champion_payload = load_model(
        champion_checkpoint,
        device,
        channels_last=config.arena.channels_last,
    )
    candidate_model, candidate_payload = load_model(
        candidate_checkpoint,
        device,
        channels_last=config.arena.channels_last,
    )
    evaluator_config = EvaluatorConfig(
        precision=config.arena.precision,
        channels_last=config.arena.channels_last,
        max_batch_size=config.arena.concurrent_games,
    )
    champion_evaluator = NeuralEvaluator(champion_model, device, evaluator_config)
    candidate_evaluator = NeuralEvaluator(candidate_model, device, evaluator_config)

    search_config = SearchConfig(
        simulations=config.arena.simulations,
        c_puct_init=config.self_play.c_puct_init,
        c_puct_base=config.self_play.c_puct_base,
        dirichlet_alpha=config.self_play.dirichlet_alpha,
        dirichlet_epsilon=0.0,
    )
    rng = np.random.default_rng(config.seed + iteration * 200_003)
    all_games = make_arena_games(config)
    finished_games: list[ArenaGame] = []
    progress = tqdm(total=len(all_games), desc="Arena games", dynamic_ncols=True)

    for batch_start in range(0, len(all_games), config.arena.concurrent_games):
        active = all_games[
            batch_start : batch_start + config.arena.concurrent_games
        ]

        while active:
            for game in active:
                game.tree = SearchTree(
                    game.board,
                    game.repetitions,
                    search_config,
                    np.random.default_rng(
                        int(rng.integers(0, np.iinfo(np.int64).max))
                    ),
                )

            candidate_games = [game for game in active if game.candidate_to_move]
            champion_games = [game for game in active if not game.candidate_to_move]

            run_batched_search(
                [game.tree for game in candidate_games if game.tree is not None],
                candidate_evaluator,
                simulations=config.arena.simulations,
            )
            run_batched_search(
                [game.tree for game in champion_games if game.tree is not None],
                champion_evaluator,
                simulations=config.arena.simulations,
            )

            for game in active:
                game.apply_tree_move()
                game.check_finished(config.arena.max_game_plies)

            newly_finished = [game for game in active if game.finished]
            finished_games.extend(newly_finished)
            progress.update(len(newly_finished))
            active = [game for game in active if not game.finished]

    progress.close()

    wins = sum(game.candidate_result() > 0 for game in finished_games)
    losses = sum(game.candidate_result() < 0 for game in finished_games)
    draws = len(finished_games) - wins - losses
    score = (wins + 0.5 * draws) / max(1, len(finished_games))
    promoted = score >= config.arena.promotion_score

    iteration_root = (
        Path(config.run.root)
        / "iterations"
        / f"iteration_{iteration:06d}"
    )
    details_path = iteration_root / "arena_games.jsonl"
    with details_path.open("w", encoding="utf-8") as handle:
        for game in finished_games:
            handle.write(
                json.dumps(
                    {
                        "game_id": game.game_id,
                        "candidate_color": (
                            "white"
                            if game.candidate_color == bulletchess.WHITE
                            else "black"
                        ),
                        "candidate_result": game.candidate_result(),
                        "result_white": game.result_white,
                        "termination": game.termination,
                        "plies": len(game.moves_uci),
                        "moves_uci": game.moves_uci,
                    }
                )
                + "\n"
            )

    summary = {
        "iteration": iteration,
        "champion_generation": int(champion_payload.get("generation", 0)),
        "candidate_generation": int(candidate_payload.get("generation", 0)),
        "games": len(finished_games),
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score": score,
        "promotion_threshold": config.arena.promotion_score,
        "promoted": promoted,
    }
    (iteration_root / "arena_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("selflearn_config.yaml"))
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--champion", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()

    summary = evaluate_candidate(
        load_config(args.config),
        iteration=args.iteration,
        champion_checkpoint=args.champion,
        candidate_checkpoint=args.candidate,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
