from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import bulletchess
import numpy as np
import torch
from tqdm import tqdm

from .checkpoint import load_model
from .config import AppConfig, load_config
from .encoding import encode_board, initial_repetition_counts, previous_board, repetition_key
from .evaluator import EvaluatorConfig, NeuralEvaluator
from .mcts import SearchConfig, SearchTree, run_batched_search
from .replay import (
    FinishedGame,
    ReplayWriter,
    TrainingSample,
    update_replay_manifest,
)


@dataclass(slots=True)
class PendingSample:
    state: np.ndarray
    policy_indices: np.ndarray
    policy_probabilities: np.ndarray
    to_play: object
    ply: int


class SelfPlayGame:
    def __init__(
        self,
        game_id: int,
        config: AppConfig,
        rng: np.random.Generator,
    ) -> None:
        self.game_id = game_id
        self.config = config
        self.rng = rng
        self.board = bulletchess.Board()
        self.repetitions = initial_repetition_counts(self.board)
        self.tree = SearchTree(
            self.board,
            self.repetitions,
            SearchConfig(
                simulations=config.self_play.simulations,
                c_puct_init=config.self_play.c_puct_init,
                c_puct_base=config.self_play.c_puct_base,
                dirichlet_alpha=config.self_play.dirichlet_alpha,
                dirichlet_epsilon=config.self_play.dirichlet_epsilon,
            ),
            rng,
        )
        self.pending: list[PendingSample] = []
        self.moves_uci: list[str] = []
        self.finished = False
        self.result_white = 0
        self.termination = ""

    @property
    def ply(self) -> int:
        return len(self.moves_uci)

    def record_and_play(self) -> None:
        root_key = repetition_key(self.tree.board)
        state = encode_board(
            self.tree.board,
            prior_board=previous_board(self.tree.board),
            repetition_count=self.tree.repetitions[root_key],
        )
        policy_indices, policy_probabilities = self.tree.policy_target()
        action, move = self.tree.select_move(
            ply=self.ply,
            temperature_moves=self.config.self_play.temperature_moves,
            temperature=self.config.self_play.temperature,
        )

        self.pending.append(
            PendingSample(
                state=state,
                policy_indices=policy_indices,
                policy_probabilities=policy_probabilities,
                to_play=self.tree.board.turn,
                ply=self.ply,
            )
        )
        self.moves_uci.append(move.uci())
        self.tree.advance(action, move)
        self.board = self.tree.board
        self.repetitions = self.tree.repetitions
        self._check_finished()

    def _check_finished(self) -> None:
        if self.board in bulletchess.CHECKMATE:
            self.finished = True
            self.termination = "checkmate"
            self.result_white = (
                -1 if self.board.turn == bulletchess.WHITE else 1
            )
        elif self.board in bulletchess.DRAW:
            self.finished = True
            self.termination = "draw"
            self.result_white = 0
        elif self.ply >= self.config.self_play.max_game_plies:
            self.finished = True
            self.termination = "ply_limit"
            self.result_white = 0

    def finish(self) -> FinishedGame:
        if not self.finished:
            raise RuntimeError("Cannot finalize an unfinished game")

        samples = [
            TrainingSample(
                state=pending.state,
                policy_indices=pending.policy_indices,
                policy_probabilities=pending.policy_probabilities,
                value=(
                    self.result_white
                    if pending.to_play == bulletchess.WHITE
                    else -self.result_white
                ),
                ply=pending.ply,
                game_id=self.game_id,
            )
            for pending in self.pending
        ]
        return FinishedGame(
            samples=samples,
            result_white=self.result_white,
            moves_uci=list(self.moves_uci),
            termination=self.termination,
        )


def generate_self_play(
    config: AppConfig,
    *,
    iteration: int,
    checkpoint_path: str | Path,
) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA-enabled PyTorch is required for self-play")

    run_root = Path(config.run.root)
    iteration_root = run_root / "iterations" / f"iteration_{iteration:06d}"
    iteration_root.mkdir(parents=True, exist_ok=True)
    replay_path = iteration_root / "selfplay.h5"
    games_path = iteration_root / "games.jsonl"

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    model, checkpoint = load_model(
        checkpoint_path,
        device,
        channels_last=config.self_play.channels_last,
    )
    generation = int(checkpoint.get("generation", 0))
    evaluator = NeuralEvaluator(
        model,
        device,
        EvaluatorConfig(
            precision=config.self_play.precision,
            channels_last=config.self_play.channels_last,
            max_batch_size=config.self_play.inference_batch_size,
        ),
    )

    master_rng = np.random.default_rng(config.seed + iteration * 100_003)
    total_games = config.self_play.games_per_iteration
    completed_games = 0
    total_positions = 0
    result_counts = {"white": 0, "black": 0, "draw": 0}
    termination_counts: Counter[str] = Counter()
    started = time.perf_counter()

    with ReplayWriter(
        replay_path,
        max_policy_moves=config.self_play.max_policy_moves,
        generation=generation,
    ) as writer, games_path.open("w", encoding="utf-8") as game_log:
        progress = tqdm(total=total_games, desc="Self-play games", dynamic_ncols=True)

        while completed_games < total_games:
            batch_size = min(
                config.self_play.concurrent_games,
                total_games - completed_games,
            )
            games = [
                SelfPlayGame(
                    game_id=(iteration << 32) | (completed_games + index),
                    config=config,
                    rng=np.random.default_rng(
                        int(master_rng.integers(0, np.iinfo(np.int64).max))
                    ),
                )
                for index in range(batch_size)
            ]

            active = games
            while active:
                run_batched_search(
                    [game.tree for game in active],
                    evaluator,
                    simulations=config.self_play.simulations,
                )
                for game in active:
                    game.record_and_play()
                active = [game for game in active if not game.finished]

            finished = [game.finish() for game in games]
            writer.append_games(finished)

            for game in finished:
                total_positions += len(game.samples)
                termination_counts[game.termination] += 1
                if game.result_white > 0:
                    result_counts["white"] += 1
                elif game.result_white < 0:
                    result_counts["black"] += 1
                else:
                    result_counts["draw"] += 1
                game_log.write(
                    json.dumps(
                        {
                            "game_id": game.samples[0].game_id if game.samples else -1,
                            "result_white": game.result_white,
                            "termination": game.termination,
                            "plies": len(game.moves_uci),
                            "moves_uci": game.moves_uci,
                        }
                    )
                    + "\n"
                )

            completed_games += len(finished)
            progress.update(len(finished))
            elapsed = max(time.perf_counter() - started, 1e-9)
            progress.set_postfix(
                positions=f"{total_positions:,}",
                pos_s=f"{total_positions / elapsed:,.1f}",
            )

        progress.close()

    update_replay_manifest(
        run_root,
        iteration=iteration,
        replay_path=replay_path,
        games=completed_games,
        positions=total_positions,
        generation=generation,
    )

    elapsed = time.perf_counter() - started
    summary = {
        "iteration": iteration,
        "generation": generation,
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "games": completed_games,
        "positions": total_positions,
        "results": result_counts,
        "terminations": dict(termination_counts),
        "elapsed_seconds": elapsed,
        "positions_per_second": total_positions / max(elapsed, 1e-9),
        "replay_path": str(replay_path.resolve()),
    }
    (iteration_root / "selfplay_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("selflearn_config.yaml"))
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    summary = generate_self_play(
        config,
        iteration=args.iteration,
        checkpoint_path=args.checkpoint,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
