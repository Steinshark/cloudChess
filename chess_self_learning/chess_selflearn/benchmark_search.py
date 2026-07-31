from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import bulletchess
import numpy as np
import torch

from .checkpoint import load_model
from .config import load_config
from .encoding import initial_repetition_counts
from .evaluator import EvaluatorConfig, NeuralEvaluator
from .mcts import SearchConfig, SearchMetrics, SearchTree, run_batched_search


def make_tree(search_config: SearchConfig, seed: int) -> SearchTree:
    board = bulletchess.Board()
    return SearchTree(
        board,
        initial_repetition_counts(board),
        search_config,
        np.random.default_rng(seed),
    )


def benchmark_setting(
    *,
    model: torch.nn.Module,
    device: torch.device,
    precision: str,
    channels_last: bool,
    concurrency: int,
    inference_batch_size: int,
    simulations: int,
    plies: int,
    search_config: SearchConfig,
    seed: int,
) -> dict[str, object]:
    evaluator = NeuralEvaluator(
        model,
        device,
        EvaluatorConfig(
            precision=precision,
            channels_last=channels_last,
            max_batch_size=inference_batch_size,
        ),
    )
    trees = [make_tree(search_config, seed + index * 997) for index in range(concurrency)]
    metrics = SearchMetrics()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()

    for ply in range(plies):
        metrics.merge(
            run_batched_search(
                trees,
                evaluator,
                simulations=simulations,
            )
        )
        next_trees: list[SearchTree] = []
        for index, tree in enumerate(trees):
            action, move = tree.select_move(
                ply=ply,
                temperature_moves=0,
                temperature=0.0,
            )
            tree.advance(action, move)
            if SearchTree.terminal_value(tree.board) is not None:
                tree = make_tree(
                    search_config,
                    seed + (ply + 1) * 1_000_003 + index,
                )
            next_trees.append(tree)
        trees = next_trees

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started

    return {
        "concurrent_games": concurrency,
        "inference_batch_size": inference_batch_size,
        "leaves_per_tree": search_config.leaves_per_tree,
        "simulations_per_move": simulations,
        "measured_plies": plies,
        "elapsed_seconds": elapsed,
        "search": metrics.as_dict(elapsed),
        "evaluator": evaluator.metrics.as_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare MCTS throughput at 64, 96, and 128 concurrent games."
    )
    parser.add_argument("--config", type=Path, default=Path("selflearn_config.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--concurrency",
        type=int,
        nargs="+",
        default=[64, 96, 128],
    )
    parser.add_argument("--simulations", type=int, default=64)
    parser.add_argument("--plies", type=int, default=6)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA-enabled PyTorch is required for this benchmark")

    config = load_config(args.config)
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    model, _ = load_model(
        args.checkpoint,
        device,
        channels_last=config.self_play.channels_last,
    )
    search_config = SearchConfig(
        simulations=args.simulations,
        c_puct_init=config.self_play.c_puct_init,
        c_puct_base=config.self_play.c_puct_base,
        dirichlet_alpha=config.self_play.dirichlet_alpha,
        dirichlet_epsilon=config.self_play.dirichlet_epsilon,
        leaves_per_tree=config.self_play.leaves_per_tree,
        virtual_loss=config.self_play.virtual_loss,
    )

    # Warm cuDNN/autocast kernels so the first measured setting is comparable.
    warmup_evaluator = NeuralEvaluator(
        model,
        device,
        EvaluatorConfig(
            precision=config.self_play.precision,
            channels_last=config.self_play.channels_last,
            max_batch_size=min(16, config.self_play.inference_batch_size),
        ),
    )
    warmup_trees = [make_tree(search_config, config.seed + i) for i in range(4)]
    run_batched_search(warmup_trees, warmup_evaluator, simulations=8)
    torch.cuda.synchronize(device)

    results = [
        benchmark_setting(
            model=model,
            device=device,
            precision=config.self_play.precision,
            channels_last=config.self_play.channels_last,
            concurrency=concurrency,
            inference_batch_size=config.self_play.inference_batch_size,
            simulations=args.simulations,
            plies=args.plies,
            search_config=search_config,
            seed=config.seed + concurrency * 10_007,
        )
        for concurrency in args.concurrency
    ]
    payload = {
        "checkpoint": str(args.checkpoint.resolve()),
        "results": results,
        "recommended_concurrent_games": max(
            results,
            key=lambda row: float(
                row["search"].get("simulations_per_second", 0.0)  # type: ignore[union-attr]
            ),
        )["concurrent_games"],
    }
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
