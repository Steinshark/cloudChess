from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import bulletchess
import numpy as np
import torch

from .checkpoint import load_model
from .encoding import encode_board, initial_repetition_counts, legal_action_map
from .evaluator import EvaluatorConfig, NeuralEvaluator
from .mcts import SearchConfig, SearchTree, run_batched_search


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()

    board = bulletchess.Board()
    repetitions = initial_repetition_counts(board)
    state = encode_board(board, repetition_count=1)
    actions, moves = legal_action_map(board)
    assert state.shape == (34, 8, 8)
    assert len(actions) == 20
    assert len(set(int(action) for action in actions)) == 20

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the full smoke test")
    device = torch.device("cuda")
    model, _ = load_model(args.checkpoint, device)
    evaluator = NeuralEvaluator(
        model,
        device,
        EvaluatorConfig(max_batch_size=8),
    )
    tree = SearchTree(
        board,
        repetitions,
        SearchConfig(simulations=8, leaves_per_tree=4, virtual_loss=1.0),
        np.random.default_rng(1),
    )
    metrics = run_batched_search([tree], evaluator, simulations=8)
    assert tree.root.virtual_visits == 0
    _, move = tree.select_move(0, 0, 0.0)
    assert move in board.legal_moves()
    print(f"State shape: {state.shape}")
    print(f"Legal actions: {len(actions)}")
    print(f"MCTS root visits: {tree.root.visit_count}")
    print(f"Selected legal move: {move.uci()}")
    print(f"Search metrics: {metrics.as_dict()}")
    print(f"Evaluator metrics: {evaluator.metrics.as_dict()}")


if __name__ == "__main__":
    main()
