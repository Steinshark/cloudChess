from __future__ import annotations

from collections import Counter

import bulletchess
import numpy as np

from .encoding import encode_board_with_history, initial_repetition_counts
from .mcts import SearchConfig, SearchTree, run_batched_search


class UniformEvaluator:
    """CPU-only structural evaluator used to test MCTS scheduling."""

    def __init__(self, max_batch_size: int = 32) -> None:
        self.max_batch_size = max_batch_size
        self.batch_sizes: list[int] = []

    def evaluate(
        self,
        states: list[np.ndarray],
        legal_actions: list[np.ndarray],
    ) -> tuple[list[np.ndarray], np.ndarray]:
        self.batch_sizes.append(len(states))
        policies = [
            np.zeros(actions.size, dtype=np.float32)
            for actions in legal_actions
        ]
        values = np.zeros(len(states), dtype=np.float32)
        return policies, values


def assert_no_reservations(node: object) -> None:
    assert getattr(node, "virtual_visits") == 0
    assert not getattr(node, "in_flight")
    for child in getattr(node, "children").values():
        assert_no_reservations(child)


def main() -> None:
    board = bulletchess.Board()
    move = bulletchess.Move.from_uci("e2e4")
    if move is None:
        raise RuntimeError("bulletchess rejected e2e4")
    board.apply(move)
    before_fen = board.fen()
    before_history = list(board.history)
    encoded = encode_board_with_history(board, repetition_count=1)
    assert board.fen() == before_fen
    assert board.history == before_history
    assert encoded.shape == (34, 8, 8)
    assert np.any(encoded[12:24])

    search_config = SearchConfig(
        simulations=32,
        leaves_per_tree=4,
        virtual_loss=1.0,
    )
    trees: list[SearchTree] = []
    for index in range(8):
        root_board = bulletchess.Board()
        trees.append(
            SearchTree(
                root_board,
                initial_repetition_counts(root_board),
                search_config,
                np.random.default_rng(index),
            )
        )

    evaluator = UniformEvaluator(max_batch_size=32)
    metrics = run_batched_search(trees, evaluator, simulations=32)

    for tree in trees:
        assert tree.root.visit_count == 32
        assert tree.board.fen() == bulletchess.Board().fen()
        assert len(tree.board.history) == 0
        assert_no_reservations(tree.root)

    assert metrics.simulations_completed == 8 * 32
    assert max(evaluator.batch_sizes) > len(trees), evaluator.batch_sizes

    print("Optimized MCTS structural smoke test passed")
    print(f"Inference batches: {evaluator.batch_sizes}")
    print(f"Search metrics: {metrics.as_dict()}")


if __name__ == "__main__":
    main()
