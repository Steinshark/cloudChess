from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

import bulletchess
import numpy as np

from .encoding import (
    encode_board,
    legal_action_map,
    opposite,
    previous_board,
    repetition_key,
)
from .evaluator import NeuralEvaluator


@dataclass(slots=True)
class SearchConfig:
    simulations: int
    c_puct_init: float = 1.25
    c_puct_base: float = 19652.0
    dirichlet_alpha: float = 0.30
    dirichlet_epsilon: float = 0.25


@dataclass(slots=True)
class Node:
    to_play: object
    prior: float = 0.0
    move: bulletchess.Move | None = None
    visit_count: int = 0
    value_sum: float = 0.0
    expanded: bool = False
    children: dict[int, "Node"] = field(default_factory=dict)

    @property
    def mean_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


@dataclass(slots=True)
class Leaf:
    tree: "SearchTree"
    board: bulletchess.Board
    path: list[Node]
    repetition_count: int
    terminal_value: float | None


class SearchTree:
    def __init__(
        self,
        board: bulletchess.Board,
        repetitions: Counter[str],
        config: SearchConfig,
        rng: np.random.Generator,
    ) -> None:
        self.board = board.copy()
        self.repetitions = Counter(repetitions)
        self.config = config
        self.rng = rng
        self.root = Node(to_play=self.board.turn, prior=1.0)
        self.root_noise_applied = False

    @staticmethod
    def terminal_value(board: bulletchess.Board) -> float | None:
        if board in bulletchess.CHECKMATE:
            return -1.0
        if board in bulletchess.DRAW:
            return 0.0
        return None

    def _puct_score(self, parent: Node, child: Node) -> float:
        q_value = -child.mean_value if child.visit_count else 0.0
        parent_visits = max(1, parent.visit_count)
        pb_c = math.log(
            (parent_visits + self.config.c_puct_base + 1.0)
            / self.config.c_puct_base
        ) + self.config.c_puct_init
        exploration = (
            pb_c
            * child.prior
            * math.sqrt(parent_visits)
            / (child.visit_count + 1)
        )
        return q_value + exploration

    def _select_child(self, node: Node) -> tuple[int, Node]:
        best_action = -1
        best_child: Node | None = None
        best_score = -float("inf")

        for action, child in node.children.items():
            score = self._puct_score(node, child)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        if best_child is None:
            raise RuntimeError("Cannot select a child from an empty node")
        return best_action, best_child

    def select_leaf(self) -> Leaf:
        board = self.board.copy()
        repetitions = Counter(self.repetitions)
        node = self.root
        path = [node]

        while node.expanded and node.children:
            _, child = self._select_child(node)
            if child.move is None:
                raise RuntimeError("Non-root node has no move")
            board.apply(child.move)
            repetitions[repetition_key(board)] += 1
            node = child
            path.append(node)

        terminal = self.terminal_value(board)
        current_count = repetitions[repetition_key(board)]
        return Leaf(
            tree=self,
            board=board,
            path=path,
            repetition_count=current_count,
            terminal_value=terminal,
        )

    def expand_and_backup(
        self,
        leaf: Leaf,
        policy_logits: np.ndarray | None,
        value: float,
    ) -> None:
        node = leaf.path[-1]

        if leaf.terminal_value is None and not node.expanded:
            if policy_logits is None:
                raise ValueError("Non-terminal leaf requires policy logits")
            actions, moves = legal_action_map(leaf.board)
            if not actions:
                raise RuntimeError("Non-terminal board has no legal moves")

            legal_logits = policy_logits[np.asarray(actions, dtype=np.int64)]
            legal_logits = legal_logits - np.max(legal_logits)
            priors = np.exp(legal_logits)
            prior_sum = float(priors.sum())
            if not math.isfinite(prior_sum) or prior_sum <= 0.0:
                priors = np.full(len(actions), 1.0 / len(actions), dtype=np.float64)
            else:
                priors = priors / prior_sum

            child_to_play = opposite(leaf.board.turn)
            node.children = {
                action: Node(
                    to_play=child_to_play,
                    prior=float(prior),
                    move=move,
                )
                for action, move, prior in zip(actions, moves, priors)
            }
            node.expanded = True

            if node is self.root:
                self.add_root_noise()

        leaf_to_play = leaf.board.turn
        for path_node in reversed(leaf.path):
            path_node.visit_count += 1
            if path_node.to_play == leaf_to_play:
                path_node.value_sum += value
            else:
                path_node.value_sum -= value

    def add_root_noise(self) -> None:
        if self.root_noise_applied or not self.root.children:
            return
        children = list(self.root.children.values())
        noise = self.rng.dirichlet(
            [self.config.dirichlet_alpha] * len(children)
        )
        epsilon = self.config.dirichlet_epsilon
        for child, sample in zip(children, noise):
            child.prior = (1.0 - epsilon) * child.prior + epsilon * float(sample)
        self.root_noise_applied = True

    def policy_target(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.root.children:
            raise RuntimeError("Search root is not expanded")
        actions = np.fromiter(self.root.children.keys(), dtype=np.uint16)
        visits = np.fromiter(
            (child.visit_count for child in self.root.children.values()),
            dtype=np.float64,
        )
        total = float(visits.sum())
        if total <= 0:
            probabilities = np.full(len(visits), 1.0 / len(visits), dtype=np.float32)
        else:
            probabilities = (visits / total).astype(np.float32)
        return actions, probabilities

    def select_move(
        self,
        ply: int,
        temperature_moves: int,
        temperature: float,
    ) -> tuple[int, bulletchess.Move]:
        actions = list(self.root.children.keys())
        children = list(self.root.children.values())
        visits = np.asarray([child.visit_count for child in children], dtype=np.float64)

        if ply >= temperature_moves or temperature <= 1e-8:
            selected = int(np.argmax(visits))
        else:
            scaled = np.power(visits, 1.0 / temperature)
            if float(scaled.sum()) <= 0.0:
                scaled = np.ones_like(scaled)
            scaled /= scaled.sum()
            selected = int(self.rng.choice(len(actions), p=scaled))

        move = children[selected].move
        if move is None:
            raise RuntimeError("Selected child has no move")
        return actions[selected], move

    def advance(self, action: int, move: bulletchess.Move) -> None:
        child = self.root.children.get(action)
        if child is None or child.move != move:
            child = Node(to_play=opposite(self.board.turn), prior=1.0, move=move)
        self.board.apply(move)
        self.repetitions[repetition_key(self.board)] += 1
        child.prior = 1.0
        self.root = child
        self.root_noise_applied = False


def run_batched_search(
    trees: list[SearchTree],
    evaluator: NeuralEvaluator,
    simulations: int | None = None,
) -> None:
    if not trees:
        return
    additional = simulations if simulations is not None else trees[0].config.simulations
    targets = {id(tree): tree.root.visit_count + additional for tree in trees}
    for tree in trees:
        if tree.root.expanded:
            tree.add_root_noise()

    while True:
        pending = [
            tree
            for tree in trees
            if tree.root.visit_count < targets[id(tree)]
            and SearchTree.terminal_value(tree.board) is None
        ]
        if not pending:
            return

        leaves = [tree.select_leaf() for tree in pending]
        evaluable: list[Leaf] = []

        for leaf in leaves:
            if leaf.terminal_value is not None:
                leaf.tree.expand_and_backup(
                    leaf,
                    policy_logits=None,
                    value=leaf.terminal_value,
                )
            else:
                evaluable.append(leaf)

        if not evaluable:
            continue

        states = [
            encode_board(
                leaf.board,
                prior_board=previous_board(leaf.board),
                repetition_count=leaf.repetition_count,
            )
            for leaf in evaluable
        ]
        policies, values = evaluator.evaluate(states)

        for leaf, policy, value in zip(evaluable, policies, values):
            leaf.tree.expand_and_backup(
                leaf,
                policy_logits=policy,
                value=float(value),
            )
