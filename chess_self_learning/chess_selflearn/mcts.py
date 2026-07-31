from __future__ import annotations

import math
import time
from collections import Counter
from dataclasses import asdict, dataclass, field

import bulletchess
import numpy as np

from .encoding import (
    encode_board_with_history,
    legal_action_map,
    opposite,
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
    leaves_per_tree: int = 4
    virtual_loss: float = 1.0


@dataclass(slots=True)
class Node:
    to_play: object
    prior: float = 0.0
    move: bulletchess.Move | None = None
    visit_count: int = 0
    value_sum: float = 0.0
    expanded: bool = False
    children: dict[int, "Node"] = field(default_factory=dict)

    # Outstanding batched searches reserve paths with virtual visits/loss.
    virtual_visits: int = 0
    in_flight: bool = False

    # PUCT's parent-only log/sqrt term is shared by every child. Cache it for
    # the current effective visit count instead of recomputing per child.
    puct_cache_visits: int = -1
    puct_cache_scale: float = 0.0

    @property
    def mean_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


@dataclass(slots=True)
class Leaf:
    tree: "SearchTree"
    path: tuple[Node, ...]
    leaf_to_play: object
    repetition_count: int
    terminal_value: float | None
    state: np.ndarray | None
    legal_actions: np.ndarray | None
    legal_moves: tuple[bulletchess.Move, ...]
    reserved: bool = True


@dataclass(slots=True)
class SearchMetrics:
    calls: int = 0
    inference_batches: int = 0
    simulations_completed: int = 0
    neural_leaves: int = 0
    terminal_leaves: int = 0
    requested_evaluation_states: int = 0
    maximum_evaluation_batch: int = 0
    selection_seconds: float = 0.0
    evaluation_seconds: float = 0.0
    backup_seconds: float = 0.0

    def merge(self, other: "SearchMetrics") -> None:
        self.calls += other.calls
        self.inference_batches += other.inference_batches
        self.simulations_completed += other.simulations_completed
        self.neural_leaves += other.neural_leaves
        self.terminal_leaves += other.terminal_leaves
        self.requested_evaluation_states += other.requested_evaluation_states
        self.maximum_evaluation_batch = max(
            self.maximum_evaluation_batch,
            other.maximum_evaluation_batch,
        )
        self.selection_seconds += other.selection_seconds
        self.evaluation_seconds += other.evaluation_seconds
        self.backup_seconds += other.backup_seconds

    def as_dict(self, elapsed_seconds: float | None = None) -> dict[str, int | float]:
        payload = asdict(self)
        payload["mean_evaluation_batch"] = (
            self.requested_evaluation_states / self.inference_batches
            if self.inference_batches
            else 0.0
        )
        if elapsed_seconds is not None and elapsed_seconds > 0.0:
            payload["simulations_per_second"] = (
                self.simulations_completed / elapsed_seconds
            )
            payload["neural_leaves_per_second"] = (
                self.neural_leaves / elapsed_seconds
            )
        return payload


class SearchTree:
    """Mutable MCTS tree with one reusable scratch board.

    ``select_leaf`` applies a path directly to ``self.board``, captures the
    encoded state and legal moves, then undoes the path before returning. There
    is no per-simulation ``Board.copy()`` and no copied repetition ``Counter``.
    """

    def __init__(
        self,
        board: bulletchess.Board,
        repetitions: Counter[str],
        config: SearchConfig,
        rng: np.random.Generator,
    ) -> None:
        # One ownership copy at tree construction is intentional. All search
        # simulations after this use apply/undo on this same board.
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

    def _parent_exploration_scale(self, parent: Node) -> float:
        effective_visits = max(1, parent.visit_count + parent.virtual_visits)
        if parent.puct_cache_visits != effective_visits:
            pb_c = math.log(
                (effective_visits + self.config.c_puct_base + 1.0)
                / self.config.c_puct_base
            ) + self.config.c_puct_init
            parent.puct_cache_visits = effective_visits
            parent.puct_cache_scale = pb_c * math.sqrt(effective_visits)
        return parent.puct_cache_scale

    def _select_child(self, node: Node) -> tuple[int, Node] | None:
        best_action = -1
        best_child: Node | None = None
        best_score = -float("inf")
        exploration_scale = self._parent_exploration_scale(node)
        virtual_loss = self.config.virtual_loss

        for action, child in node.children.items():
            # An unexpanded node can only have one outstanding neural request.
            # Other selections are steered to another branch until it returns.
            if child.in_flight and not child.expanded:
                continue

            effective_visits = child.visit_count + child.virtual_visits
            if effective_visits:
                effective_value_sum = (
                    child.value_sum + child.virtual_visits * virtual_loss
                )
                q_value = -(effective_value_sum / effective_visits)
            else:
                q_value = 0.0

            exploration = (
                exploration_scale
                * child.prior
                / (effective_visits + 1)
            )
            score = q_value + exploration
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        if best_child is None:
            return None
        return best_action, best_child

    @staticmethod
    def _reserve_path(path: tuple[Node, ...]) -> None:
        for node in path:
            node.virtual_visits += 1
        path[-1].in_flight = True

    @staticmethod
    def _release_path(leaf: Leaf) -> None:
        if not leaf.reserved:
            return
        for node in leaf.path:
            node.virtual_visits -= 1
            if node.virtual_visits < 0:
                raise RuntimeError("MCTS virtual visit count became negative")
        leaf.path[-1].in_flight = False
        leaf.reserved = False

    def cancel_leaf(self, leaf: Leaf) -> None:
        """Release a reservation after an evaluator or scheduler exception."""
        self._release_path(leaf)

    def select_leaf(self) -> Leaf | None:
        board = self.board
        node = self.root
        path: list[Node] = [node]
        applied_moves = 0
        path_repetitions: Counter[str] = Counter()

        try:
            if node.in_flight and not node.expanded:
                return None

            while node.expanded and node.children:
                selected = self._select_child(node)
                if selected is None:
                    return None
                _, child = selected
                if child.move is None:
                    raise RuntimeError("Non-root node has no move")

                board.apply(child.move)
                applied_moves += 1
                path_repetitions[repetition_key(board)] += 1
                node = child
                path.append(node)

            if node.in_flight and not node.expanded:
                return None

            terminal = self.terminal_value(board)
            leaf_key = repetition_key(board)
            repetition_count = (
                self.repetitions.get(leaf_key, 0)
                + path_repetitions.get(leaf_key, 0)
            )
            leaf_to_play = board.turn

            state: np.ndarray | None = None
            legal_actions: np.ndarray | None = None
            legal_moves: tuple[bulletchess.Move, ...] = ()
            if terminal is None:
                actions, moves = legal_action_map(board)
                if actions.size == 0:
                    raise RuntimeError("Non-terminal board has no legal moves")
                state = encode_board_with_history(
                    board,
                    repetition_count=repetition_count,
                )
                legal_actions = actions
                legal_moves = tuple(moves)

            leaf = Leaf(
                tree=self,
                path=tuple(path),
                leaf_to_play=leaf_to_play,
                repetition_count=repetition_count,
                terminal_value=terminal,
                state=state,
                legal_actions=legal_actions,
                legal_moves=legal_moves,
            )
            self._reserve_path(leaf.path)
            return leaf
        finally:
            for _ in range(applied_moves):
                board.undo()

    def expand_and_backup(
        self,
        leaf: Leaf,
        legal_policy_logits: np.ndarray | None,
        value: float,
    ) -> None:
        node = leaf.path[-1]
        self._release_path(leaf)

        if leaf.terminal_value is None and not node.expanded:
            if legal_policy_logits is None or leaf.legal_actions is None:
                raise ValueError("Non-terminal leaf requires legal policy logits")
            if len(legal_policy_logits) != len(leaf.legal_actions):
                raise ValueError(
                    "Legal policy-logit count does not match legal action count"
                )
            if len(leaf.legal_moves) != len(leaf.legal_actions):
                raise ValueError("Legal move/action counts do not match")

            logits = np.asarray(legal_policy_logits, dtype=np.float32)
            logits = logits - np.max(logits)
            priors = np.exp(logits)
            prior_sum = float(priors.sum())
            if not math.isfinite(prior_sum) or prior_sum <= 0.0:
                priors = np.full(
                    len(leaf.legal_actions),
                    1.0 / len(leaf.legal_actions),
                    dtype=np.float32,
                )
            else:
                priors = priors / prior_sum

            child_to_play = opposite(leaf.leaf_to_play)
            node.children = {
                int(action): Node(
                    to_play=child_to_play,
                    prior=float(prior),
                    move=move,
                )
                for action, move, prior in zip(
                    leaf.legal_actions,
                    leaf.legal_moves,
                    priors,
                )
            }
            node.expanded = True

            if node is self.root:
                self.add_root_noise()

        for path_node in reversed(leaf.path):
            path_node.visit_count += 1
            if path_node.to_play == leaf.leaf_to_play:
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
            probabilities = np.full(
                len(visits),
                1.0 / len(visits),
                dtype=np.float32,
            )
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
        visits = np.asarray(
            [child.visit_count for child in children],
            dtype=np.float64,
        )

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
        if child.virtual_visits:
            raise RuntimeError("Cannot advance a tree with outstanding searches")

        self.board.apply(move)
        self.repetitions[repetition_key(self.board)] += 1
        child.prior = 1.0
        child.in_flight = False
        self.root = child
        self.root_noise_applied = False


def run_batched_search(
    trees: list[SearchTree],
    evaluator: NeuralEvaluator,
    simulations: int | None = None,
) -> SearchMetrics:
    """Run MCTS using multiple virtually reserved leaves from every tree."""
    metrics = SearchMetrics(calls=1)
    if not trees:
        return metrics

    additional = simulations if simulations is not None else trees[0].config.simulations
    targets = {id(tree): tree.root.visit_count + additional for tree in trees}
    for tree in trees:
        if tree.root.expanded:
            tree.add_root_noise()

    while True:
        # Active game roots are known non-terminal by their owners. Avoid the
        # old repeated root status test on every scheduler round.
        pending = [
            tree
            for tree in trees
            if tree.root.visit_count + tree.root.virtual_visits
            < targets[id(tree)]
        ]
        if not pending:
            return metrics

        evaluable: list[Leaf] = []
        made_progress = False
        selection_started = time.perf_counter()

        for tree in pending:
            per_tree_limit = max(1, tree.config.leaves_per_tree)
            selected_for_tree = 0

            while (
                selected_for_tree < per_tree_limit
                and len(evaluable) < evaluator.max_batch_size
                and tree.root.visit_count + tree.root.virtual_visits
                < targets[id(tree)]
            ):
                leaf = tree.select_leaf()
                if leaf is None:
                    break

                made_progress = True
                selected_for_tree += 1
                if leaf.terminal_value is not None:
                    backup_started = time.perf_counter()
                    tree.expand_and_backup(
                        leaf,
                        legal_policy_logits=None,
                        value=leaf.terminal_value,
                    )
                    metrics.backup_seconds += time.perf_counter() - backup_started
                    metrics.terminal_leaves += 1
                    metrics.simulations_completed += 1
                else:
                    evaluable.append(leaf)

            if len(evaluable) >= evaluator.max_batch_size:
                break

        metrics.selection_seconds += time.perf_counter() - selection_started

        if not evaluable:
            if made_progress:
                continue
            raise RuntimeError(
                "MCTS scheduler made no progress; outstanding leaf reservations "
                "were not released"
            )

        states: list[np.ndarray] = []
        action_lists: list[np.ndarray] = []
        for leaf in evaluable:
            if leaf.state is None or leaf.legal_actions is None:
                raise RuntimeError("Evaluable leaf is missing state/actions")
            states.append(leaf.state)
            action_lists.append(leaf.legal_actions)

        metrics.inference_batches += 1
        metrics.neural_leaves += len(evaluable)
        metrics.requested_evaluation_states += len(evaluable)
        metrics.maximum_evaluation_batch = max(
            metrics.maximum_evaluation_batch,
            len(evaluable),
        )

        evaluation_started = time.perf_counter()
        try:
            legal_policies, values = evaluator.evaluate(states, action_lists)
        except Exception:
            for leaf in evaluable:
                leaf.tree.cancel_leaf(leaf)
            raise
        metrics.evaluation_seconds += time.perf_counter() - evaluation_started

        backup_started = time.perf_counter()
        for leaf, legal_policy, value in zip(
            evaluable,
            legal_policies,
            values,
        ):
            leaf.tree.expand_and_backup(
                leaf,
                legal_policy_logits=legal_policy,
                value=float(value),
            )
        metrics.backup_seconds += time.perf_counter() - backup_started
        metrics.simulations_completed += len(evaluable)
