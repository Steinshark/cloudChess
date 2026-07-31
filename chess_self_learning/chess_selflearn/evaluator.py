from __future__ import annotations

import contextlib
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn


@dataclass(slots=True)
class EvaluatorConfig:
    precision: str = "fp16"
    channels_last: bool = True
    max_batch_size: int = 128


@dataclass(slots=True)
class EvaluatorMetrics:
    calls: int = 0
    forward_batches: int = 0
    requested_states: int = 0
    unique_states: int = 0
    gathered_legal_logits: int = 0
    inference_seconds: float = 0.0
    maximum_requested_batch: int = 0
    maximum_unique_batch: int = 0

    def as_dict(self) -> dict[str, int | float]:
        payload = asdict(self)
        payload["mean_requested_batch"] = (
            self.requested_states / self.calls if self.calls else 0.0
        )
        payload["mean_unique_batch"] = (
            self.unique_states / self.forward_batches
            if self.forward_batches
            else 0.0
        )
        payload["duplicate_fraction"] = (
            1.0 - self.unique_states / self.requested_states
            if self.requested_states
            else 0.0
        )
        payload["unique_states_per_second"] = (
            self.unique_states / self.inference_seconds
            if self.inference_seconds > 0.0
            else 0.0
        )
        return payload


class NeuralEvaluator:
    """Batched neural inference that returns only logits for legal actions."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        config: EvaluatorConfig,
    ) -> None:
        self.model = model
        self.device = device
        self.config = config
        self.metrics = EvaluatorMetrics()
        self.model.eval()

    @property
    def max_batch_size(self) -> int:
        return max(1, self.config.max_batch_size)

    def _autocast(self):
        if self.device.type != "cuda" or self.config.precision == "fp32":
            return contextlib.nullcontext()
        dtype = (
            torch.float16
            if self.config.precision == "fp16"
            else torch.bfloat16
        )
        return torch.autocast(device_type="cuda", dtype=dtype)

    @torch.inference_mode()
    def evaluate(
        self,
        states: list[np.ndarray],
        legal_actions: list[np.ndarray],
    ) -> tuple[list[np.ndarray], np.ndarray]:
        """Evaluate states and gather variable-length legal logits on the GPU.

        ``legal_actions[index]`` must correspond to ``states[index]``. Exact
        duplicate state/action pairs are evaluated once and mapped back without
        ever transferring a dense [batch, 4672] policy tensor to the CPU.
        """
        if len(states) != len(legal_actions):
            raise ValueError("states and legal_actions must have equal length")
        if not states:
            return [], np.empty((0,), dtype=np.float32)

        self.metrics.calls += 1
        self.metrics.requested_states += len(states)
        self.metrics.maximum_requested_batch = max(
            self.metrics.maximum_requested_batch,
            len(states),
        )

        unique_states: list[np.ndarray] = []
        unique_actions: list[np.ndarray] = []
        key_to_index: dict[tuple[bytes, bytes], int] = {}
        inverse = np.empty(len(states), dtype=np.int64)

        for index, (state, actions) in enumerate(zip(states, legal_actions)):
            state_array = np.asarray(state, dtype=np.uint8)
            action_array = np.asarray(actions, dtype=np.int64).reshape(-1)
            if state_array.shape != (34, 8, 8):
                raise ValueError(
                    f"Expected state shape (34, 8, 8), got {state_array.shape}"
                )
            if action_array.size == 0:
                raise ValueError("A non-terminal evaluation has no legal actions")
            if np.any((action_array < 0) | (action_array >= 4672)):
                raise ValueError("Legal action index lies outside [0, 4671]")

            key = (state_array.tobytes(), action_array.tobytes())
            unique_index = key_to_index.get(key)
            if unique_index is None:
                unique_index = len(unique_states)
                key_to_index[key] = unique_index
                unique_states.append(state_array)
                unique_actions.append(action_array)
            inverse[index] = unique_index

        unique_count = len(unique_states)
        self.metrics.unique_states += unique_count
        self.metrics.maximum_unique_batch = max(
            self.metrics.maximum_unique_batch,
            min(unique_count, self.max_batch_size),
        )

        unique_legal_logits: list[np.ndarray | None] = [None] * unique_count
        unique_values = np.empty(unique_count, dtype=np.float32)

        for start in range(0, unique_count, self.max_batch_size):
            stop = min(start + self.max_batch_size, unique_count)
            chunk_states = unique_states[start:stop]
            chunk_actions = unique_actions[start:stop]
            counts = np.fromiter(
                (actions.size for actions in chunk_actions),
                dtype=np.int64,
                count=len(chunk_actions),
            )
            maximum_legal = int(counts.max())

            state_array = np.stack(chunk_states, axis=0)
            padded_actions = np.zeros(
                (len(chunk_actions), maximum_legal),
                dtype=np.int64,
            )
            for row, actions in enumerate(chunk_actions):
                padded_actions[row, : actions.size] = actions

            started = time.perf_counter()

            # Transfer compact uint8 planes, then normalize on the GPU.
            tensor = torch.from_numpy(state_array).to(
                self.device,
                non_blocking=True,
            )
            tensor = tensor.to(dtype=torch.float32).mul_(1.0 / 255.0)
            if self.config.channels_last:
                tensor = tensor.contiguous(memory_format=torch.channels_last)

            action_tensor = torch.from_numpy(padded_actions).to(
                self.device,
                non_blocking=True,
            )

            with self._autocast():
                dense_policy, value = self.model(tensor)
                gathered_policy = torch.gather(
                    dense_policy,
                    dim=1,
                    index=action_tensor,
                )

            # The CPU copy synchronizes this inference chunk, so wall timing is
            # meaningful without adding a separate cuda.synchronize() call.
            gathered_cpu = gathered_policy.float().cpu().numpy()
            value_cpu = value.float().cpu().numpy()
            elapsed = time.perf_counter() - started

            self.metrics.forward_batches += 1
            self.metrics.inference_seconds += elapsed
            self.metrics.gathered_legal_logits += int(counts.sum())

            for local_row, count in enumerate(counts):
                unique_index = start + local_row
                unique_legal_logits[unique_index] = gathered_cpu[
                    local_row,
                    : int(count),
                ].copy()
                unique_values[unique_index] = value_cpu[local_row]

        resolved_logits: list[np.ndarray] = []
        for unique_index in inverse:
            logits = unique_legal_logits[int(unique_index)]
            if logits is None:
                raise RuntimeError("Evaluator failed to populate legal logits")
            resolved_logits.append(logits)

        return resolved_logits, unique_values[inverse]
