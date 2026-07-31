from __future__ import annotations

import contextlib
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass(slots=True)
class EvaluatorConfig:
    precision: str = "fp16"
    channels_last: bool = True
    max_batch_size: int = 64


class NeuralEvaluator:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        config: EvaluatorConfig,
    ) -> None:
        self.model = model
        self.device = device
        self.config = config
        self.model.eval()

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
    ) -> tuple[np.ndarray, np.ndarray]:
        if not states:
            return (
                np.empty((0, 4672), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
            )

        # Concurrent self-play games often reach identical opening positions.
        # Evaluate each exact tensor once, then expand results back to input order.
        unique_states: list[np.ndarray] = []
        key_to_index: dict[bytes, int] = {}
        inverse = np.empty(len(states), dtype=np.int64)
        for index, state in enumerate(states):
            key = state.tobytes()
            unique_index = key_to_index.get(key)
            if unique_index is None:
                unique_index = len(unique_states)
                key_to_index[key] = unique_index
                unique_states.append(state)
            inverse[index] = unique_index

        policy_parts: list[np.ndarray] = []
        value_parts: list[np.ndarray] = []
        batch_size = max(1, self.config.max_batch_size)

        for start in range(0, len(unique_states), batch_size):
            array = np.stack(unique_states[start : start + batch_size], axis=0)
            tensor = torch.from_numpy(array).to(
                self.device,
                dtype=torch.float32,
                non_blocking=True,
            )
            tensor.div_(255.0)
            if self.config.channels_last:
                tensor = tensor.contiguous(memory_format=torch.channels_last)

            with self._autocast():
                policy, value = self.model(tensor)

            policy_parts.append(policy.float().cpu().numpy())
            value_parts.append(value.float().cpu().numpy())

        unique_policy = np.concatenate(policy_parts, axis=0)
        unique_value = np.concatenate(value_parts, axis=0)
        return unique_policy[inverse], unique_value[inverse]
