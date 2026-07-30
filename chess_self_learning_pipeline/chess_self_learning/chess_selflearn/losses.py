from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def sparse_policy_loss(
    logits: Tensor,
    indices: Tensor,
    probabilities: Tensor,
    counts: Tensor,
) -> Tensor:
    indices = indices.to(logits.device, non_blocking=True)
    probabilities = probabilities.to(
        logits.device,
        dtype=torch.float32,
        non_blocking=True,
    )
    counts = counts.to(logits.device, non_blocking=True)

    width = indices.shape[1]
    mask = torch.arange(width, device=logits.device)[None, :] < counts[:, None]
    safe_indices = torch.where(mask, indices, torch.zeros_like(indices))
    selected_log_probs = F.log_softmax(logits, dim=1).gather(1, safe_indices)
    weighted = probabilities * selected_log_probs * mask
    return -weighted.sum(dim=1).mean()
