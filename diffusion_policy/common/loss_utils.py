from typing import Optional

import torch


def reduce_action_loss(
    elementwise_loss: torch.Tensor,
    action_valid_mask: Optional[torch.Tensor] = None,
    action_loss_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Reduce an elementwise action loss with optional token masks/weights.

    ``action_valid_mask`` and ``action_loss_weight`` are both shaped ``[B, H]``
    and combine multiplicatively.  With neither field present this is exactly a
    mean over every element, matching the legacy diffusion-policy objective.
    """
    if action_valid_mask is None and action_loss_weight is None:
        return elementwise_loss.mean()

    effective_weight = None
    for value in (action_valid_mask, action_loss_weight):
        if value is None:
            continue
        value = value.to(
            device=elementwise_loss.device,
            dtype=elementwise_loss.dtype,
        )
        if torch.any(value < 0):
            raise ValueError("Action masks and loss weights must be non-negative")
        effective_weight = value if effective_weight is None else effective_weight * value

    while effective_weight.ndim < elementwise_loss.ndim:
        effective_weight = effective_weight.unsqueeze(-1)
    expanded_weight = effective_weight.expand_as(elementwise_loss)
    denominator = expanded_weight.sum().clamp(min=1.0)
    return (elementwise_loss * expanded_weight).sum() / denominator
