"""Small deterministic action-chunk residual model."""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn


def _mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    dropout: float,
) -> nn.Sequential:
    layers = []
    previous = int(input_dim)
    for hidden in hidden_dims:
        hidden = int(hidden)
        layers.extend(
            [
                nn.Linear(previous, hidden),
                nn.GELU(),
                nn.LayerNorm(hidden),
                nn.Dropout(float(dropout)),
            ]
        )
        previous = hidden
    return nn.Sequential(*layers)


class ChunkResidualMLP(nn.Module):
    """Predict an H-step geometric residual and an optional chunk gate."""

    def __init__(
        self,
        obs_feature_dim: int,
        action_horizon: Optional[int] = 16,
        base_action_horizon: Optional[int] = None,
        residual_horizon: Optional[int] = None,
        base_action_dim: int = 10,
        residual_action_dim: int = 7,
        obs_projection_dim: int = 256,
        base_chunk_feature_dim: int = 256,
        hidden_dims: Sequence[int] = (512, 512, 256),
        dropout: float = 0.1,
        condition_on_base_action: bool = True,
        gate_enabled: bool = False,
        zero_initialize_residual_head: bool = True,
        gate_initial_probability: float = 0.1,
    ) -> None:
        super().__init__()
        self.obs_feature_dim = int(obs_feature_dim)
        if base_action_horizon is None:
            if action_horizon is None:
                raise ValueError(
                    "base_action_horizon or action_horizon must be provided"
                )
            base_action_horizon = action_horizon
        if residual_horizon is None:
            if action_horizon is None:
                raise ValueError(
                    "residual_horizon or action_horizon must be provided"
                )
            residual_horizon = action_horizon
        self.base_action_horizon = int(base_action_horizon)
        self.residual_horizon = int(residual_horizon)
        if self.base_action_horizon <= 0 or self.residual_horizon <= 0:
            raise ValueError("action horizons must be positive")
        # Historical V1/V2 code and checkpoints use action_horizon for the
        # residual output horizon.  Keep that public attribute while allowing
        # the active base window and predicted residual window to be explicit.
        self.action_horizon = self.residual_horizon
        self.base_action_dim = int(base_action_dim)
        self.residual_action_dim = int(residual_action_dim)
        self.condition_on_base_action = bool(condition_on_base_action)
        self.gate_enabled = bool(gate_enabled)

        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one layer")
        if not 0 < gate_initial_probability < 1:
            raise ValueError("gate_initial_probability must be in (0, 1)")

        self.obs_projection = _mlp(
            self.obs_feature_dim,
            (int(obs_projection_dim),),
            dropout,
        )
        trunk_input_dim = int(obs_projection_dim)
        if self.condition_on_base_action:
            flattened_action_dim = (
                self.base_action_horizon * self.base_action_dim
            )
            self.base_chunk_encoder = nn.Sequential(
                nn.LayerNorm(flattened_action_dim),
                nn.Linear(flattened_action_dim, int(base_chunk_feature_dim)),
                nn.GELU(),
                nn.LayerNorm(int(base_chunk_feature_dim)),
                nn.Dropout(float(dropout)),
            )
            trunk_input_dim += int(base_chunk_feature_dim)
        else:
            self.base_chunk_encoder = None

        self.trunk = _mlp(trunk_input_dim, hidden_dims, dropout)
        last_dim = int(hidden_dims[-1])
        self.residual_head = nn.Linear(
            last_dim, self.residual_horizon * self.residual_action_dim
        )
        self.gate_head = nn.Linear(last_dim, 1) if self.gate_enabled else None

        if zero_initialize_residual_head:
            nn.init.zeros_(self.residual_head.weight)
            nn.init.zeros_(self.residual_head.bias)
        if self.gate_head is not None:
            nn.init.zeros_(self.gate_head.weight)
            initial_logit = torch.logit(
                torch.tensor(float(gate_initial_probability))
            )
            nn.init.constant_(self.gate_head.bias, float(initial_logit))

    def forward(
        self,
        obs_feature: torch.Tensor,
        base_action: torch.Tensor,
    ) -> dict[str, torch.Tensor | None]:
        if obs_feature.ndim != 2 or obs_feature.shape[-1] != self.obs_feature_dim:
            raise ValueError(
                "obs_feature must have shape "
                f"(B, {self.obs_feature_dim}), got {tuple(obs_feature.shape)}"
            )
        if base_action.ndim != 3 or base_action.shape[1:] != (
            self.base_action_horizon,
            self.base_action_dim,
        ):
            raise ValueError(
                "base_action must have shape "
                f"(B, {self.base_action_horizon}, {self.base_action_dim}), got "
                f"{tuple(base_action.shape)}"
            )
        features = [self.obs_projection(obs_feature)]
        if self.base_chunk_encoder is not None:
            features.append(
                self.base_chunk_encoder(base_action.flatten(start_dim=1))
            )
        hidden = self.trunk(torch.cat(features, dim=-1))
        residual = self.residual_head(hidden).reshape(
            -1, self.residual_horizon, self.residual_action_dim
        )
        gate_logit = (
            self.gate_head(hidden).squeeze(-1)
            if self.gate_head is not None
            else None
        )
        return {"residual": residual, "gate_logit": gate_logit}
