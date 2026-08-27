"""Trainable policy wrapper and losses for chunk-level residual learning."""

from __future__ import annotations

import copy
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_policy.model.common.normalizer import (
    SingleFieldLinearNormalizer,
)
from diffusion_policy.model.residual.chunk_residual_mlp import ChunkResidualMLP


class ChunkResidualPolicy(nn.Module):
    """Predict total frozen-base-to-final-edited action residuals."""

    def __init__(
        self,
        model: ChunkResidualMLP,
        correction_loss_weight: float = 1.0,
        zero_loss_weight: float = 1.0,
        gate_loss_weight: float = 1.0,
        invalid_prefix_zero_prior_weight: float = 0.0,
        temporal_smoothness_weight: float = 0.0,
        huber_delta: float = 1.0,
        gate_inference_mode: str = "hard",
        gate_threshold: float = 0.5,
    ) -> None:
        super().__init__()
        self.model = model
        self.correction_loss_weight = float(correction_loss_weight)
        self.zero_loss_weight = float(zero_loss_weight)
        self.gate_loss_weight = float(gate_loss_weight)
        self.invalid_prefix_zero_prior_weight = float(
            invalid_prefix_zero_prior_weight
        )
        self.temporal_smoothness_weight = float(
            temporal_smoothness_weight
        )
        self.huber_delta = float(huber_delta)
        self.gate_inference_mode = str(gate_inference_mode)
        self.gate_threshold = float(gate_threshold)
        self.residual_normalizer = SingleFieldLinearNormalizer()

        if self.gate_inference_mode not in {"hard", "soft", "disabled"}:
            raise ValueError(
                "gate_inference_mode must be hard, soft or disabled"
            )
        if not 0 <= self.gate_threshold <= 1:
            raise ValueError("gate_threshold must be in [0, 1]")
        for name, value in {
            "correction_loss_weight": self.correction_loss_weight,
            "zero_loss_weight": self.zero_loss_weight,
            "gate_loss_weight": self.gate_loss_weight,
            "invalid_prefix_zero_prior_weight": (
                self.invalid_prefix_zero_prior_weight
            ),
            "temporal_smoothness_weight": self.temporal_smoothness_weight,
        }.items():
            if value < 0:
                raise ValueError(f"{name} must be non-negative")

    @property
    def action_horizon(self) -> int:
        return self.model.action_horizon

    @property
    def base_action_horizon(self) -> int:
        return self.model.base_action_horizon

    @property
    def residual_horizon(self) -> int:
        return self.model.residual_horizon

    @property
    def residual_action_dim(self) -> int:
        return self.model.residual_action_dim

    def set_residual_normalizer(
        self, normalizer: SingleFieldLinearNormalizer
    ) -> None:
        self.residual_normalizer.load_state_dict(
            copy.deepcopy(normalizer.state_dict())
        )

    @staticmethod
    def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        while mask.ndim < value.ndim:
            mask = mask.unsqueeze(-1)
        expanded = mask.to(dtype=value.dtype).expand_as(value)
        denominator = expanded.sum().clamp(min=1.0)
        return (value * expanded).sum() / denominator

    def compute_loss(
        self,
        obs_feature: torch.Tensor,
        normalized_base_action: torch.Tensor,
        residual_action: torch.Tensor,
        valid_action_mask: torch.Tensor,
        correction_label: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        prediction = self.model(obs_feature, normalized_base_action)
        predicted_normalized = prediction["residual"]
        target_normalized = self.residual_normalizer.normalize(residual_action)
        valid_action_mask = valid_action_mask.to(dtype=torch.bool)
        correction_label = correction_label.reshape(-1).to(
            dtype=predicted_normalized.dtype
        )
        if valid_action_mask.shape != predicted_normalized.shape[:2]:
            raise ValueError(
                "valid_action_mask must have shape "
                f"{tuple(predicted_normalized.shape[:2])}, got "
                f"{tuple(valid_action_mask.shape)}"
            )
        if correction_label.shape[0] != predicted_normalized.shape[0]:
            raise ValueError("correction_label batch dimension mismatch")

        element_loss = F.huber_loss(
            predicted_normalized,
            target_normalized,
            reduction="none",
            delta=self.huber_delta,
        )
        correction_rows = correction_label > 0.5
        zero_rows = ~correction_rows
        correction_mask = valid_action_mask & correction_rows[:, None]
        zero_mask = torch.ones_like(valid_action_mask) & zero_rows[:, None]
        invalid_prefix_mask = (~valid_action_mask) & correction_rows[:, None]

        correction_loss = self._masked_mean(element_loss, correction_mask)
        zero_loss = self._masked_mean(element_loss, zero_mask)
        prefix_prior_element = F.huber_loss(
            predicted_normalized,
            torch.zeros_like(predicted_normalized),
            reduction="none",
            delta=self.huber_delta,
        )
        invalid_prefix_loss = self._masked_mean(
            prefix_prior_element, invalid_prefix_mask
        )

        if predicted_normalized.shape[1] > 1:
            velocity = torch.diff(predicted_normalized, dim=1)
            adjacent_mask = valid_action_mask[:, 1:] & valid_action_mask[:, :-1]
            adjacent_mask &= correction_rows[:, None]
            smoothness_loss = self._masked_mean(
                velocity.square(), adjacent_mask
            )
        else:
            smoothness_loss = predicted_normalized.sum() * 0.0

        gate_logit = prediction["gate_logit"]
        if gate_logit is not None:
            gate_loss = F.binary_cross_entropy_with_logits(
                gate_logit, correction_label
            )
            gate_probability = torch.sigmoid(gate_logit)
        else:
            gate_loss = predicted_normalized.sum() * 0.0
            gate_probability = torch.ones_like(correction_label)

        total_loss = (
            self.correction_loss_weight * correction_loss
            + self.zero_loss_weight * zero_loss
            + self.gate_loss_weight * gate_loss
            + self.invalid_prefix_zero_prior_weight * invalid_prefix_loss
            + self.temporal_smoothness_weight * smoothness_loss
        )
        predicted_residual = self.residual_normalizer.unnormalize(
            predicted_normalized
        )
        return {
            "loss": total_loss,
            "correction_loss": correction_loss,
            "zero_loss": zero_loss,
            "gate_loss": gate_loss,
            "invalid_prefix_loss": invalid_prefix_loss,
            "smoothness_loss": smoothness_loss,
            "predicted_residual": predicted_residual,
            "predicted_normalized_residual": predicted_normalized,
            "gate_probability": gate_probability,
        }

    def predict_residual(
        self,
        obs_feature: torch.Tensor,
        normalized_base_action: torch.Tensor,
        gate_mode: str | None = None,
        gate_threshold: float | None = None,
    ) -> Dict[str, torch.Tensor]:
        prediction = self.model(obs_feature, normalized_base_action)
        normalized_residual = prediction["residual"]
        residual = self.residual_normalizer.unnormalize(normalized_residual)
        gate_logit = prediction["gate_logit"]
        gate_probability = (
            torch.sigmoid(gate_logit)
            if gate_logit is not None
            else torch.ones(
                residual.shape[0], device=residual.device, dtype=residual.dtype
            )
        )
        mode = self.gate_inference_mode if gate_mode is None else str(gate_mode)
        threshold = (
            self.gate_threshold
            if gate_threshold is None
            else float(gate_threshold)
        )
        if mode == "hard" and gate_logit is not None:
            gate_scale = (gate_probability >= threshold).to(residual.dtype)
        elif mode == "soft" and gate_logit is not None:
            gate_scale = gate_probability.to(residual.dtype)
        elif mode == "disabled" or gate_logit is None:
            gate_scale = torch.ones_like(gate_probability, dtype=residual.dtype)
        else:
            raise ValueError(f"Unsupported gate mode: {mode}")
        applied_residual = residual * gate_scale[:, None, None]
        return {
            "residual": residual,
            "applied_residual": applied_residual,
            "gate_probability": gate_probability,
            "gate_scale": gate_scale,
        }

    def forward(
        self,
        obs_feature: torch.Tensor,
        normalized_base_action: torch.Tensor,
        residual_action: torch.Tensor | None = None,
        valid_action_mask: torch.Tensor | None = None,
        correction_label: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        if residual_action is not None:
            if valid_action_mask is None or correction_label is None:
                raise ValueError(
                    "Training forward requires valid_action_mask and "
                    "correction_label"
                )
            return self.compute_loss(
                obs_feature=obs_feature,
                normalized_base_action=normalized_base_action,
                residual_action=residual_action,
                valid_action_mask=valid_action_mask,
                correction_label=correction_label,
            )
        return self.predict_residual(obs_feature, normalized_base_action)
