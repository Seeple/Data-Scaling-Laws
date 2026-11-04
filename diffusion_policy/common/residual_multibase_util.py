"""Geometric acceptance tests for filtered multi-base residual augmentation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from diffusion_policy.common.residual_action_util import (
    compute_world_frame_residual,
    normalize_quaternion_xyzw,
    validate_absolute_action_chunk,
)


@dataclass(frozen=True)
class MultiBaseFilterConfig:
    comparison_steps: int = 5
    max_first_position_deviation_m: float = 0.025
    max_prefix_position_deviation_m: float = 0.050
    max_prefix_rotation_deviation_rad: float = 0.35
    max_prefix_gripper_deviation_m: float = 0.030
    max_target_position_residual_m: float = 0.100
    max_target_rotation_residual_rad: float = 0.80
    max_target_gripper_residual_m: float = 0.050
    min_diversity_position_m: float = 0.001
    min_diversity_rotation_rad: float = 0.01

    def __post_init__(self) -> None:
        if self.comparison_steps <= 0:
            raise ValueError("comparison_steps must be positive")
        for name, value in asdict(self).items():
            if name == "comparison_steps":
                continue
            if float(value) < 0:
                raise ValueError(f"{name} must be non-negative")


def _rotation_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_rotation = Rotation.from_quat(normalize_quaternion_xyzw(left))
    right_rotation = Rotation.from_quat(normalize_quaternion_xyzw(right))
    return (left_rotation.inv() * right_rotation).magnitude()


def evaluate_multibase_candidate(
    candidate_base: np.ndarray,
    recorded_base: np.ndarray,
    edited_goal: np.ndarray,
    valid_action_mask: np.ndarray,
    accepted_bases: Sequence[np.ndarray],
    config: MultiBaseFilterConfig,
) -> tuple[bool, str, Dict[str, float]]:
    """Accept only candidates for which the recorded edited goal stays valid.

    Compatibility is measured against the actually observed active base plan.
    A second bound limits the resulting residual to the edited goal, and a
    diversity test prevents spending dataset capacity on duplicate DP samples.
    """
    candidate = validate_absolute_action_chunk(candidate_base)
    recorded = validate_absolute_action_chunk(
        recorded_base, horizon=candidate.shape[0]
    )
    edited = validate_absolute_action_chunk(
        edited_goal, horizon=candidate.shape[0]
    )
    mask = np.asarray(valid_action_mask, dtype=bool).reshape(-1)
    if mask.shape != (candidate.shape[0],) or not np.any(mask):
        raise ValueError("valid_action_mask must select at least one waypoint")
    valid_indices = np.flatnonzero(mask)
    compare_indices = valid_indices[: min(config.comparison_steps, len(valid_indices))]

    position_deviation = np.linalg.norm(
        candidate[compare_indices, :3] - recorded[compare_indices, :3], axis=-1
    )
    rotation_deviation = _rotation_distance(
        candidate[compare_indices, 3:7], recorded[compare_indices, 3:7]
    )
    gripper_deviation = np.abs(
        candidate[compare_indices, 7] - recorded[compare_indices, 7]
    )
    residual = compute_world_frame_residual(candidate, edited)
    target_position = np.linalg.norm(residual[mask, :3], axis=-1)
    target_rotation = np.linalg.norm(residual[mask, 3:6], axis=-1)
    target_gripper = np.abs(residual[mask, 6])
    metrics = {
        "first_position_deviation_m": float(position_deviation[0]),
        "max_prefix_position_deviation_m": float(position_deviation.max()),
        "max_prefix_rotation_deviation_rad": float(rotation_deviation.max()),
        "max_prefix_gripper_deviation_m": float(gripper_deviation.max()),
        "max_target_position_residual_m": float(target_position.max()),
        "max_target_rotation_residual_rad": float(target_rotation.max()),
        "max_target_gripper_residual_m": float(target_gripper.max()),
    }
    checks = (
        (
            metrics["first_position_deviation_m"]
            <= config.max_first_position_deviation_m,
            "first_position_deviation",
        ),
        (
            metrics["max_prefix_position_deviation_m"]
            <= config.max_prefix_position_deviation_m,
            "prefix_position_deviation",
        ),
        (
            metrics["max_prefix_rotation_deviation_rad"]
            <= config.max_prefix_rotation_deviation_rad,
            "prefix_rotation_deviation",
        ),
        (
            metrics["max_prefix_gripper_deviation_m"]
            <= config.max_prefix_gripper_deviation_m,
            "prefix_gripper_deviation",
        ),
        (
            metrics["max_target_position_residual_m"]
            <= config.max_target_position_residual_m,
            "target_position_residual",
        ),
        (
            metrics["max_target_rotation_residual_rad"]
            <= config.max_target_rotation_residual_rad,
            "target_rotation_residual",
        ),
        (
            metrics["max_target_gripper_residual_m"]
            <= config.max_target_gripper_residual_m,
            "target_gripper_residual",
        ),
    )
    for accepted, reason in checks:
        if not accepted:
            return False, reason, metrics

    for previous in accepted_bases:
        previous = validate_absolute_action_chunk(
            previous, horizon=candidate.shape[0]
        )
        position_diversity = float(
            np.max(
                np.linalg.norm(
                    candidate[compare_indices, :3]
                    - previous[compare_indices, :3],
                    axis=-1,
                )
            )
        )
        rotation_diversity = float(
            np.max(
                _rotation_distance(
                    candidate[compare_indices, 3:7],
                    previous[compare_indices, 3:7],
                )
            )
        )
        if (
            position_diversity < config.min_diversity_position_m
            and rotation_diversity < config.min_diversity_rotation_rad
        ):
            metrics["position_diversity_m"] = position_diversity
            metrics["rotation_diversity_rad"] = rotation_diversity
            return False, "insufficient_diversity", metrics
    return True, "accepted", metrics
