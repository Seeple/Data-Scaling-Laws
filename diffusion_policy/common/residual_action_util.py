"""Geometry and normalization helpers for active-plan residual policies.

The residual data contract uses absolute robot commands with layout
``[xyz, quaternion_xyzw, gripper]``.  Translation and rotation residuals are
expressed in the world frame, with rotation composed on the left::

    p_edited = p_base + delta_p
    R_edited = Exp(delta_r) @ R_base

The frozen Diffusion Policy itself uses a relative 10-D representation
``[relative_xyz, relative_rotation_6d, gripper]``.  Helpers in this module are
the single source of truth for converting between those representations.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from diffusion_policy.common.pose_repr_util import convert_pose_mat_rep
from umi.common.pose_util import mat_to_pose10d


ABSOLUTE_ACTION_DIM = 8
BASE_ACTION_DIM = 10
RESIDUAL_ACTION_DIM = 7
TCP_POSE_SLICE = slice(14, 21)
GRIPPER_INDEX = 21


def _as_float_array(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def normalize_quaternion_xyzw(quaternion: np.ndarray) -> np.ndarray:
    """Normalize one or more xyzw quaternions, rejecting degenerate rows."""
    quaternion = _as_float_array(quaternion, "quaternion")
    if quaternion.shape[-1] != 4:
        raise ValueError(
            f"quaternion must end in dimension 4, got {quaternion.shape}"
        )
    norm = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    if np.any(norm <= 1e-8):
        raise ValueError("quaternion contains a zero-norm row")
    return quaternion / norm


def validate_absolute_action_chunk(
    action: np.ndarray,
    horizon: int | None = None,
) -> np.ndarray:
    """Validate and return an absolute ``(H, 8)`` action chunk."""
    action = _as_float_array(action, "absolute action")
    if action.ndim != 2 or action.shape[1] != ABSOLUTE_ACTION_DIM:
        raise ValueError(
            "absolute action must have shape (horizon, 8) with "
            "[xyz, quaternion_xyzw, gripper], got "
            f"{action.shape}"
        )
    if horizon is not None and action.shape[0] != int(horizon):
        raise ValueError(
            f"absolute action horizon mismatch: expected {horizon}, "
            f"got {action.shape[0]}"
        )
    normalize_quaternion_xyzw(action[:, 3:7])
    return action


def flexiv_state_to_pose_matrix(robot_state: np.ndarray) -> np.ndarray:
    """Convert recorded 22-D Rizon states to homogeneous pose matrices.

    The recorded TCP layout is ``[x, y, z, qw, qx, qy, qz]``.
    """
    state = _as_float_array(robot_state, "robot_state")
    if state.shape[-1] <= GRIPPER_INDEX:
        raise ValueError(
            f"robot_state must have at least 22 elements, got {state.shape}"
        )
    tcp = state[..., TCP_POSE_SLICE]
    quaternion_xyzw = np.concatenate(
        [tcp[..., 4:7], tcp[..., 3:4]], axis=-1
    )
    quaternion_xyzw = normalize_quaternion_xyzw(quaternion_xyzw)
    matrix = np.zeros(state.shape[:-1] + (4, 4), dtype=np.float64)
    matrix[..., :3, :3] = Rotation.from_quat(quaternion_xyzw).as_matrix()
    matrix[..., :3, 3] = tcp[..., :3]
    matrix[..., 3, 3] = 1.0
    return matrix


def flexiv_state_to_gripper(robot_state: np.ndarray) -> np.ndarray:
    state = _as_float_array(robot_state, "robot_state")
    if state.shape[-1] <= GRIPPER_INDEX:
        raise ValueError(
            f"robot_state must have at least 22 elements, got {state.shape}"
        )
    return state[..., GRIPPER_INDEX : GRIPPER_INDEX + 1]


def absolute_action8_to_relative_action10(
    absolute_action: np.ndarray,
    latest_observation_pose: np.ndarray,
) -> np.ndarray:
    """Convert an absolute command chunk to the base DP action convention."""
    action = validate_absolute_action_chunk(absolute_action)
    latest_pose = _as_float_array(
        latest_observation_pose, "latest_observation_pose"
    )
    if latest_pose.shape != (4, 4):
        raise ValueError(
            "latest_observation_pose must have shape (4, 4), got "
            f"{latest_pose.shape}"
        )
    action_matrix = np.zeros((len(action), 4, 4), dtype=np.float64)
    action_matrix[:, :3, :3] = Rotation.from_quat(
        normalize_quaternion_xyzw(action[:, 3:7])
    ).as_matrix()
    action_matrix[:, :3, 3] = action[:, :3]
    action_matrix[:, 3, 3] = 1.0
    relative_matrix = convert_pose_mat_rep(
        action_matrix,
        base_pose_mat=latest_pose,
        pose_rep="relative",
        backward=False,
    )
    relative_pose9 = mat_to_pose10d(relative_matrix)
    return np.concatenate([relative_pose9, action[:, 7:8]], axis=-1).astype(
        np.float32
    )


def compute_world_frame_residual(
    base_action: np.ndarray,
    edited_action: np.ndarray,
) -> np.ndarray:
    """Return ``(H, 7)`` world-frame residual from base to edited action."""
    base = validate_absolute_action_chunk(base_action)
    edited = validate_absolute_action_chunk(
        edited_action, horizon=base.shape[0]
    )
    base_rotation = Rotation.from_quat(
        normalize_quaternion_xyzw(base[:, 3:7])
    )
    edited_rotation = Rotation.from_quat(
        normalize_quaternion_xyzw(edited[:, 3:7])
    )
    rotation_residual = (edited_rotation * base_rotation.inv()).as_rotvec()
    return np.concatenate(
        [
            edited[:, :3] - base[:, :3],
            rotation_residual,
            edited[:, 7:8] - base[:, 7:8],
        ],
        axis=-1,
    ).astype(np.float32)


def compose_world_frame_residual(
    base_action: np.ndarray,
    residual_action: np.ndarray,
) -> np.ndarray:
    """Apply a ``(H, 7)`` residual to an absolute ``(H, 8)`` chunk."""
    base = validate_absolute_action_chunk(base_action)
    residual = _as_float_array(residual_action, "residual action")
    if residual.shape != (base.shape[0], RESIDUAL_ACTION_DIM):
        raise ValueError(
            "residual action must have shape "
            f"({base.shape[0]}, 7), got {residual.shape}"
        )
    output = base.copy()
    output[:, :3] += residual[:, :3]
    base_rotation = Rotation.from_quat(
        normalize_quaternion_xyzw(base[:, 3:7])
    )
    output[:, 3:7] = (
        Rotation.from_rotvec(residual[:, 3:6]) * base_rotation
    ).as_quat()
    output[:, 7:8] += residual[:, 6:7]
    return output


def residual_recomposition_errors(
    base_action: np.ndarray,
    edited_action: np.ndarray,
    residual_action: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Return per-waypoint position, rotation and gripper errors."""
    edited = validate_absolute_action_chunk(edited_action)
    recomposed = compose_world_frame_residual(base_action, residual_action)
    position = np.linalg.norm(recomposed[:, :3] - edited[:, :3], axis=-1)
    rotation = (
        Rotation.from_quat(normalize_quaternion_xyzw(recomposed[:, 3:7])).inv()
        * Rotation.from_quat(normalize_quaternion_xyzw(edited[:, 3:7]))
    ).magnitude()
    gripper = np.abs(recomposed[:, 7] - edited[:, 7])
    return {"position": position, "rotation": rotation, "gripper": gripper}


def clip_residual_by_norm(
    residual_action: np.ndarray,
    max_position_m: float | None = None,
    max_rotation_rad: float | None = None,
    max_gripper_m: float | None = None,
) -> np.ndarray:
    """Clip residual vector norms without changing their directions."""
    residual = _as_float_array(residual_action, "residual action").copy()
    if residual.shape[-1] != RESIDUAL_ACTION_DIM:
        raise ValueError(
            f"residual action must end in dimension 7, got {residual.shape}"
        )

    def clip_vector(values: np.ndarray, maximum: float | None) -> np.ndarray:
        if maximum is None:
            return values
        maximum = float(maximum)
        if maximum <= 0:
            raise ValueError("residual norm limits must be positive")
        norms = np.linalg.norm(values, axis=-1, keepdims=True)
        scale = np.minimum(1.0, maximum / np.maximum(norms, 1e-12))
        return values * scale

    residual[..., :3] = clip_vector(residual[..., :3], max_position_m)
    residual[..., 3:6] = clip_vector(
        residual[..., 3:6], max_rotation_rad
    )
    if max_gripper_m is not None:
        maximum = float(max_gripper_m)
        if maximum <= 0:
            raise ValueError("max_gripper_m must be positive")
        residual[..., 6:7] = np.clip(
            residual[..., 6:7], -maximum, maximum
        )
    return residual.astype(np.float32)


def fit_zero_centered_scale(
    residual_action: np.ndarray,
    quantile: float = 0.995,
    minimum_scale: float = 1e-4,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Fit a robust symmetric scale with an exact zero fixed point.

    Returns a multiplier suitable for ``normalized = residual * scale`` and
    input statistics for checkpoint diagnostics.
    """
    residual = _as_float_array(residual_action, "residual action")
    if residual.ndim != 2 or residual.shape[1] != RESIDUAL_ACTION_DIM:
        raise ValueError(
            "residual samples must have shape (N, 7), got "
            f"{residual.shape}"
        )
    if len(residual) == 0:
        raise ValueError("cannot fit residual scale from an empty array")
    if not 0 < quantile <= 1:
        raise ValueError("quantile must be in (0, 1]")
    magnitude = np.quantile(np.abs(residual), quantile, axis=0)
    magnitude = np.maximum(magnitude, float(minimum_scale))
    scale = (1.0 / magnitude).astype(np.float32)
    stats = {
        "min": residual.min(axis=0).astype(np.float32),
        "max": residual.max(axis=0).astype(np.float32),
        "mean": residual.mean(axis=0).astype(np.float32),
        "std": residual.std(axis=0).astype(np.float32),
    }
    return scale, stats
