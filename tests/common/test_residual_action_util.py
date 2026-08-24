import numpy as np
from scipy.spatial.transform import Rotation

from diffusion_policy.common.residual_action_util import (
    absolute_action8_to_relative_action10,
    clip_residual_by_norm,
    compose_world_frame_residual,
    compute_world_frame_residual,
    fit_zero_centered_scale,
    relative_action10_to_absolute_action8,
)


def _action_chunk(horizon=16):
    action = np.zeros((horizon, 8), dtype=np.float64)
    action[:, :3] = np.array([0.4, -0.2, 0.3])
    action[:, 3:7] = Rotation.identity().as_quat()
    action[:, 7] = 0.04
    return action


def test_world_residual_round_trip():
    base = _action_chunk()
    edited = base.copy()
    edited[5:, 0] += 0.02
    edited[5:, 3:7] = (
        Rotation.from_rotvec(np.tile([0.0, 0.1, 0.0], (11, 1)))
        * Rotation.from_quat(base[5:, 3:7])
    ).as_quat()
    residual = compute_world_frame_residual(base, edited)
    recomposed = compose_world_frame_residual(base, residual)
    np.testing.assert_allclose(recomposed[:, :3], edited[:, :3], atol=1e-7)
    rotation_error = (
        Rotation.from_quat(recomposed[:, 3:7]).inv()
        * Rotation.from_quat(edited[:, 3:7])
    ).magnitude()
    assert rotation_error.max() < 1e-7


def test_absolute_action_to_relative_base_frame():
    base_pose = np.eye(4)
    base_pose[:3, 3] = [0.3, 0.1, -0.2]
    action = _action_chunk(2)
    relative = absolute_action8_to_relative_action10(action, base_pose)
    np.testing.assert_allclose(
        relative[:, :3], action[:, :3] - base_pose[:3, 3], atol=1e-7
    )
    assert relative.shape == (2, 10)
    round_trip = relative_action10_to_absolute_action8(relative, base_pose)
    np.testing.assert_allclose(round_trip[:, :3], action[:, :3], atol=1e-7)
    rotation_error = (
        Rotation.from_quat(round_trip[:, 3:7]).inv()
        * Rotation.from_quat(action[:, 3:7])
    ).magnitude()
    assert rotation_error.max() < 1e-7
    np.testing.assert_allclose(round_trip[:, 7], action[:, 7], atol=1e-7)


def test_zero_centered_scale_and_norm_clipping():
    residual = np.array(
        [[0.01, -0.02, 0.0, 0.1, 0.0, 0.0, 0.005]], dtype=np.float32
    )
    scale, _ = fit_zero_centered_scale(residual, quantile=1.0)
    normalized_zero = np.zeros(7, dtype=np.float32) * scale
    np.testing.assert_array_equal(normalized_zero, np.zeros(7))
    clipped = clip_residual_by_norm(
        residual, max_position_m=0.01, max_rotation_rad=0.05, max_gripper_m=0.002
    )
    assert np.linalg.norm(clipped[0, :3]) <= 0.010001
    assert np.linalg.norm(clipped[0, 3:6]) <= 0.050001
    assert abs(clipped[0, 6]) <= 0.002001
