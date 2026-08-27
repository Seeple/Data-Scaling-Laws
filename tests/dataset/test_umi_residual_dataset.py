from pathlib import Path

import numpy as np
import zarr
from scipy.spatial.transform import Rotation

from diffusion_policy.common.residual_action_util import compute_world_frame_residual
from diffusion_policy.dataset.umi_residual_dataset import UmiResidualDataset


def _shape_meta():
    obs = {
        "camera0_rgb": {"shape": [3, 32, 32], "horizon": 2},
        "robot0_eef_pos": {"shape": [3], "horizon": 2},
        "robot0_eef_rot_axis_angle": {"shape": [6], "horizon": 2},
        "robot0_eef_rot_axis_angle_wrt_start": {
            "shape": [6],
            "horizon": 2,
        },
        "robot0_gripper_width": {"shape": [1], "horizon": 2},
    }
    return {"obs": obs, "action": {"shape": [10], "horizon": 16}}


def _write_dataset(path: Path, active_suffix: bool = False):
    sample_count, horizon = 4, 16
    base = np.zeros((sample_count, horizon, 8), dtype=np.float32)
    base[..., :3] = [0.4, 0.0, 0.3]
    base[..., 3:7] = Rotation.identity().as_quat()
    base[..., 7] = 0.04
    edited = base.copy()
    labels = np.array([1, 0, 1, 0], dtype=np.uint8)
    masks = np.ones((sample_count, horizon), dtype=np.uint8)
    anchors = np.array([4, 0, 6, 0], dtype=np.int64)
    masks[0, :4] = 0
    masks[2, :6] = 0
    edited[0, 4:, 0] += 0.01
    edited[2, 6:, 1] -= 0.02
    if active_suffix:
        masks[0] = 0
        masks[0, :12] = 1
        masks[2] = 0
        masks[2, :10] = 1
        anchors[[0, 2]] = 0
        edited[0] = base[0]
        edited[2] = base[2]
        edited[0, :12, 0] += 0.01
        edited[2, :10, 1] -= 0.02
    residual = np.stack(
        [compute_world_frame_residual(b, e) for b, e in zip(base, edited)]
    )
    robot_state = np.zeros((sample_count, 2, 22), dtype=np.float64)
    robot_state[..., 14:17] = [0.4, 0.0, 0.3]
    robot_state[..., 17] = 1.0  # qw
    robot_state[..., 21] = 0.04
    episode_start = np.tile(np.eye(4), (sample_count, 1, 1))
    episode_start[:, :3, 3] = [0.4, 0.0, 0.3]

    store = zarr.ZipStore(str(path), mode="w")
    root = zarr.group(store=store)
    data = root.create_group("data")
    arrays = {
        "camera0_rgb": np.zeros((sample_count, 2, 32, 32, 3), dtype=np.uint8),
        "robot_state": robot_state,
        "episode_start_pose_matrix": episode_start,
        "frozen_base_action": base,
        "final_edited_action": edited,
        "residual_position": residual[..., :3],
        "residual_rotation_axis_angle": residual[..., 3:6],
        "residual_gripper": residual[..., 6:7],
        "valid_action_mask": masks,
        "anchor_index": anchors,
        "correction_label": labels,
        "source_episode_index": np.array([0, 0, 1, 1], dtype=np.int64),
    }
    for key, value in arrays.items():
        data.create_dataset(key, data=value, chunks=(1,) + value.shape[1:])
    root.attrs.update(
        {
            "schema": "superinference.residual_policy_events",
            "schema_version": 1,
            "sample_count": sample_count,
            "observation_horizon": 2,
            "action_horizon": horizon,
            "action_dim": 8,
            "action_alignment": (
                "active_suffix_left_aligned" if active_suffix else "recorded_chunk"
            ),
            "valid_action_mask_layout": (
                "contiguous_prefix" if active_suffix else "contiguous_suffix"
            ),
        }
    )
    store.close()


def test_residual_dataset_contract_and_episode_split(tmp_path):
    path = tmp_path / "residual.zarr.zip"
    _write_dataset(path)
    dataset = UmiResidualDataset(
        dataset_path=str(path),
        shape_meta=_shape_meta(),
        sample_mode="all",
        split="train",
        val_episode_indices=[1],
        correction_fraction=0.5,
    )
    assert len(dataset) == 2
    assert set(dataset.episode_ids.tolist()) == {0}
    sample = dataset[0]
    assert sample["obs"]["camera0_rgb"].shape == (2, 3, 32, 32)
    assert sample["base_action"].shape == (16, 10)
    assert sample["residual_action"].shape == (16, 7)
    assert sample["valid_action_mask"].shape == (16,)
    weights = dataset.get_sampling_weights()
    np.testing.assert_allclose(float(weights.sum()), 1.0)
    normalizer = dataset.get_residual_normalizer()
    zero = np.zeros((1, 7), dtype=np.float32)
    assert normalizer.normalize(zero).abs().max() == 0

    validation = dataset.get_validation_dataset()
    assert len(validation) == 2
    assert set(validation.episode_ids.tolist()) == {1}


def test_active_suffix_prefix_mask_contract(tmp_path):
    path = tmp_path / "active_residual.zarr.zip"
    _write_dataset(path, active_suffix=True)
    dataset = UmiResidualDataset(
        dataset_path=str(path),
        shape_meta=_shape_meta(),
        sample_mode="all",
        split="all",
        val_ratio=0.0,
        correction_fraction=0.5,
        expected_action_alignment="active_suffix_left_aligned",
    )
    assert dataset.action_alignment == "active_suffix_left_aligned"
    assert dataset.valid_action_mask_layout == "contiguous_prefix"
    assert dataset.audit_summary()["valid_correction_coverage"][:4] == [2, 2, 2, 2]


def test_multibase_sampling_balances_correction_families():
    dataset = object.__new__(UmiResidualDataset)
    dataset.indices = np.arange(4)
    dataset.labels = np.array([1, 1, 1, 0], dtype=np.uint8)
    dataset.correction_family_ids = np.array([5, 5, 9, 3], dtype=np.int64)
    dataset.sample_mode = "all"
    dataset.correction_fraction = 0.5
    dataset.balance_corrections_by_source_event = True
    dataset.balance_zeros_by_source_event = False
    weights = dataset.get_sampling_weights().numpy()
    np.testing.assert_allclose(weights, [0.125, 0.125, 0.25, 0.5])


def test_step_active_dataset_uses_explicit_horizon_and_family_balance(tmp_path):
    path = tmp_path / "step_active.zarr.zip"
    sample_count, horizon = 5, 5
    base = np.zeros((sample_count, horizon, 8), dtype=np.float32)
    base[..., :3] = [0.4, 0.0, 0.3]
    base[..., 3:7] = Rotation.identity().as_quat()
    base[..., 7] = 0.04
    edited = base.copy()
    labels = np.asarray([1, 1, 1, 0, 0], dtype=np.uint8)
    edited[:3, :, 0] += 0.01
    residual = np.stack(
        [compute_world_frame_residual(b, e) for b, e in zip(base, edited)]
    )
    robot_state = np.zeros((sample_count, 2, 22), dtype=np.float64)
    robot_state[..., 14:17] = [0.4, 0.0, 0.3]
    robot_state[..., 17] = 1.0
    robot_state[..., 21] = 0.04
    with zarr.ZipStore(str(path), mode="w") as store:
        root = zarr.group(store=store)
        data = root.create_group("data")
        arrays = {
            "camera0_rgb": np.zeros(
                (sample_count, 2, 32, 32, 3), dtype=np.uint8
            ),
            "robot_state": robot_state,
            "episode_start_pose_matrix": np.tile(
                np.eye(4), (sample_count, 1, 1)
            ),
            "frozen_base_action": base,
            "final_edited_action": edited,
            "residual_position": residual[..., :3],
            "residual_rotation_axis_angle": residual[..., 3:6],
            "residual_gripper": residual[..., 6:7],
            "valid_action_mask": np.ones(
                (sample_count, horizon), dtype=np.uint8
            ),
            "anchor_index": np.zeros(sample_count, dtype=np.int64),
            "correction_label": labels,
            "source_episode_index": np.asarray([0, 0, 1, 2, 2]),
            "source_event_index": np.asarray([10, 10, 20, -1, -1]),
        }
        for key, value in arrays.items():
            data.create_dataset(key, data=value, chunks=(1,) + value.shape[1:])
        root.attrs.update(
            {
                "schema": "superinference.step_active_plan_residual",
                "schema_version": 1,
                "sample_count": sample_count,
                "observation_horizon": 2,
                "action_horizon": horizon,
                "action_dim": 8,
                "action_alignment": "step_active_plan_left_aligned",
                "valid_action_mask_layout": "contiguous_prefix",
            }
        )

    dataset = UmiResidualDataset(
        dataset_path=str(path),
        shape_meta=_shape_meta(),
        split="all",
        val_ratio=0.0,
        expected_schema="superinference.step_active_plan_residual",
        expected_action_alignment="step_active_plan_left_aligned",
        action_horizon=5,
        correction_fraction=0.5,
        balance_zeros_by_source_event=True,
    )
    assert dataset[0]["base_action"].shape == (5, 10)
    np.testing.assert_allclose(
        dataset.get_sampling_weights().numpy(),
        [0.125, 0.125, 0.25, 0.25, 0.25],
    )
