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


def _write_dataset(path: Path):
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
