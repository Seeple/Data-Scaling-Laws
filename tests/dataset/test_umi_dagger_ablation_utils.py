"""Unit tests for opt-in HITL DAgger ablation helpers."""

import copy
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from diffusion_policy.dataset.umi_dagger_dataset import DaggerMixedUmiDataset


class DaggerAblationUtilsTest(unittest.TestCase):
    def test_contiguous_prefix_never_reenables_mask(self) -> None:
        raw = np.array([1, 1, 0, 0, 1, 1, 0], dtype=np.float32)

        actual = DaggerMixedUmiDataset._make_contiguous_prefix(raw)

        np.testing.assert_array_equal(
            actual,
            np.array([1, 1, 0, 0, 0, 0, 0], dtype=np.float32),
        )

    def test_count_contiguous_valid_steps(self) -> None:
        cases = [
            ([1, 1, 1, 0, 1], 3),
            ([1, 1, 1, 1], 4),
            ([0, 1, 1, 1], 0),
        ]
        for mask, expected in cases:
            with self.subTest(mask=mask):
                self.assertEqual(
                    DaggerMixedUmiDataset._count_contiguous_valid_steps(mask),
                    expected,
                )

    def test_invalid_tail_padding_repeats_last_valid_action(self) -> None:
        action = torch.arange(12, dtype=torch.float32).reshape(4, 3)
        mask = torch.tensor([1, 1, 0, 0], dtype=torch.float32)

        actual = DaggerMixedUmiDataset._pad_invalid_action_tail(action, mask)

        torch.testing.assert_close(actual[:2], action[:2])
        torch.testing.assert_close(actual[2], action[1])
        torch.testing.assert_close(actual[3], action[1])
        # Padding must not mutate the UmiDataset sample in place.
        torch.testing.assert_close(
            action,
            torch.arange(12, dtype=torch.float32).reshape(4, 3),
        )

    def test_invalid_tail_padding_without_valid_prefix_is_noop(self) -> None:
        action = torch.arange(8, dtype=torch.float32).reshape(4, 2)
        mask = torch.zeros(4, dtype=torch.float32)

        actual = DaggerMixedUmiDataset._pad_invalid_action_tail(action, mask)

        self.assertIs(actual, action)

    def test_separate_downsample_does_not_mutate_shape_meta(self) -> None:
        shape_meta = {
            "obs": {
                "camera0_rgb": {"down_sample_steps": 3},
                "robot0_eef_pos": {"down_sample_steps": 3},
            },
            "action": {"down_sample_steps": 3},
        }
        original = copy.deepcopy(shape_meta)

        actual = DaggerMixedUmiDataset._adjust_downsample(
            shape_meta,
            obs_multiplier=1,
            action_multiplier=2,
        )

        self.assertEqual(
            actual["obs"]["camera0_rgb"]["down_sample_steps"], 3
        )
        self.assertEqual(
            actual["obs"]["robot0_eef_pos"]["down_sample_steps"], 3
        )
        self.assertEqual(actual["action"]["down_sample_steps"], 6)
        self.assertEqual(shape_meta, original)

    def test_downsample_multipliers_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be positive"):
            DaggerMixedUmiDataset._adjust_downsample(
                {"obs": {}, "action": {"down_sample_steps": 3}},
                obs_multiplier=0,
                action_multiplier=2,
            )

    def test_min_valid_steps_filters_short_human_prefixes(self) -> None:
        class FakeReplayBuffer(dict):
            episode_ends = np.array([8], dtype=np.int64)

        dataset = SimpleNamespace(
            replay_buffer=FakeReplayBuffer(
                hitl_tag=np.array(
                    [1, 1, 1, 1, 0, 0, 0, 0], dtype=np.int64
                )
            ),
            key_horizon={"action": 4},
            key_down_sample_steps={"action": 1},
            sampler=SimpleNamespace(
                indices=[(idx, 0, 8, True) for idx in range(5)]
            ),
        )
        dagger = DaggerMixedUmiDataset.__new__(DaggerMixedUmiDataset)
        dagger.hitl_tag_key = "hitl_tag"
        dagger.hitl_only_tag = True
        dagger.hitl_require_full_action_tag = False
        dagger.hitl_min_valid_steps_enabled = True
        dagger.hitl_min_valid_steps = 3
        dagger.hitl_skip_rising_edge = False
        dagger.hitl_skip_rising_edge_steps = 0
        dagger.hitl_treat_segments_as_episodes = False

        dagger._apply_hitl_tag_filter(dataset, dataset_name="unit_test")

        self.assertEqual(
            [entry[0] for entry in dataset.sampler.indices], [0, 1]
        )


if __name__ == "__main__":
    unittest.main()
