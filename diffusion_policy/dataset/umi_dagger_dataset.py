import copy
import random
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from diffusion_policy.dataset.umi_dataset import UmiDataset
from diffusion_policy.dataset.base_dataset import BaseDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.common.normalize_util import (
    array_to_stats,
    concatenate_normalizer,
    get_identity_normalizer_from_stat,
    get_image_identity_normalizer,
    get_range_normalizer_from_stat,
)


class DaggerMixedUmiDataset(BaseDataset):
    """
    Offline DAgger mixture of two UMI zarr datasets (teleop + HITL/DAgger corrections).

    Sampling: per-sample Bernoulli with probability `hitl_prob` of drawing from HITL dataset.
    Validation: returns a dict with split-specific datasets (teleop / hitl / mixed).
    Normalizer: computed over the mixed training distribution.
    """

    def __init__(
        self,
        shape_meta: dict,
        teleop_dataset_path: str,
        hitl_dataset_path: str,
        dataset_path: Optional[str] = None,  # ignored; kept for Hydra compatibility
        hitl_prob: float = 0.5,
        normalizer_num_workers: Optional[int] = None,
        cache_dir: Optional[str] = None,
        pose_repr: dict = {},
        action_padding: bool = False,
        temporally_independent_normalization: bool = False,
        repeat_frame_prob: float = 0.0,
        seed: int = 42,
        val_ratio: float = 0.05,
        max_duration: Optional[float] = None,
        use_ratio: float = 1.0,
        dataset_idx: Optional[str] = None,
    ):
        super().__init__()
        assert 0.0 <= hitl_prob <= 1.0
        self.hitl_prob = hitl_prob
        self.rng = np.random.default_rng(seed)
        self.normalizer_num_workers = normalizer_num_workers

        common_kwargs = dict(
            shape_meta=shape_meta,
            cache_dir=cache_dir,
            pose_repr=pose_repr,
            action_padding=action_padding,
            temporally_independent_normalization=temporally_independent_normalization,
            repeat_frame_prob=repeat_frame_prob,
            seed=seed,
            val_ratio=val_ratio,
            max_duration=max_duration,
            use_ratio=use_ratio,
            dataset_idx=dataset_idx,
        )

        self.teleop_dataset = UmiDataset(dataset_path=teleop_dataset_path, **common_kwargs)
        self.hitl_dataset = UmiDataset(dataset_path=hitl_dataset_path, **common_kwargs)

        # expose shared attributes for convenience
        self.shape_meta = shape_meta
        self.rgb_keys = self.teleop_dataset.rgb_keys
        self.lowdim_keys = self.teleop_dataset.lowdim_keys
        self.key_horizon = self.teleop_dataset.key_horizon
        self.key_latency_steps = self.teleop_dataset.key_latency_steps
        self.key_down_sample_steps = self.teleop_dataset.key_down_sample_steps
        self.num_robot = self.teleop_dataset.num_robot
        self.temporally_independent_normalization = temporally_independent_normalization

    def __len__(self) -> int:
        # Use the larger dataset length as one epoch length.
        return max(len(self.teleop_dataset), len(self.hitl_dataset))

    def _sample_dataset(self):
        return self.hitl_dataset if self.rng.random() < self.hitl_prob else self.teleop_dataset

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        dataset = self._sample_dataset()
        # Re-index inside chosen dataset to avoid IndexError
        mapped_idx = idx % len(dataset)
        return dataset[mapped_idx]

    # ==================== validation datasets ====================
    def get_validation_dataset(self):
        teleop_val = self.teleop_dataset.get_validation_dataset()
        hitl_val = self.hitl_dataset.get_validation_dataset()
        mixed_val = _MixedValDataset(teleop_val, hitl_val, self.hitl_prob)
        return {
            "teleop": teleop_val,
            "hitl": hitl_val,
            "mixed": mixed_val,
        }

    # ==================== normalizer ====================
    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        """
        Compute normalizer over the mixed training distribution.
        """
        normalizer = LinearNormalizer()

        data_cache = {key: list() for key in self.lowdim_keys + ["action"]}
        # build a temporary dataloader to iterate once
        num_workers = kwargs.get("num_workers", self.normalizer_num_workers)
        if num_workers is None:
            num_workers = 0
        dataloader = DataLoader(self, batch_size=64, num_workers=num_workers)
        for batch in tqdm(dataloader, desc="iterating mixed dataset to get normalization"):
            for key in self.lowdim_keys:
                data_cache[key].append(copy.deepcopy(batch["obs"][key]))
            data_cache["action"].append(copy.deepcopy(batch["action"]))

        for key in data_cache.keys():
            data_cache[key] = np.concatenate(data_cache[key])
            assert len(data_cache[key].shape) == 3
            B, T, D = data_cache[key].shape
            if not self.temporally_independent_normalization:
                data_cache[key] = data_cache[key].reshape(B * T, D)

        # action
        assert data_cache["action"].shape[-1] % self.num_robot == 0
        dim_a = data_cache["action"].shape[-1] // self.num_robot
        action_normalizers = list()
        for i in range(self.num_robot):
            action_normalizers.append(
                get_range_normalizer_from_stat(array_to_stats(data_cache["action"][..., i * dim_a : i * dim_a + 3]))
            )  # pos
            action_normalizers.append(
                get_identity_normalizer_from_stat(
                    array_to_stats(data_cache["action"][..., i * dim_a + 3 : (i + 1) * dim_a - 1])
                )
            )  # rot
            action_normalizers.append(
                get_range_normalizer_from_stat(
                    array_to_stats(data_cache["action"][..., (i + 1) * dim_a - 1 : (i + 1) * dim_a])
                )
            )  # gripper

        normalizer["action"] = concatenate_normalizer(action_normalizers)

        # obs
        for key in self.lowdim_keys:
            stat = array_to_stats(data_cache[key])

            if key.endswith("pos") or "pos_wrt" in key:
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith("pos_abs"):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith("rot_axis_angle") or "rot_axis_angle_wrt" in key:
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith("gripper_width"):
                this_normalizer = get_range_normalizer_from_stat(stat)
            else:
                raise RuntimeError("unsupported")
            normalizer[key] = this_normalizer

        # image
        for key in self.rgb_keys:
            normalizer[key] = get_image_identity_normalizer()
        return normalizer


class _MixedValDataset(BaseDataset):
    """Validation-time mixture dataset with the same Bernoulli sampling."""

    def __init__(self, teleop_val: BaseDataset, hitl_val: BaseDataset, hitl_prob: float):
        self.teleop_val = teleop_val
        self.hitl_val = hitl_val
        self.hitl_prob = hitl_prob
        self.rng = np.random.default_rng(0)

    def __len__(self):
        return max(len(self.teleop_val), len(self.hitl_val))

    def __getitem__(self, idx: int):
        dataset = self.hitl_val if self.rng.random() < self.hitl_prob else self.teleop_val
        mapped_idx = idx % len(dataset)
        return dataset[mapped_idx]
