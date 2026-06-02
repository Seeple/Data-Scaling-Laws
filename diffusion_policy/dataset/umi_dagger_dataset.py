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

try:
    from imagecodecs._jpegxl import JpegxlError
except Exception:  # pragma: no cover - fallback if imagecodecs internals change
    class JpegxlError(Exception):
        pass


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
        hitl_disable_downsample: bool = False,
        hitl_downsample_multiplier: float = 3.0,
        hitl_only_tag: bool = False,
        hitl_tag_key: str = "hitl_tag",
        action_normalizer_source: str = "teleop",
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
        self.hitl_only_tag = hitl_only_tag
        self.hitl_tag_key = hitl_tag_key
        assert action_normalizer_source in {"teleop", "mixed"}
        self.action_normalizer_source = action_normalizer_source

        def _adjust_downsample(meta: dict, multiplier: float) -> dict:
            meta = copy.deepcopy(meta)
            for key, attr in meta.get("obs", {}).items():
                if "down_sample_steps" in attr:
                    step = float(attr["down_sample_steps"])
                    attr["down_sample_steps"] = max(1, int(round(step * multiplier)))
            if "action" in meta and "down_sample_steps" in meta["action"]:
                step = float(meta["action"]["down_sample_steps"])
                meta["action"]["down_sample_steps"] = max(1, int(round(step * multiplier)))
            return meta

        hitl_shape_meta = (
            _adjust_downsample(shape_meta, hitl_downsample_multiplier)
            if hitl_disable_downsample
            else shape_meta
        )

        common_kwargs = dict(
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

        self.teleop_dataset = UmiDataset(shape_meta=shape_meta, dataset_path=teleop_dataset_path, **common_kwargs)
        self.hitl_dataset = UmiDataset(shape_meta=hitl_shape_meta, dataset_path=hitl_dataset_path, **common_kwargs)
        if self.hitl_only_tag:
            self._apply_hitl_tag_filter(self.hitl_dataset)

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
        max_retries = 5
        attempt = 0
        while True:
            dataset = self._sample_dataset()
            # Re-index inside chosen dataset to avoid IndexError
            mapped_idx = idx % len(dataset)
            try:
                return dataset[mapped_idx]
            except JpegxlError as exc:
                attempt += 1
                if attempt > max_retries:
                    raise
                idx = int(self.rng.integers(0, len(self)))
                print(
                    f"[DaggerMixedUmiDataset] JpegXL decode failed (attempt {attempt}/{max_retries}). "
                    f"Resampling idx={idx}. Error: {exc}"
                )

    # ==================== validation datasets ====================
    def get_validation_dataset(self):
        teleop_val = self.teleop_dataset.get_validation_dataset()
        hitl_val = self.hitl_dataset.get_validation_dataset()
        if self.hitl_only_tag:
            self._apply_hitl_tag_filter(hitl_val)
        mixed_val = _MixedValDataset(teleop_val, hitl_val, self.hitl_prob)
        return {
            "teleop": teleop_val,
            "hitl": hitl_val,
            "mixed": mixed_val,
        }

    def _apply_hitl_tag_filter(self, dataset: UmiDataset) -> None:
        if self.hitl_tag_key not in dataset.replay_buffer:
            raise KeyError(
                f"hitl_only_tag enabled but '{self.hitl_tag_key}' not found in HITL dataset"
            )
        hitl_tag = np.asarray(dataset.replay_buffer[self.hitl_tag_key]).reshape(-1)
        filtered = []
        for entry in dataset.sampler.indices:
            current_idx = entry[0]
            if hitl_tag[current_idx] == 1:
                filtered.append(entry)
        dataset.sampler.indices = filtered
        if len(dataset.sampler.indices) == 0:
            raise ValueError(
                "hitl_only_tag resulted in an empty HITL dataset; check hitl_tag values"
            )

    # ==================== normalizer ====================
    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        """
        Compute normalizer for DAgger training.
        - obs/image normalizers: mixed training distribution
        - action normalizer: configurable source
        """
        normalizer = LinearNormalizer()
        print(
            "[DaggerMixedUmiDataset] Normalizer config: "
            f"action_normalizer_source={self.action_normalizer_source} "
            "(default=teleop/offline)"
        )

        data_cache = {key: list() for key in self.lowdim_keys}
        # build a temporary dataloader to iterate once
        num_workers = kwargs.get("num_workers", self.normalizer_num_workers)
        if num_workers is None:
            num_workers = 8
        dataloader = DataLoader(self, batch_size=64, num_workers=num_workers)
        for batch in tqdm(
            dataloader,
            desc="iterating mixed dataset to get OBS normalization",
        ):
            for key in self.lowdim_keys:
                data_cache[key].append(copy.deepcopy(batch["obs"][key]))

        for key in data_cache.keys():
            data_cache[key] = np.concatenate(data_cache[key])
            assert len(data_cache[key].shape) == 3
            B, T, D = data_cache[key].shape
            if not self.temporally_independent_normalization:
                data_cache[key] = data_cache[key].reshape(B * T, D)

        # action
        if self.action_normalizer_source == "teleop":
            # Preferred for DAgger finetuning stability:
            # keep action scale anchored to offline expert distribution.
            print(
                "[DaggerMixedUmiDataset] Building ACTION normalizer from "
                "teleop/offline buffer."
            )
            teleop_normalizer = self.teleop_dataset.get_normalizer()
            normalizer["action"] = teleop_normalizer["action"]
        else:
            # Kept for ablation/backward compatibility:
            # compute action normalizer from mixed (teleop + HITL) samples.
            mixed_action_cache = list()
            action_dataloader = DataLoader(self, batch_size=64, num_workers=num_workers)
            print(
                "[DaggerMixedUmiDataset] Building ACTION normalizer from mixed "
                "(teleop + HITL) buffer."
            )
            for batch in tqdm(
                action_dataloader,
                desc="iterating mixed dataset to get ACTION normalization",
            ):
                mixed_action_cache.append(copy.deepcopy(batch["action"]))
            mixed_action_cache = np.concatenate(mixed_action_cache)
            assert len(mixed_action_cache.shape) == 3
            B, T, D = mixed_action_cache.shape
            if not self.temporally_independent_normalization:
                mixed_action_cache = mixed_action_cache.reshape(B * T, D)

            assert mixed_action_cache.shape[-1] % self.num_robot == 0
            dim_a = mixed_action_cache.shape[-1] // self.num_robot
            action_normalizers = list()
            for i in range(self.num_robot):
                action_normalizers.append(
                    get_range_normalizer_from_stat(
                        array_to_stats(mixed_action_cache[..., i * dim_a : i * dim_a + 3])
                    )
                )  # pos
                action_normalizers.append(
                    get_identity_normalizer_from_stat(
                        array_to_stats(mixed_action_cache[..., i * dim_a + 3 : (i + 1) * dim_a - 1])
                    )
                )  # rot
                action_normalizers.append(
                    get_range_normalizer_from_stat(
                        array_to_stats(mixed_action_cache[..., (i + 1) * dim_a - 1 : (i + 1) * dim_a])
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
