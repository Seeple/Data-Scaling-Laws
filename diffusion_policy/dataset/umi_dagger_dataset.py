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
    Offline DAgger mixture of one teleop buffer and one or more HITL buffers.

    Sampling: per-sample categorical draw from `rlpd_ratio` if provided. Otherwise,
    fall back to the historical two-buffer Bernoulli `hitl_prob` behavior.
    Validation: returns split-specific datasets plus a weighted mixed validation set.
    Normalizer: action stats can stay fixed to teleop/offline, while low-dim obs
    stats can come from the weighted mixed training distribution.
    """

    def __init__(
        self,
        shape_meta: dict,
        teleop_dataset_path: str,
        hitl_dataset_path: str,
        dataset_path: Optional[str] = None,  # ignored; kept for Hydra compatibility
        online_dataset_paths: Optional[object] = None,
        rlpd_ratio: Optional[object] = None,
        hitl_prob: float = 0.5,
        hitl_disable_downsample: bool = False,
        hitl_downsample_multiplier: float = 3.0,
        hitl_separate_downsample_multipliers: bool = False,
        hitl_obs_downsample_multiplier: float = 1.0,
        hitl_action_downsample_multiplier: float = 1.0,
        hitl_only_tag: bool = False,
        hitl_tag_key: str = "hitl_tag",
        hitl_require_full_action_tag: bool = False,
        hitl_action_mask: bool = False,
        hitl_contiguous_action_mask: bool = False,
        hitl_invalid_tail_padding: bool = False,
        hitl_min_valid_steps_enabled: bool = False,
        hitl_min_valid_steps: int = 5,
        hitl_skip_rising_edge: bool = False,
        hitl_skip_rising_edge_steps: int = 5,
        hitl_treat_segments_as_episodes: bool = False,
        action_normalizer_source: str = "teleop",
        lowdim_obs_normalizer_source: str = "mixed",
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
        self.hitl_require_full_action_tag = hitl_require_full_action_tag
        self.hitl_action_mask = hitl_action_mask
        self.hitl_contiguous_action_mask = bool(hitl_contiguous_action_mask)
        self.hitl_invalid_tail_padding = bool(hitl_invalid_tail_padding)
        self.hitl_min_valid_steps_enabled = bool(hitl_min_valid_steps_enabled)
        self.hitl_min_valid_steps = int(hitl_min_valid_steps)
        self.hitl_skip_rising_edge = hitl_skip_rising_edge
        self.hitl_skip_rising_edge_steps = int(hitl_skip_rising_edge_steps)
        self.hitl_treat_segments_as_episodes = hitl_treat_segments_as_episodes
        assert action_normalizer_source in {"teleop", "mixed"}
        self.action_normalizer_source = action_normalizer_source
        assert lowdim_obs_normalizer_source in {"mixed", "teleop"}
        self.lowdim_obs_normalizer_source = lowdim_obs_normalizer_source
        self.online_dataset_paths = self._coerce_list(online_dataset_paths)
        if len(self.online_dataset_paths) == 0:
            self.online_dataset_paths = [hitl_dataset_path]

        if self.hitl_contiguous_action_mask and not self.hitl_action_mask:
            raise ValueError(
                "hitl_contiguous_action_mask requires hitl_action_mask=true"
            )
        if self.hitl_invalid_tail_padding and not self.hitl_contiguous_action_mask:
            raise ValueError(
                "hitl_invalid_tail_padding requires "
                "hitl_contiguous_action_mask=true"
            )
        if self.hitl_min_valid_steps_enabled and self.hitl_min_valid_steps < 1:
            raise ValueError("hitl_min_valid_steps must be at least 1")

        if hitl_separate_downsample_multipliers:
            obs_multiplier = float(hitl_obs_downsample_multiplier)
            action_multiplier = float(hitl_action_downsample_multiplier)
        else:
            # Preserve the historical shared-multiplier behavior unless the new
            # feature switch is explicitly enabled.
            obs_multiplier = float(hitl_downsample_multiplier)
            action_multiplier = float(hitl_downsample_multiplier)

        hitl_shape_meta = (
            self._adjust_downsample(
                shape_meta,
                obs_multiplier=obs_multiplier,
                action_multiplier=action_multiplier,
            )
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
        self.online_datasets = [
            UmiDataset(shape_meta=hitl_shape_meta, dataset_path=path, **common_kwargs)
            for path in self.online_dataset_paths
        ]
        # Backward-compatible alias for code/tests that still expect one HITL dataset.
        self.hitl_dataset = self.online_datasets[0]
        if self._needs_hitl_tag_filter():
            for online_idx, dataset in enumerate(self.online_datasets):
                self._apply_hitl_tag_filter(dataset, dataset_name=f"online_{online_idx}")

        self.datasets = [self.teleop_dataset] + self.online_datasets
        self.dataset_names = ["teleop"] + [
            "hitl" if i == 0 else f"hitl_{i}"
            for i in range(len(self.online_datasets))
        ]
        self.dataset_is_online = [False] + [True] * len(self.online_datasets)
        self.dataset_weights = self._resolve_sampling_weights(rlpd_ratio)
        print(
            "[DaggerMixedUmiDataset] Sampling buffers: "
            + ", ".join(
                f"{name}=weight:{weight:.6f},len:{len(dataset)}"
                for name, weight, dataset in zip(
                    self.dataset_names,
                    self.dataset_weights,
                    self.datasets,
                )
            )
        )
        print(
            "[DaggerMixedUmiDataset] HITL ablations: "
            f"contiguous_action_mask={self.hitl_contiguous_action_mask}, "
            f"invalid_tail_padding={self.hitl_invalid_tail_padding}, "
            f"min_valid_steps_enabled={self.hitl_min_valid_steps_enabled}, "
            f"min_valid_steps={self.hitl_min_valid_steps}, "
            f"separate_downsample_multipliers="
            f"{hitl_separate_downsample_multipliers}, "
            f"obs_multiplier={obs_multiplier}, "
            f"action_multiplier={action_multiplier}"
        )

        # expose shared attributes for convenience
        self.shape_meta = shape_meta
        self.rgb_keys = self.teleop_dataset.rgb_keys
        self.lowdim_keys = self.teleop_dataset.lowdim_keys
        self.key_horizon = self.teleop_dataset.key_horizon
        self.key_latency_steps = self.teleop_dataset.key_latency_steps
        self.key_down_sample_steps = self.teleop_dataset.key_down_sample_steps
        self.num_robot = self.teleop_dataset.num_robot
        self.temporally_independent_normalization = temporally_independent_normalization

    @staticmethod
    def _coerce_list(value) -> list:
        if value is None:
            return []
        if isinstance(value, str):
            value = value.strip()
            if value == "" or value.lower() in {"none", "null"}:
                return []
            if value.startswith("[") and value.endswith("]"):
                value = value[1:-1]
            return [
                item.strip().strip("'\"")
                for item in value.split(",")
                if item.strip()
            ]
        return [
            item
            for item in list(value)
            if item is not None and str(item).strip().lower() not in {"", "none", "null"}
        ]

    @staticmethod
    def _coerce_float_list(value) -> Optional[list]:
        if value is None:
            return None
        if isinstance(value, str):
            value = value.strip()
            if value == "" or value.lower() in {"none", "null"}:
                return None
            if value.startswith("[") and value.endswith("]"):
                value = value[1:-1]
            return [
                float(item.strip())
                for item in value.split(",")
                if item.strip()
            ]
        return [float(item) for item in list(value)]

    @staticmethod
    def _adjust_downsample(
        shape_meta: dict,
        obs_multiplier: float,
        action_multiplier: float,
    ) -> dict:
        """Scale HITL observation and action strides without mutating shape_meta."""
        if obs_multiplier <= 0 or action_multiplier <= 0:
            raise ValueError(
                "HITL downsample multipliers must be positive, got "
                f"obs={obs_multiplier}, action={action_multiplier}"
            )

        meta = copy.deepcopy(shape_meta)
        for attr in meta.get("obs", {}).values():
            if "down_sample_steps" in attr:
                step = float(attr["down_sample_steps"])
                attr["down_sample_steps"] = max(
                    1, int(round(step * obs_multiplier))
                )
        if "action" in meta and "down_sample_steps" in meta["action"]:
            step = float(meta["action"]["down_sample_steps"])
            meta["action"]["down_sample_steps"] = max(
                1, int(round(step * action_multiplier))
            )
        return meta

    def _resolve_sampling_weights(self, rlpd_ratio) -> np.ndarray:
        weights = self._coerce_float_list(rlpd_ratio)
        if weights is None:
            # Historical two-buffer behavior: offline with 1-hitl_prob, first
            # online/HITL buffer with hitl_prob. Additional online buffers must
            # opt into explicit rlpd_ratio to avoid surprising defaults.
            if len(self.datasets) != 2:
                raise ValueError(
                    "Multiple online buffers require explicit rlpd_ratio with "
                    "one weight for teleop/offline plus one per online buffer."
                )
            weights = [1.0 - self.hitl_prob, self.hitl_prob]

        if len(weights) != len(self.datasets):
            raise ValueError(
                "rlpd_ratio length mismatch: expected "
                f"{len(self.datasets)} weights for {self.dataset_names}, got {len(weights)}"
            )
        weights = np.asarray(weights, dtype=np.float64)
        if np.any(weights < 0):
            raise ValueError(f"rlpd_ratio must be non-negative, got {weights.tolist()}")
        total = float(weights.sum())
        if total <= 0:
            raise ValueError(f"rlpd_ratio must have positive sum, got {weights.tolist()}")
        return weights / total

    def __len__(self) -> int:
        # Use the largest buffer length as one epoch length; smaller buffers are
        # re-indexed cyclically after categorical sampling, matching old behavior.
        return max(len(dataset) for dataset in self.datasets)

    def _sample_dataset(self):
        dataset_idx = int(self.rng.choice(len(self.datasets), p=self.dataset_weights))
        return dataset_idx, self.datasets[dataset_idx]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        max_retries = 5
        attempt = 0
        while True:
            dataset_idx, dataset = self._sample_dataset()
            # Re-index inside chosen dataset to avoid IndexError
            mapped_idx = idx % len(dataset)
            try:
                sample = dataset[mapped_idx]
                if self.hitl_action_mask:
                    action_valid_mask = self._get_action_valid_mask(
                        dataset=dataset,
                        mapped_idx=mapped_idx,
                        action_length=sample["action"].shape[0],
                        is_online=self.dataset_is_online[dataset_idx],
                    )
                    if (
                        self.hitl_invalid_tail_padding
                        and self.dataset_is_online[dataset_idx]
                    ):
                        sample["action"] = self._pad_invalid_action_tail(
                            sample["action"], action_valid_mask
                        )
                    sample["action_valid_mask"] = action_valid_mask
                return sample
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
        online_vals = [
            dataset.get_validation_dataset()
            for dataset in self.online_datasets
        ]
        if self._needs_hitl_tag_filter():
            for online_idx, val_dataset in enumerate(online_vals):
                self._apply_hitl_tag_filter(
                    val_dataset,
                    dataset_name=f"val_online_{online_idx}",
                )
        mixed_val = _MixedValDataset(
            datasets=[teleop_val] + online_vals,
            weights=self.dataset_weights,
        )
        result = {
            "teleop": teleop_val,
            "mixed": mixed_val,
        }
        if len(online_vals) > 0:
            result["hitl"] = online_vals[0]
        for online_idx, val_dataset in enumerate(online_vals[1:], start=1):
            result[f"hitl_{online_idx}"] = val_dataset
        return result

    def _needs_hitl_tag_filter(self) -> bool:
        return (
            self.hitl_only_tag
            or self.hitl_require_full_action_tag
            or self.hitl_min_valid_steps_enabled
            or self.hitl_skip_rising_edge
            or self.hitl_treat_segments_as_episodes
        )

    def _get_hitl_tag(self, dataset: UmiDataset) -> np.ndarray:
        if self.hitl_tag_key not in dataset.replay_buffer:
            raise KeyError(
                f"HITL tag logic enabled but '{self.hitl_tag_key}' not found in HITL dataset"
            )
        return np.asarray(dataset.replay_buffer[self.hitl_tag_key]).reshape(-1)

    @staticmethod
    def _build_hitl_segments(hitl_tag: np.ndarray, episode_ends: np.ndarray):
        segments_by_episode = []
        for ep_idx, ep_end in enumerate(episode_ends):
            start_idx = 0 if ep_idx == 0 else int(episode_ends[ep_idx - 1])
            end_idx = int(ep_end)
            segments = []
            in_segment = False
            seg_start = start_idx
            for idx in range(start_idx, end_idx):
                active = hitl_tag[idx] == 1
                if active and not in_segment:
                    in_segment = True
                    seg_start = idx
                elif not active and in_segment:
                    segments.append((seg_start, idx))
                    in_segment = False
            if in_segment:
                segments.append((seg_start, end_idx))
            segments_by_episode.append(segments)
        return segments_by_episode

    @staticmethod
    def _find_segment(current_idx: int, segments):
        for seg_start, seg_end in segments:
            if seg_start <= current_idx < seg_end:
                return seg_start, seg_end
        return None

    @staticmethod
    def _future_action_indices(current_idx: int, horizon: int, stride: int) -> np.ndarray:
        return current_idx + np.arange(horizon, dtype=np.int64) * int(stride)

    @staticmethod
    def _make_contiguous_prefix(mask: np.ndarray) -> np.ndarray:
        """Keep valid entries only until the first invalid action position."""
        mask = np.asarray(mask, dtype=np.float32)
        if mask.ndim != 1:
            raise ValueError(f"Expected a 1-D action mask, got shape {mask.shape}")
        return np.cumprod(mask, dtype=np.float32)

    @staticmethod
    def _count_contiguous_valid_steps(mask: np.ndarray) -> int:
        """Count leading valid positions in an action mask."""
        mask = np.asarray(mask).reshape(-1)
        invalid = np.flatnonzero(mask <= 0)
        return int(invalid[0]) if len(invalid) > 0 else int(len(mask))

    @staticmethod
    def _pad_invalid_action_tail(
        action: torch.Tensor,
        action_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Repeat the last valid human action over a contiguous invalid suffix."""
        mask = action_valid_mask.reshape(-1)
        if action.shape[0] != mask.shape[0]:
            raise ValueError(
                "Action/mask length mismatch: "
                f"action={action.shape[0]}, mask={mask.shape[0]}"
            )

        invalid = torch.nonzero(mask <= 0, as_tuple=False).flatten()
        if invalid.numel() == 0:
            return action

        first_invalid = int(invalid[0].item())
        if first_invalid == 0:
            # No human action exists to use as padding. This is possible when
            # online policy frames are sampled without hitl_only_tag.
            return action

        padded = action.clone()
        padded[first_invalid:] = padded[first_invalid - 1]
        return padded

    def _apply_hitl_tag_filter(self, dataset: UmiDataset, dataset_name: str = "hitl") -> None:
        hitl_tag = self._get_hitl_tag(dataset)
        episode_ends = np.asarray(dataset.replay_buffer.episode_ends[:], dtype=np.int64)
        segments_by_episode = self._build_hitl_segments(hitl_tag, episode_ends)
        action_horizon = dataset.key_horizon["action"]
        action_stride = int(dataset.key_down_sample_steps["action"])

        filtered = []
        for entry in dataset.sampler.indices:
            current_idx, start_idx, end_idx, before_first_grasp = entry
            current_idx = int(current_idx)
            if hitl_tag[current_idx] != 1:
                continue

            ep_idx = int(np.searchsorted(episode_ends, current_idx, side="right"))
            segment = self._find_segment(current_idx, segments_by_episode[ep_idx])
            if segment is None:
                continue
            seg_start, seg_end = segment

            if self.hitl_skip_rising_edge:
                if current_idx < seg_start + self.hitl_skip_rising_edge_steps:
                    continue

            future_idx = self._future_action_indices(
                current_idx=current_idx,
                horizon=action_horizon,
                stride=action_stride,
            )
            if future_idx[-1] >= int(end_idx):
                continue

            if self.hitl_require_full_action_tag and not np.all(hitl_tag[future_idx] == 1):
                continue

            if self.hitl_min_valid_steps_enabled:
                raw_mask = (hitl_tag[future_idx] == 1).astype(np.float32)
                contiguous_steps = self._count_contiguous_valid_steps(raw_mask)
                if contiguous_steps < self.hitl_min_valid_steps:
                    continue

            if self.hitl_treat_segments_as_episodes:
                # Make the sampler pad obs history at the segment start and avoid
                # supervising labels outside the continuous human-control interval.
                if future_idx[-1] >= seg_end:
                    continue
                filtered.append((current_idx, seg_start, seg_end, before_first_grasp))
            else:
                filtered.append(entry)

        dataset.sampler.indices = filtered
        if len(dataset.sampler.indices) == 0:
            raise ValueError(
                "HITL tag filtering resulted in an empty HITL dataset; check hitl_tag values/settings"
            )
        print(
            f"[DaggerMixedUmiDataset:{dataset_name}] HITL sampler after tag filters: "
            f"{len(dataset.sampler.indices)} samples "
            f"(hitl_only_tag={self.hitl_only_tag}, "
            f"hitl_require_full_action_tag={self.hitl_require_full_action_tag}, "
            f"hitl_min_valid_steps_enabled={self.hitl_min_valid_steps_enabled}, "
            f"hitl_min_valid_steps={self.hitl_min_valid_steps}, "
            f"hitl_skip_rising_edge={self.hitl_skip_rising_edge}, "
            f"hitl_skip_rising_edge_steps={self.hitl_skip_rising_edge_steps}, "
            f"hitl_treat_segments_as_episodes={self.hitl_treat_segments_as_episodes})"
        )

    def _get_action_valid_mask(
        self,
        dataset: UmiDataset,
        mapped_idx: int,
        action_length: int,
        is_online: bool,
    ) -> torch.Tensor:
        if not is_online:
            return torch.ones(action_length, dtype=torch.float32)

        hitl_tag = self._get_hitl_tag(dataset)
        current_idx = int(dataset.sampler.indices[mapped_idx][0])
        action_stride = int(dataset.key_down_sample_steps["action"])
        future_idx = self._future_action_indices(
            current_idx=current_idx,
            horizon=action_length,
            stride=action_stride,
        )
        mask = np.zeros(action_length, dtype=np.float32)
        valid = future_idx < len(hitl_tag)
        mask[valid] = (hitl_tag[future_idx[valid]] == 1).astype(np.float32)
        if self.hitl_contiguous_action_mask:
            mask = self._make_contiguous_prefix(mask)
        return torch.from_numpy(mask)

    # ==================== normalizer ====================
    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        """
        Compute normalizer for DAgger training.
        - low-dim obs normalizers: mixed or teleop/offline distribution
        - action normalizer: configurable source
        """
        normalizer = LinearNormalizer()
        print(
            "[DaggerMixedUmiDataset] Normalizer config: "
            f"action_normalizer_source={self.action_normalizer_source} "
            "(default=teleop/offline), "
            f"lowdim_obs_normalizer_source={self.lowdim_obs_normalizer_source} "
            "(default=mixed)"
        )

        num_workers = kwargs.get("num_workers", self.normalizer_num_workers)
        if num_workers is None:
            num_workers = 8
        teleop_normalizer = None

        def get_teleop_normalizer():
            nonlocal teleop_normalizer
            if teleop_normalizer is None:
                teleop_normalizer = self.teleop_dataset.get_normalizer()
            return teleop_normalizer

        # action
        if self.action_normalizer_source == "teleop":
            # Preferred for DAgger finetuning stability:
            # keep action scale anchored to offline expert distribution.
            print(
                "[DaggerMixedUmiDataset] Building ACTION normalizer from "
                "teleop/offline buffer."
            )
            normalizer["action"] = get_teleop_normalizer()["action"]
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
        if self.lowdim_obs_normalizer_source == "teleop":
            print(
                "[DaggerMixedUmiDataset] Building LOW-DIM OBS normalizer from "
                "teleop/offline buffer."
            )
            teleop_norm = get_teleop_normalizer()
            for key in self.lowdim_keys:
                normalizer[key] = teleop_norm[key]
        else:
            data_cache = {key: list() for key in self.lowdim_keys}
            # build a temporary dataloader to iterate once
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
    """Validation-time weighted mixture dataset."""

    def __init__(self, datasets, weights):
        self.datasets = list(datasets)
        self.weights = np.asarray(weights, dtype=np.float64)
        self.weights = self.weights / self.weights.sum()
        self.rng = np.random.default_rng(0)

    def __len__(self):
        return max(len(dataset) for dataset in self.datasets)

    def __getitem__(self, idx: int):
        dataset_idx = int(self.rng.choice(len(self.datasets), p=self.weights))
        dataset = self.datasets[dataset_idx]
        mapped_idx = idx % len(dataset)
        return dataset[mapped_idx]
