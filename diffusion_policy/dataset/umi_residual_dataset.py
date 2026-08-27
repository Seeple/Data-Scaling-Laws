"""Event-level dataset for chunk residual learning on Rizon/UMI policies."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import torch
import zarr

from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
from diffusion_policy.common.pose_repr_util import convert_pose_mat_rep
from diffusion_policy.common.residual_action_util import (
    ABSOLUTE_ACTION_DIM,
    BASE_ACTION_DIM,
    RESIDUAL_ACTION_DIM,
    absolute_action8_to_relative_action10,
    compute_world_frame_residual,
    fit_zero_centered_scale,
    flexiv_state_to_gripper,
    flexiv_state_to_pose_matrix,
    residual_recomposition_errors,
)
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import (
    SingleFieldLinearNormalizer,
)
from umi.common.pose_util import mat_to_pose10d


register_codecs()

EXPECTED_SCHEMA = "superinference.residual_policy_events"
EXPECTED_SCHEMA_VERSION = 1
STEP_ACTIVE_PLAN_SCHEMA = "superinference.step_active_plan_residual"
ACTION_ALIGNMENT_RECORDED_CHUNK = "recorded_chunk"
ACTION_ALIGNMENT_ACTIVE_SUFFIX = "active_suffix_left_aligned"
ACTION_ALIGNMENT_STEP_ACTIVE_PLAN = "step_active_plan_left_aligned"
VALID_MASK_LAYOUT_SUFFIX = "contiguous_suffix"
VALID_MASK_LAYOUT_PREFIX = "contiguous_prefix"


class UmiResidualDataset(BaseImageDataset):
    """One sample per frozen-base/final-edited action-chunk event.

    This is deliberately not a fixed-rate sequence dataset.  Each row already
    contains the exact observation history that generated one frozen base
    action chunk.
    """

    def __init__(
        self,
        dataset_path: str,
        shape_meta: dict,
        sample_mode: str = "all",
        split: str = "train",
        seed: int = 42,
        val_ratio: float = 0.15,
        val_episode_indices: Optional[Sequence[int]] = None,
        expected_schema: str = EXPECTED_SCHEMA,
        expected_schema_version: int = EXPECTED_SCHEMA_VERSION,
        expected_action_alignment: Optional[str] = None,
        action_horizon: Optional[int] = None,
        residual_scale_quantile: float = 0.995,
        residual_minimum_scale: float = 1e-4,
        correction_fraction: Optional[float] = 0.5,
        balance_corrections_by_source_event: bool = True,
        balance_zeros_by_source_event: bool = False,
        strict_recomposition_tolerance: float = 1e-5,
        return_metadata: bool = False,
        **_unused_umi_dataset_kwargs,
    ) -> None:
        super().__init__()
        self.dataset_path = str(Path(dataset_path).expanduser().resolve())
        self.shape_meta = copy.deepcopy(shape_meta)
        self.sample_mode = str(sample_mode)
        self.split = str(split)
        self.seed = int(seed)
        self.val_ratio = float(val_ratio)
        self.val_episode_indices = (
            None
            if val_episode_indices is None
            else tuple(int(value) for value in val_episode_indices)
        )
        self.expected_schema_version = int(expected_schema_version)
        self.expected_schema = str(expected_schema)
        self.expected_action_alignment = (
            None
            if expected_action_alignment in {None, "", "auto"}
            else str(expected_action_alignment)
        )
        self.expected_action_horizon = (
            None if action_horizon is None else int(action_horizon)
        )
        self.residual_scale_quantile = float(residual_scale_quantile)
        self.residual_minimum_scale = float(residual_minimum_scale)
        self.correction_fraction = (
            None
            if correction_fraction is None
            else float(correction_fraction)
        )
        self.balance_corrections_by_source_event = bool(
            balance_corrections_by_source_event
        )
        self.balance_zeros_by_source_event = bool(
            balance_zeros_by_source_event
        )
        self.strict_recomposition_tolerance = float(
            strict_recomposition_tolerance
        )
        self.return_metadata = bool(return_metadata)

        if self.sample_mode not in {"all", "correction_only", "zero_only"}:
            raise ValueError(
                "sample_mode must be all, correction_only or zero_only"
            )
        if self.split not in {"train", "val", "all"}:
            raise ValueError("split must be train, val or all")
        if not 0 <= self.val_ratio < 1:
            raise ValueError("val_ratio must be in [0, 1)")
        if self.correction_fraction is not None and not (
            0 < self.correction_fraction < 1
        ):
            raise ValueError("correction_fraction must be in (0, 1)")
        if not Path(self.dataset_path).is_file():
            raise FileNotFoundError(
                f"Residual Zarr does not exist: {self.dataset_path}"
            )

        self._store = None
        self._root = None
        self._data = None
        self._opened_pid = None
        self._initialize_index()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_store"] = None
        state["_root"] = None
        state["_data"] = None
        state["_opened_pid"] = None
        return state

    def __del__(self):
        self.close()

    def close(self) -> None:
        if getattr(self, "_store", None) is not None:
            self._store.close()
        self._store = None
        self._root = None
        self._data = None
        self._opened_pid = None

    def _ensure_open(self) -> None:
        pid = os.getpid()
        if self._store is not None and self._opened_pid == pid:
            return
        self.close()
        self._store = zarr.ZipStore(self.dataset_path, mode="r")
        self._root = zarr.group(store=self._store)
        self._data = self._root["data"]
        self._opened_pid = pid

    def _initialize_index(self) -> None:
        self._ensure_open()
        schema = self._root.attrs.get("schema")
        schema_version = int(self._root.attrs.get("schema_version", -1))
        self.recorded_base_checkpoint = str(
            self._root.attrs.get("base_policy_checkpoint", "")
        )
        self.recorded_base_policy_class = str(
            self._root.attrs.get("base_policy_class", "")
        )
        self.recorded_base_checkpoint_sha256 = str(
            self._root.attrs.get("base_policy_checkpoint_sha256", "")
        )
        self.action_alignment = str(
            self._root.attrs.get(
                "action_alignment", ACTION_ALIGNMENT_RECORDED_CHUNK
            )
        )
        self.valid_action_mask_layout = str(
            self._root.attrs.get(
                "valid_action_mask_layout", VALID_MASK_LAYOUT_SUFFIX
            )
        )
        if schema != self.expected_schema:
            raise ValueError(
                f"Unexpected residual schema {schema!r}; expected "
                f"{self.expected_schema!r}"
            )
        if schema_version != self.expected_schema_version:
            raise ValueError(
                "Residual schema version mismatch: expected "
                f"{self.expected_schema_version}, got {schema_version}"
            )
        if self.action_alignment not in {
            ACTION_ALIGNMENT_RECORDED_CHUNK,
            ACTION_ALIGNMENT_ACTIVE_SUFFIX,
            ACTION_ALIGNMENT_STEP_ACTIVE_PLAN,
        }:
            raise ValueError(
                f"Unsupported residual action alignment: {self.action_alignment}"
            )
        expected_layout = (
            VALID_MASK_LAYOUT_PREFIX
            if self.action_alignment
            in {
                ACTION_ALIGNMENT_ACTIVE_SUFFIX,
                ACTION_ALIGNMENT_STEP_ACTIVE_PLAN,
            }
            else VALID_MASK_LAYOUT_SUFFIX
        )
        if self.valid_action_mask_layout != expected_layout:
            raise ValueError(
                "Residual action alignment/mask layout mismatch: "
                f"{self.action_alignment} requires {expected_layout}, got "
                f"{self.valid_action_mask_layout}"
            )
        if (
            self.expected_action_alignment is not None
            and self.action_alignment != self.expected_action_alignment
        ):
            raise ValueError(
                "Residual dataset action alignment mismatch: expected "
                f"{self.expected_action_alignment}, got {self.action_alignment}"
            )

        required = {
            "camera0_rgb",
            "robot_state",
            "episode_start_pose_matrix",
            "frozen_base_action",
            "final_edited_action",
            "residual_position",
            "residual_rotation_axis_angle",
            "residual_gripper",
            "valid_action_mask",
            "anchor_index",
            "correction_label",
            "source_episode_index",
        }
        missing = sorted(required.difference(self._data.keys()))
        if missing:
            raise KeyError(f"Residual Zarr is missing fields: {missing}")

        sample_count = int(self._root.attrs.get("sample_count", 0))
        observation_horizon = int(
            self._root.attrs.get("observation_horizon", -1)
        )
        action_horizon = int(self._root.attrs.get("action_horizon", -1))
        action_dim = int(self._root.attrs.get("action_dim", -1))
        expected_obs_horizon = int(
            self.shape_meta["obs"]["camera0_rgb"]["horizon"]
        )
        expected_action_horizon = (
            int(self.shape_meta["action"]["horizon"])
            if self.expected_action_horizon is None
            else self.expected_action_horizon
        )
        if sample_count <= 0:
            raise ValueError("Residual Zarr contains no samples")
        if observation_horizon != expected_obs_horizon:
            raise ValueError(
                "Observation horizon mismatch: dataset has "
                f"{observation_horizon}, shape_meta expects "
                f"{expected_obs_horizon}"
            )
        if action_horizon != expected_action_horizon:
            raise ValueError(
                f"Action horizon mismatch: dataset has {action_horizon}, "
                f"dataset config expects {expected_action_horizon}"
            )
        if action_dim != ABSOLUTE_ACTION_DIM:
            raise ValueError(
                f"Residual frozen action dim must be 8, got {action_dim}"
            )
        if int(self.shape_meta["action"]["shape"][0]) != BASE_ACTION_DIM:
            raise ValueError("Base UMI action shape must be 10-D")

        labels = np.asarray(self._data["correction_label"][:], dtype=np.uint8)
        episode_ids = np.asarray(
            self._data["source_episode_index"][:], dtype=np.int64
        )
        if labels.shape != (sample_count,) or episode_ids.shape != (
            sample_count,
        ):
            raise ValueError("Residual labels/episode indices have invalid shape")
        if not set(np.unique(labels)).issubset({0, 1}):
            raise ValueError("correction_label must be binary")

        unique_episodes = np.unique(episode_ids)
        if self.val_episode_indices is not None:
            val_episodes = np.asarray(self.val_episode_indices, dtype=np.int64)
            unknown = np.setdiff1d(val_episodes, unique_episodes)
            if len(unknown):
                raise ValueError(
                    f"Unknown validation episode indices: {unknown.tolist()}"
                )
        elif self.val_ratio > 0 and len(unique_episodes) >= 2:
            rng = np.random.default_rng(self.seed)
            val_count = max(1, int(round(len(unique_episodes) * self.val_ratio)))
            val_count = min(val_count, len(unique_episodes) - 1)
            val_episodes = np.sort(
                rng.choice(unique_episodes, val_count, replace=False)
            )
        else:
            val_episodes = np.empty(0, dtype=np.int64)

        split_mask = np.ones(sample_count, dtype=bool)
        if self.split == "train":
            split_mask = ~np.isin(episode_ids, val_episodes)
        elif self.split == "val":
            split_mask = np.isin(episode_ids, val_episodes)

        label_mask = np.ones(sample_count, dtype=bool)
        if self.sample_mode == "correction_only":
            label_mask = labels == 1
        elif self.sample_mode == "zero_only":
            label_mask = labels == 0
        self.indices = np.flatnonzero(split_mask & label_mask).astype(np.int64)
        if self.split != "val" and len(self.indices) == 0:
            raise ValueError(
                f"No samples remain for split={self.split}, "
                f"sample_mode={self.sample_mode}"
            )

        self.labels = labels[self.indices]
        self.episode_ids = episode_ids[self.indices]
        if "source_event_index" in self._data:
            all_family_ids = np.asarray(
                self._data["source_event_index"][:], dtype=np.int64
            )
            if all_family_ids.shape != (sample_count,):
                raise ValueError("source_event_index has an invalid shape")
            self.sample_family_ids = all_family_ids[self.indices]
        elif "multibase_source_sample_index" in self._data:
            all_family_ids = np.asarray(
                self._data["multibase_source_sample_index"][:], dtype=np.int64
            )
            if all_family_ids.shape != (sample_count,):
                raise ValueError(
                    "multibase_source_sample_index has an invalid shape"
                )
            self.sample_family_ids = all_family_ids[self.indices]
        else:
            self.sample_family_ids = self.indices.copy()
        # Historical name retained for external analysis scripts.
        self.correction_family_ids = self.sample_family_ids
        self.validation_episode_indices = tuple(
            int(value) for value in val_episodes.tolist()
        )
        self.sample_count = sample_count
        self.observation_horizon = observation_horizon
        self.action_horizon = action_horizon
        self._audit_masks_and_residuals()
        self.close()

    def _audit_masks_and_residuals(self) -> None:
        if len(self.indices) == 0:
            return
        masks = np.asarray(
            self._data["valid_action_mask"].oindex[self.indices],
            dtype=np.uint8,
        )
        anchors = np.asarray(
            self._data["anchor_index"].oindex[self.indices], dtype=np.int64
        )
        for local_index, (mask, anchor) in enumerate(zip(masks, anchors)):
            expected = np.zeros(self.action_horizon, dtype=np.uint8)
            if self.valid_action_mask_layout == VALID_MASK_LAYOUT_PREFIX:
                valid_length = int(np.count_nonzero(mask))
                if int(anchor) != 0:
                    raise ValueError(
                        f"Sample {int(self.indices[local_index])} has an "
                        "active-suffix anchor other than zero"
                    )
                expected[:valid_length] = 1
            else:
                expected[int(anchor) :] = 1
            if not np.array_equal(mask, expected):
                source_index = int(self.indices[local_index])
                raise ValueError(
                    f"Sample {source_index} has an invalid "
                    f"{self.valid_action_mask_layout} mask"
                )

        # Strictly check stored residual fields against base/final geometry.
        base = np.asarray(
            self._data["frozen_base_action"].oindex[self.indices]
        )
        edited = np.asarray(
            self._data["final_edited_action"].oindex[self.indices]
        )
        stored = np.concatenate(
            [
                np.asarray(
                    self._data["residual_position"].oindex[self.indices]
                ),
                np.asarray(
                    self._data["residual_rotation_axis_angle"].oindex[
                        self.indices
                    ]
                ),
                np.asarray(
                    self._data["residual_gripper"].oindex[self.indices]
                ),
            ],
            axis=-1,
        )
        for local_index in range(len(self.indices)):
            expected = compute_world_frame_residual(
                base[local_index], edited[local_index]
            )
            if not np.allclose(
                expected, stored[local_index], atol=self.strict_recomposition_tolerance
            ):
                raise ValueError(
                    f"Sample {int(self.indices[local_index])} stores an "
                    "inconsistent residual"
                )
            errors = residual_recomposition_errors(
                base[local_index], edited[local_index], stored[local_index]
            )
            if max(float(np.max(value)) for value in errors.values()) > (
                self.strict_recomposition_tolerance
            ):
                raise ValueError(
                    f"Sample {int(self.indices[local_index])} fails residual "
                    "recomposition"
                )

    def __len__(self) -> int:
        return len(self.indices)

    def _load_array(self, key: str, source_index: int) -> np.ndarray:
        self._ensure_open()
        return np.asarray(self._data[key][source_index])

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        source_index = int(self.indices[index])
        camera = self._load_array("camera0_rgb", source_index)
        robot_state = self._load_array("robot_state", source_index)
        episode_start_pose = self._load_array(
            "episode_start_pose_matrix", source_index
        )
        base_action_absolute = self._load_array(
            "frozen_base_action", source_index
        )
        residual_action = np.concatenate(
            [
                self._load_array("residual_position", source_index),
                self._load_array(
                    "residual_rotation_axis_angle", source_index
                ),
                self._load_array("residual_gripper", source_index),
            ],
            axis=-1,
        ).astype(np.float32)
        valid_mask = self._load_array(
            "valid_action_mask", source_index
        ).astype(bool)

        if camera.shape[0] != self.observation_horizon:
            raise ValueError(f"Sample {source_index} camera horizon changed")
        if robot_state.shape != (self.observation_horizon, 22):
            raise ValueError(
                f"Sample {source_index} robot_state must have shape "
                f"({self.observation_horizon}, 22), got {robot_state.shape}"
            )
        pose_matrix = flexiv_state_to_pose_matrix(robot_state)
        latest_pose = pose_matrix[-1]
        relative_pose = convert_pose_mat_rep(
            pose_matrix,
            base_pose_mat=latest_pose,
            pose_rep="relative",
            backward=False,
        )
        relative_pose9 = mat_to_pose10d(relative_pose)
        wrt_start = convert_pose_mat_rep(
            pose_matrix,
            base_pose_mat=episode_start_pose,
            pose_rep="relative",
            backward=False,
        )
        wrt_start_pose9 = mat_to_pose10d(wrt_start)
        base_action = absolute_action8_to_relative_action10(
            base_action_absolute, latest_pose
        )

        obs = {
            "camera0_rgb": torch.from_numpy(
                np.moveaxis(camera, -1, 1).astype(np.float32) / 255.0
            ),
            "robot0_eef_pos": torch.from_numpy(
                relative_pose9[:, :3].astype(np.float32)
            ),
            "robot0_eef_rot_axis_angle": torch.from_numpy(
                relative_pose9[:, 3:].astype(np.float32)
            ),
            "robot0_eef_rot_axis_angle_wrt_start": torch.from_numpy(
                wrt_start_pose9[:, 3:].astype(np.float32)
            ),
            "robot0_gripper_width": torch.from_numpy(
                flexiv_state_to_gripper(robot_state).astype(np.float32)
            ),
        }
        sample = {
            "obs": obs,
            "base_action": torch.from_numpy(base_action),
            "residual_action": torch.from_numpy(residual_action),
            "valid_action_mask": torch.from_numpy(valid_mask),
            "correction_label": torch.tensor(
                int(self._load_array("correction_label", source_index)),
                dtype=torch.float32,
            ),
        }
        if self.return_metadata:
            sample.update(
                {
                    "source_index": torch.tensor(source_index),
                    "source_episode_index": torch.tensor(
                        int(
                            self._load_array(
                                "source_episode_index", source_index
                            )
                        )
                    ),
                    "anchor_index": torch.tensor(
                        int(self._load_array("anchor_index", source_index))
                    ),
                }
            )
            for metadata_key in (
                "source_event_index",
                "source_action_index",
                "policy_chunk_generation_id",
            ):
                if metadata_key in self._data:
                    sample[metadata_key] = torch.tensor(
                        int(self._load_array(metadata_key, source_index))
                    )
        return sample

    def get_validation_dataset(self) -> "UmiResidualDataset":
        return UmiResidualDataset(
            dataset_path=self.dataset_path,
            shape_meta=self.shape_meta,
            sample_mode=self.sample_mode,
            split="val",
            seed=self.seed,
            val_ratio=self.val_ratio,
            val_episode_indices=self.validation_episode_indices,
            expected_schema=self.expected_schema,
            expected_schema_version=self.expected_schema_version,
            expected_action_alignment=self.expected_action_alignment,
            action_horizon=self.expected_action_horizon,
            residual_scale_quantile=self.residual_scale_quantile,
            residual_minimum_scale=self.residual_minimum_scale,
            correction_fraction=self.correction_fraction,
            balance_corrections_by_source_event=(
                self.balance_corrections_by_source_event
            ),
            balance_zeros_by_source_event=self.balance_zeros_by_source_event,
            strict_recomposition_tolerance=self.strict_recomposition_tolerance,
            return_metadata=self.return_metadata,
        )

    def get_sampling_weights(self) -> Optional[torch.Tensor]:
        """Return per-row weights that realize a target correction fraction."""
        if self.correction_fraction is None or self.sample_mode != "all":
            return None
        correction = self.labels == 1
        zero = self.labels == 0
        correction_count = int(np.count_nonzero(correction))
        zero_count = int(np.count_nonzero(zero))
        if correction_count == 0 or zero_count == 0:
            raise ValueError(
                "Balanced correction sampling requires both correction and "
                "zero-attempt samples"
            )
        weights = np.empty(len(self), dtype=np.float64)
        family_ids = getattr(
            self, "sample_family_ids", self.correction_family_ids
        )
        if self.balance_corrections_by_source_event:
            correction_families = family_ids[correction]
            unique_families, family_counts = np.unique(
                correction_families, return_counts=True
            )
            family_count_by_id = dict(zip(unique_families, family_counts))
            correction_family_count = len(unique_families)
            weights[correction] = np.asarray(
                [
                    self.correction_fraction
                    / correction_family_count
                    / family_count_by_id[family_id]
                    for family_id in correction_families
                ],
                dtype=np.float64,
            )
        else:
            weights[correction] = self.correction_fraction / correction_count
        if getattr(self, "balance_zeros_by_source_event", False):
            zero_families = family_ids[zero]
            unique_families, family_counts = np.unique(
                zero_families, return_counts=True
            )
            family_count_by_id = dict(zip(unique_families, family_counts))
            weights[zero] = np.asarray(
                [
                    (1.0 - self.correction_fraction)
                    / len(unique_families)
                    / family_count_by_id[family_id]
                    for family_id in zero_families
                ],
                dtype=np.float64,
            )
        else:
            weights[zero] = (1.0 - self.correction_fraction) / zero_count
        return torch.from_numpy(weights)

    def get_residual_normalizer(self) -> SingleFieldLinearNormalizer:
        """Fit a zero-preserving normalizer on valid correction tokens only."""
        self._ensure_open()
        correction_indices = self.indices[self.labels == 1]
        if len(correction_indices) == 0:
            raise ValueError("Residual normalizer requires correction samples")
        residual = np.concatenate(
            [
                np.asarray(
                    self._data["residual_position"].oindex[
                        correction_indices
                    ]
                ),
                np.asarray(
                    self._data["residual_rotation_axis_angle"].oindex[
                        correction_indices
                    ]
                ),
                np.asarray(
                    self._data["residual_gripper"].oindex[
                        correction_indices
                    ]
                ),
            ],
            axis=-1,
        )
        masks = np.asarray(
            self._data["valid_action_mask"].oindex[correction_indices],
            dtype=bool,
        )
        valid_residual = residual[masks]
        scale, stats = fit_zero_centered_scale(
            valid_residual,
            quantile=self.residual_scale_quantile,
            minimum_scale=self.residual_minimum_scale,
        )
        return SingleFieldLinearNormalizer.create_manual(
            scale=scale,
            offset=np.zeros(RESIDUAL_ACTION_DIM, dtype=np.float32),
            input_stats_dict=stats,
        )

    def get_all_actions(self) -> torch.Tensor:
        actions = [self[index]["residual_action"] for index in range(len(self))]
        if not actions:
            return torch.empty((0, self.action_horizon, RESIDUAL_ACTION_DIM))
        return torch.stack(actions)

    def audit_summary(self) -> Dict[str, object]:
        coverage = np.zeros(self.action_horizon, dtype=np.int64)
        if len(self.indices):
            self._ensure_open()
            correction_indices = self.indices[self.labels == 1]
            if len(correction_indices):
                coverage = np.asarray(
                    self._data["valid_action_mask"].oindex[
                        correction_indices
                    ]
                ).sum(axis=0)
        return {
            "dataset_path": self.dataset_path,
            "split": self.split,
            "sample_mode": self.sample_mode,
            "sample_count": len(self),
            "correction_count": int(np.count_nonzero(self.labels == 1)),
            "zero_count": int(np.count_nonzero(self.labels == 0)),
            "episode_count": int(len(np.unique(self.episode_ids))),
            "validation_episode_indices": list(
                self.validation_episode_indices
            ),
            "valid_correction_coverage": coverage.tolist(),
            "recorded_base_checkpoint": self.recorded_base_checkpoint,
            "recorded_base_policy_class": self.recorded_base_policy_class,
            "recorded_base_checkpoint_sha256": (
                self.recorded_base_checkpoint_sha256
            ),
            "action_alignment": self.action_alignment,
            "valid_action_mask_layout": self.valid_action_mask_layout,
            "balance_corrections_by_source_event": (
                self.balance_corrections_by_source_event
            ),
            "balance_zeros_by_source_event": self.balance_zeros_by_source_event,
            "correction_family_count": int(
                len(np.unique(self.sample_family_ids[self.labels == 1]))
            ),
            "zero_family_count": int(
                len(np.unique(self.sample_family_ids[self.labels == 0]))
            ),
        }
