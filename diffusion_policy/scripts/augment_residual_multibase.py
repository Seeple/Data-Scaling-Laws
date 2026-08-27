#!/usr/bin/env python3
"""Materialize filtered frozen-policy resamples for residual training.

Each accepted row keeps the correction observation and edited goal fixed, but
replaces the recorded active base plan with another stochastic sample from the
exact frozen Diffusion Policy. Candidates are admitted only when the edited
goal remains geometrically compatible with the recorded active plan.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
import tempfile
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import zarr
from omegaconf import OmegaConf

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
from diffusion_policy.common.frozen_base_policy_loader import load_frozen_base_policy
from diffusion_policy.common.residual_action_util import (
    compute_world_frame_residual,
    flexiv_state_to_pose_matrix,
    relative_action10_to_absolute_action8,
)
from diffusion_policy.common.residual_multibase_util import (
    MultiBaseFilterConfig,
    evaluate_multibase_candidate,
)
from diffusion_policy.dataset.umi_residual_dataset import (
    ACTION_ALIGNMENT_ACTIVE_SUFFIX,
    ACTION_ALIGNMENT_STEP_ACTIVE_PLAN,
    EXPECTED_SCHEMA,
    STEP_ACTIVE_PLAN_SCHEMA,
    UmiResidualDataset,
)


register_codecs()


def _shape_meta_from_checkpoint(cfg: Any) -> dict:
    if OmegaConf.select(cfg, "shape_meta") is not None:
        value = cfg.shape_meta
    elif OmegaConf.select(cfg, "task.shape_meta") is not None:
        value = cfg.task.shape_meta
    else:
        raise ValueError("Base checkpoint config has no shape_meta")
    return OmegaConf.to_container(value, resolve=True)


def _repeat_obs(obs: Dict[str, torch.Tensor], count: int, device: torch.device):
    return {
        key: value.unsqueeze(0).repeat((count,) + (1,) * value.ndim).to(
            device=device, non_blocking=True
        )
        for key, value in obs.items()
    }


def _copy_source_rows(source: zarr.Array, destination: zarr.Array) -> None:
    chunk = max(1, int(source.chunks[0]))
    for start in range(0, source.shape[0], chunk):
        stop = min(source.shape[0], start + chunk)
        destination[start:stop] = source[start:stop]


def _write_augmented_zarr(
    input_path: Path,
    output_path: Path,
    augmented: List[Dict[str, Any]],
    summary: Dict[str, Any],
    overwrite: bool,
) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {output_path}; pass --overwrite")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(
        tempfile.mkdtemp(prefix="residual_multibase_", dir=output_path.parent)
    )
    input_store = zarr.ZipStore(str(input_path), mode="r")
    try:
        source_root = zarr.group(store=input_store)
        source_data = source_root["data"]
        source_count = int(source_root.attrs["sample_count"])
        total_count = source_count + len(augmented)
        directory_store = zarr.DirectoryStore(str(temp_dir / "dataset.zarr"))
        output_root = zarr.group(store=directory_store)
        output_data = output_root.create_group("data")
        output_meta = output_root.create_group("meta")

        override_keys = {
            "frozen_base_action",
            "residual_position",
            "residual_rotation_axis_angle",
            "residual_gripper",
        }
        for key in source_data.keys():
            source = source_data[key]
            chunks = (min(source.chunks[0], total_count),) + source.chunks[1:]
            destination = output_data.create_dataset(
                key,
                shape=(total_count,) + source.shape[1:],
                chunks=chunks,
                dtype=source.dtype,
                compressor=source.compressor,
                filters=source.filters,
                fill_value=source.fill_value,
            )
            _copy_source_rows(source, destination)
            for offset, record in enumerate(augmented):
                destination[source_count + offset] = (
                    record[key]
                    if key in override_keys
                    else source[int(record["source_index"])]
                )

        resample_index = np.full(total_count, -1, dtype=np.int64)
        resample_source_index = np.arange(total_count, dtype=np.int64)
        for offset, record in enumerate(augmented):
            row = source_count + offset
            resample_index[row] = int(record["resample_index"])
            resample_source_index[row] = int(record["source_index"])
        output_data.create_dataset(
            "multibase_resample_index",
            data=resample_index,
            chunks=(min(1024, total_count),),
        )
        output_data.create_dataset(
            "multibase_source_sample_index",
            data=resample_source_index,
            chunks=(min(1024, total_count),),
        )

        source_records = list(source_root["meta"].attrs.get("records", []))
        output_records = copy.deepcopy(source_records)
        for record in augmented:
            metadata = copy.deepcopy(source_records[int(record["source_index"])])
            metadata.update(
                {
                    "sample_origin": "filtered_multibase_resample",
                    "multibase_source_sample_index": int(record["source_index"]),
                    "multibase_resample_index": int(record["resample_index"]),
                    "multibase_filter_metrics": record["metrics"],
                }
            )
            output_records.append(metadata)
        output_meta.attrs["records"] = output_records

        output_root.attrs.update(dict(source_root.attrs))
        output_root.attrs.update(
            {
                "sample_count": total_count,
                "multibase_resampling_enabled": True,
                "multibase_original_sample_count": source_count,
                "multibase_augmented_sample_count": len(augmented),
                "multibase_filter_config": summary["filter_config"],
                "multibase_candidates_per_correction": summary[
                    "candidates_per_correction"
                ],
                "multibase_seed": summary["seed"],
            }
        )
        if output_path.exists():
            output_path.unlink()
        with zarr.ZipStore(str(output_path), mode="w") as output_store:
            zarr.copy_store(directory_store, output_store, if_exists="replace")
    except Exception:
        output_path.unlink(missing_ok=True)
        raise
    finally:
        input_store.close()
        shutil.rmtree(temp_dir, ignore_errors=True)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    input_path = args.input_path.expanduser().resolve()
    output_path = args.output_path.expanduser().resolve()
    frozen = load_frozen_base_policy(
        args.base_policy_ckpt,
        device="cpu",
        prefer_ema=not args.use_model_weights,
    )
    shape_meta = _shape_meta_from_checkpoint(frozen.checkpoint_cfg)
    with zarr.ZipStore(str(input_path), mode="r") as inspection_store:
        inspection_root = zarr.group(store=inspection_store)
        source_schema = str(inspection_root.attrs.get("schema", ""))
        source_horizon = int(inspection_root.attrs.get("action_horizon", -1))
    if source_schema == EXPECTED_SCHEMA:
        expected_alignment = ACTION_ALIGNMENT_ACTIVE_SUFFIX
        dataset_horizon = None
    elif source_schema == STEP_ACTIVE_PLAN_SCHEMA:
        expected_alignment = ACTION_ALIGNMENT_STEP_ACTIVE_PLAN
        dataset_horizon = source_horizon
    else:
        raise ValueError(
            f"Unsupported residual dataset schema: {source_schema!r}"
        )
    dataset = UmiResidualDataset(
        dataset_path=str(input_path),
        shape_meta=shape_meta,
        sample_mode="all",
        split="all",
        val_ratio=0.0,
        correction_fraction=None,
        expected_schema=source_schema,
        expected_action_alignment=expected_alignment,
        action_horizon=dataset_horizon,
        return_metadata=True,
    )
    if (
        dataset.recorded_base_checkpoint_sha256
        and dataset.recorded_base_checkpoint_sha256 != frozen.checkpoint_sha256
    ):
        raise ValueError(
            "Input residual dataset and --base-policy-ckpt SHA256 do not match"
        )

    device_name = (
        "cuda" if torch.cuda.is_available() else "cpu"
    ) if args.device == "auto" else args.device
    device = torch.device(device_name)
    frozen.policy.to(device).eval().requires_grad_(False)
    filter_config = MultiBaseFilterConfig(
        comparison_steps=args.comparison_steps,
        max_first_position_deviation_m=args.max_first_position_deviation_m,
        max_prefix_position_deviation_m=args.max_prefix_position_deviation_m,
        max_prefix_rotation_deviation_rad=args.max_prefix_rotation_deviation_rad,
        max_prefix_gripper_deviation_m=args.max_prefix_gripper_deviation_m,
        max_target_position_residual_m=args.max_target_position_residual_m,
        max_target_rotation_residual_rad=args.max_target_rotation_residual_rad,
        max_target_gripper_residual_m=args.max_target_gripper_residual_m,
        min_diversity_position_m=args.min_diversity_position_m,
        min_diversity_rotation_rad=args.min_diversity_rotation_rad,
    )

    dataset._ensure_open()
    augmented: List[Dict[str, Any]] = []
    rejection_reasons: Counter[str] = Counter()
    correction_local_indices = np.flatnonzero(dataset.labels == 1)
    for correction_number, local_index in enumerate(correction_local_indices):
        source_index = int(dataset.indices[int(local_index)])
        sample = dataset[int(local_index)]
        recorded_base = np.asarray(
            dataset._data["frozen_base_action"][source_index], dtype=np.float64
        )
        edited_goal = np.asarray(
            dataset._data["final_edited_action"][source_index], dtype=np.float64
        )
        valid_mask = np.asarray(
            dataset._data["valid_action_mask"][source_index], dtype=bool
        )
        latest_pose = flexiv_state_to_pose_matrix(
            np.asarray(dataset._data["robot_state"][source_index])
        )[-1]
        accepted_bases: List[np.ndarray] = [recorded_base]
        accepted_count = 0
        draw_count = 0
        max_draws = args.candidates_per_correction * args.max_draw_multiplier
        torch.manual_seed(args.seed + source_index)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed + source_index)

        while accepted_count < args.candidates_per_correction and draw_count < max_draws:
            batch_count = min(args.sample_batch_size, max_draws - draw_count)
            obs_batch = _repeat_obs(sample["obs"], batch_count, device)
            with torch.no_grad():
                prediction = frozen.policy.predict_action(obs_batch)
            relative_candidates = prediction.get(
                "action_pred", prediction.get("action")
            )
            if relative_candidates is None:
                raise KeyError("Frozen base policy returned no action/action_pred")
            relative_candidates = relative_candidates.detach().cpu().numpy()
            for relative_candidate in relative_candidates:
                draw_count += 1
                # A step-active sample is left-aligned at the current
                # observation, so its H-step base window is the prefix of a
                # freshly resampled full Diffusion Policy chunk.
                relative_candidate = relative_candidate[
                    : dataset.action_horizon
                ]
                candidate = relative_action10_to_absolute_action8(
                    relative_candidate, latest_pose
                )
                accepted, reason, metrics = evaluate_multibase_candidate(
                    candidate,
                    recorded_base,
                    edited_goal,
                    valid_mask,
                    accepted_bases,
                    filter_config,
                )
                if not accepted:
                    rejection_reasons[reason] += 1
                    continue
                residual = compute_world_frame_residual(candidate, edited_goal)
                augmented.append(
                    {
                        "source_index": source_index,
                        "resample_index": accepted_count,
                        "frozen_base_action": candidate.astype(np.float32),
                        "residual_position": residual[:, :3],
                        "residual_rotation_axis_angle": residual[:, 3:6],
                        "residual_gripper": residual[:, 6:7],
                        "metrics": metrics,
                    }
                )
                accepted_bases.append(candidate)
                accepted_count += 1
                if accepted_count >= args.candidates_per_correction:
                    break
        print(
            f"[{correction_number + 1}/{len(correction_local_indices)}] "
            f"source={source_index} accepted={accepted_count}/{args.candidates_per_correction} "
            f"draws={draw_count}"
        )

    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "base_policy_checkpoint": frozen.checkpoint_path,
        "base_policy_checkpoint_sha256": frozen.checkpoint_sha256,
        "dataset_schema": source_schema,
        "action_alignment": expected_alignment,
        "action_horizon": dataset.action_horizon,
        "seed": args.seed,
        "candidates_per_correction": args.candidates_per_correction,
        "correction_count": int(len(correction_local_indices)),
        "accepted_candidate_count": len(augmented),
        "fully_augmented_correction_count": int(
            sum(
                sum(row["source_index"] == int(source) for row in augmented)
                == args.candidates_per_correction
                for source in dataset.indices[dataset.labels == 1]
            )
        ),
        "rejection_reasons": dict(rejection_reasons),
        "filter_config": asdict(filter_config),
    }
    if args.require_full_count and len(augmented) != (
        len(correction_local_indices) * args.candidates_per_correction
    ):
        raise ValueError(
            "Filtered sampling did not produce the requested number of candidates; "
            "rerun with an audited threshold change or without --require-full-count"
        )
    dataset.close()
    _write_augmented_zarr(
        input_path, output_path, augmented, summary, args.overwrite
    )
    report_path = output_path.with_suffix(output_path.suffix + ".multibase.json")
    with report_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Wrote report: {report_path}")
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Append filtered frozen-policy resamples to active-suffix or "
            "step-active-plan residual Zarr"
        )
    )
    parser.add_argument("input_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--base-policy-ckpt", type=Path, required=True)
    parser.add_argument("--candidates-per-correction", type=int, default=2)
    parser.add_argument("--max-draw-multiplier", type=int, default=8)
    parser.add_argument("--sample-batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--use-model-weights", action="store_true")
    parser.add_argument("--require-full-count", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    defaults = MultiBaseFilterConfig()
    parser.add_argument("--comparison-steps", type=int, default=defaults.comparison_steps)
    for name in (
        "max_first_position_deviation_m",
        "max_prefix_position_deviation_m",
        "max_prefix_rotation_deviation_rad",
        "max_prefix_gripper_deviation_m",
        "max_target_position_residual_m",
        "max_target_rotation_residual_rad",
        "max_target_gripper_residual_m",
        "min_diversity_position_m",
        "min_diversity_rotation_rad",
    ):
        parser.add_argument(
            "--" + name.replace("_", "-"),
            type=float,
            default=getattr(defaults, name),
        )
    args = parser.parse_args()
    for name in (
        "candidates_per_correction",
        "max_draw_multiplier",
        "sample_batch_size",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


if __name__ == "__main__":
    run(_parse_args())
