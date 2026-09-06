#!/usr/bin/env python3
"""Offline, category-aware evaluation for chunk residual checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable

import dill
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
from diffusion_policy.common.frozen_base_policy_loader import (
    load_frozen_base_policy,
)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.umi_residual_dataset import UmiResidualDataset
from diffusion_policy.model.residual.chunk_residual_mlp import ChunkResidualMLP
from diffusion_policy.policy.chunk_residual_policy import ChunkResidualPolicy


register_codecs()

KIND_NAMES = {0: "zero", 1: "human_correction", 2: "accepted_rollout"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _unpickle(payload: dict, key: str, default=None):
    value = payload.get("pickles", {}).get(key)
    return default if value is None else dill.loads(value)


def _load_checkpoint(
    checkpoint_path: Path,
    base_checkpoint_path: Path | None,
    device: torch.device,
    state_key: str,
) -> tuple[Any, ChunkResidualPolicy, dict]:
    with checkpoint_path.open("rb") as stream:
        payload = torch.load(stream, map_location="cpu", pickle_module=dill)
    cfg = payload["cfg"]
    if base_checkpoint_path is None:
        recorded_path = _unpickle(payload, "base_checkpoint_path")
        if not recorded_path:
            recorded_path = OmegaConf.select(cfg, "base_policy.ckpt_path")
        base_checkpoint_path = Path(str(recorded_path)).expanduser()
    base_checkpoint_path = base_checkpoint_path.resolve()
    expected_sha = _unpickle(payload, "base_checkpoint_sha256", "")
    actual_sha = _sha256(base_checkpoint_path)
    if expected_sha and actual_sha != expected_sha:
        raise ValueError(
            f"Frozen-base SHA mismatch: {actual_sha} != {expected_sha}"
        )
    frozen = load_frozen_base_policy(
        base_checkpoint_path, device="cpu", prefer_ema=True
    )
    base_policy = frozen.policy.to(device).eval().requires_grad_(False)
    feature_dim = int(np.prod(base_policy.obs_encoder.output_shape()))
    model_cfg = OmegaConf.to_container(cfg.policy.model, resolve=True)
    model_cfg.pop("_target_", None)
    model = ChunkResidualMLP(obs_feature_dim=feature_dim, **model_cfg)
    policy_cfg = OmegaConf.to_container(cfg.policy, resolve=True)
    policy_cfg.pop("_target_", None)
    policy_cfg.pop("model", None)
    residual_policy = ChunkResidualPolicy(model=model, **policy_cfg)
    state_dicts = payload["state_dicts"]
    if state_key == "auto":
        state_key = "ema_model" if "ema_model" in state_dicts else "model"
    residual_policy.load_state_dict(state_dicts[state_key], strict=True)
    residual_policy.to(device).eval().requires_grad_(False)
    metadata = {
        "checkpoint": str(checkpoint_path.resolve()),
        "state_key": state_key,
        "base_checkpoint": str(base_checkpoint_path),
        "base_checkpoint_sha256": actual_sha,
        "action_alignment": str(
            _unpickle(payload, "residual_action_alignment", "recorded_chunk")
        ),
    }
    return base_policy, residual_policy, metadata


def _summary(values: Iterable[float], scale: float = 1.0) -> Dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64) * scale
    if len(array) == 0:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


@torch.no_grad()
def evaluate_dataset(
    dataset_path: Path,
    shape_meta: dict,
    expected_alignment: str,
    base_policy: Any,
    residual_policy: ChunkResidualPolicy,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> dict:
    dataset = UmiResidualDataset(
        dataset_path=str(dataset_path.resolve()),
        shape_meta=shape_meta,
        sample_mode="all",
        split="all",
        val_ratio=0.0,
        correction_fraction=None,
        expected_action_alignment=expected_alignment,
        return_metadata=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    groups: dict[int, dict[str, list]] = defaultdict(
        lambda: defaultdict(list)
    )
    for batch in loader:
        batch = dict_apply(
            batch, lambda value: value.to(device, non_blocking=True)
        )
        normalized_obs = base_policy.normalizer.normalize(batch["obs"])
        obs_feature = base_policy.obs_encoder(normalized_obs)
        normalized_base = base_policy.normalizer["action"].normalize(
            batch["base_action"]
        )
        prediction = residual_policy.predict_residual(
            obs_feature,
            normalized_base,
            gate_mode="disabled",
        )["residual"]
        target = batch["residual_action"]
        valid = batch["valid_action_mask"].bool()
        kinds = batch["target_kind"].to(torch.int64)
        for row in range(len(kinds)):
            kind = int(kinds[row])
            mask = valid[row]
            predicted_row = prediction[row][mask]
            target_row = target[row][mask]
            error = predicted_row - target_row
            group = groups[kind]
            group["row_count"].append(1)
            group["token_count"].append(int(mask.sum()))
            group["position_error"].extend(
                torch.linalg.vector_norm(error[:, :3], dim=-1).cpu().tolist()
            )
            group["rotation_error"].extend(
                torch.linalg.vector_norm(error[:, 3:6], dim=-1).cpu().tolist()
            )
            group["gripper_error"].extend(
                torch.abs(error[:, 6]).cpu().tolist()
            )
            group["target_position"].extend(
                torch.linalg.vector_norm(target_row[:, :3], dim=-1)
                .cpu()
                .tolist()
            )
            group["predicted_position"].extend(
                torch.linalg.vector_norm(predicted_row[:, :3], dim=-1)
                .cpu()
                .tolist()
            )
            group["step0_position_error"].append(
                float(torch.linalg.vector_norm(error[0, :3]).cpu())
            )
            group["step0_predicted_position"].append(
                float(torch.linalg.vector_norm(predicted_row[0, :3]).cpu())
            )
            group["step0_target_position"].append(
                float(torch.linalg.vector_norm(target_row[0, :3]).cpu())
            )
            group["source_index"].append(int(batch["source_index"][row]))
            group["row_diagnostics"].append(
                {
                    "source_index": int(batch["source_index"][row]),
                    "source_episode_index": int(
                        batch["source_episode_index"][row]
                    ),
                    "anchor_index": int(batch["anchor_index"][row]),
                    "target_gripper_max_mm": float(
                        torch.abs(target_row[:, 6]).max().cpu() * 1000.0
                    ),
                    "predicted_gripper_max_mm": float(
                        torch.abs(predicted_row[:, 6]).max().cpu() * 1000.0
                    ),
                    "gripper_error_mean_mm": float(
                        torch.abs(error[:, 6]).mean().cpu() * 1000.0
                    ),
                    "gripper_error_max_mm": float(
                        torch.abs(error[:, 6]).max().cpu() * 1000.0
                    ),
                }
            )

    result_groups = {}
    for kind, values in sorted(groups.items()):
        result_groups[KIND_NAMES.get(kind, str(kind))] = {
            "row_count": int(sum(values["row_count"])),
            "token_count": int(sum(values["token_count"])),
            "position_error_mm": _summary(values["position_error"], 1000.0),
            "rotation_error_deg": _summary(
                values["rotation_error"], 180.0 / np.pi
            ),
            "gripper_error_mm": _summary(values["gripper_error"], 1000.0),
            "target_position_mm": _summary(values["target_position"], 1000.0),
            "predicted_position_mm": _summary(
                values["predicted_position"], 1000.0
            ),
            "step0_position_error_mm": _summary(
                values["step0_position_error"], 1000.0
            ),
            "step0_predicted_position_mm": _summary(
                values["step0_predicted_position"], 1000.0
            ),
            "step0_target_position_mm": _summary(
                values["step0_target_position"], 1000.0
            ),
            "largest_target_gripper_rows": sorted(
                values["row_diagnostics"],
                key=lambda row: row["target_gripper_max_mm"],
                reverse=True,
            )[:5],
        }
    dataset.close()
    return {
        "dataset": str(dataset_path.resolve()),
        "sample_count": len(dataset),
        "groups": result_groups,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("datasets", type=Path, nargs="+")
    parser.add_argument("--base-policy-ckpt", type=Path, default=None)
    parser.add_argument("--state-key", default="auto")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    base_policy, residual_policy, report = _load_checkpoint(
        args.checkpoint.resolve(),
        args.base_policy_ckpt,
        device,
        args.state_key,
    )
    shape_meta_cfg = torch.load(
        args.checkpoint.resolve(),
        map_location="cpu",
        pickle_module=dill,
    )["cfg"].shape_meta
    shape_meta = OmegaConf.to_container(shape_meta_cfg, resolve=True)
    report["datasets"] = [
        evaluate_dataset(
            path,
            shape_meta,
            report["action_alignment"],
            base_policy,
            residual_policy,
            device,
            args.batch_size,
            args.num_workers,
        )
        for path in args.datasets
    ]
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
