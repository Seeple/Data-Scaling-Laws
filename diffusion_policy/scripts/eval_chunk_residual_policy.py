#!/usr/bin/env python3
"""Offline evaluation for a trained chunk residual checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import dill
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from diffusion_policy.common.frozen_base_policy_loader import (
    load_frozen_base_policy,
)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.umi_residual_dataset import UmiResidualDataset
from diffusion_policy.model.residual.chunk_residual_mlp import ChunkResidualMLP
from diffusion_policy.policy.chunk_residual_policy import ChunkResidualPolicy


def load_residual_policy(
    checkpoint_path: Path,
    base_policy,
    device: torch.device,
) -> tuple[ChunkResidualPolicy, object]:
    with checkpoint_path.open("rb") as stream:
        payload = torch.load(
            stream, map_location="cpu", pickle_module=dill
        )
    cfg = payload["cfg"]
    feature_dim = int(np.prod(base_policy.obs_encoder.output_shape()))
    model_cfg = OmegaConf.to_container(cfg.policy.model, resolve=True)
    model_cfg.pop("_target_", None)
    model = ChunkResidualMLP(obs_feature_dim=feature_dim, **model_cfg)
    policy_cfg = OmegaConf.to_container(cfg.policy, resolve=True)
    policy_cfg.pop("_target_", None)
    policy_cfg.pop("model", None)
    policy = ChunkResidualPolicy(model=model, **policy_cfg)
    state_dicts = payload["state_dicts"]
    state_key = "ema_model" if "ema_model" in state_dicts else "model"
    policy.load_state_dict(state_dicts[state_key], strict=True)
    policy.eval().to(device)
    return policy, payload


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    residual_payload = torch.load(
        args.residual_checkpoint,
        map_location="cpu",
        pickle_module=dill,
    )
    cfg = residual_payload["cfg"]
    stored_base_path = residual_payload.get("pickles", {}).get(
        "base_checkpoint_path"
    )
    if stored_base_path is not None:
        stored_base_path = dill.loads(stored_base_path)
    base_path = args.base_checkpoint or stored_base_path
    if base_path is None:
        base_path = cfg.base_policy.ckpt_path
    if base_path is None:
        raise ValueError(
            "No base checkpoint path is available; pass --base-checkpoint"
        )
    frozen = load_frozen_base_policy(base_path, device=device, prefer_ema=True)
    expected_hash = residual_payload.get("pickles", {}).get(
        "base_checkpoint_sha256"
    )
    if expected_hash is not None:
        expected_hash = dill.loads(expected_hash)
        if expected_hash != frozen.checkpoint_sha256:
            raise ValueError(
                "Base checkpoint SHA256 does not match residual checkpoint"
            )
    residual_policy, _ = load_residual_policy(
        args.residual_checkpoint, frozen.policy, device
    )
    dataset_cfg = OmegaConf.to_container(cfg.task.dataset, resolve=True)
    for key in (
        "_target_",
        "dataset_path",
        "shape_meta",
        "sample_mode",
        "split",
        "val_ratio",
        "val_episode_indices",
        "correction_fraction",
        "return_metadata",
    ):
        dataset_cfg.pop(key, None)
    dataset = UmiResidualDataset(
        dataset_path=str(args.dataset),
        shape_meta=OmegaConf.to_container(cfg.shape_meta, resolve=True),
        sample_mode=args.sample_mode,
        split="all",
        val_ratio=0.0,
        correction_fraction=None,
        return_metadata=True,
        **dataset_cfg,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    correction_base_position = []
    correction_pred_position = []
    correction_base_rotation = []
    correction_pred_rotation = []
    correction_base_gripper = []
    correction_pred_gripper = []
    zero_position = []
    zero_rotation = []
    zero_gripper = []
    gate_probabilities = []
    gate_labels = []

    for batch in dataloader:
        batch = dict_apply(batch, lambda value: value.to(device))
        normalized_obs = frozen.policy.normalizer.normalize(batch["obs"])
        obs_feature = frozen.policy.obs_encoder(normalized_obs)
        normalized_base = frozen.policy.normalizer["action"].normalize(
            batch["base_action"]
        )
        prediction = residual_policy.predict_residual(
            obs_feature,
            normalized_base,
            gate_mode=args.gate_mode,
            gate_threshold=args.gate_threshold,
        )
        predicted_residual = prediction["applied_residual"]
        target = batch["residual_action"]
        correction_mask = (
            batch["valid_action_mask"].bool()
            & (batch["correction_label"] > 0.5)[:, None]
        )
        zero_mask = (
            torch.ones_like(batch["valid_action_mask"], dtype=torch.bool)
            & (batch["correction_label"] <= 0.5)[:, None]
        )
        if correction_mask.any():
            target_valid = target[correction_mask]
            error = predicted_residual[correction_mask] - target_valid
            correction_base_position.append(
                torch.linalg.vector_norm(target_valid[:, :3], dim=-1).cpu()
            )
            correction_pred_position.append(
                torch.linalg.vector_norm(error[:, :3], dim=-1).cpu()
            )
            correction_base_rotation.append(
                torch.linalg.vector_norm(target_valid[:, 3:6], dim=-1).cpu()
            )
            correction_pred_rotation.append(
                torch.linalg.vector_norm(error[:, 3:6], dim=-1).cpu()
            )
            correction_base_gripper.append(
                torch.abs(target_valid[:, 6]).cpu()
            )
            correction_pred_gripper.append(
                torch.abs(error[:, 6]).cpu()
            )
        if zero_mask.any():
            zero_prediction = predicted_residual[zero_mask]
            zero_position.append(
                torch.linalg.vector_norm(zero_prediction[:, :3], dim=-1).cpu()
            )
            zero_rotation.append(
                torch.linalg.vector_norm(zero_prediction[:, 3:6], dim=-1).cpu()
            )
            zero_gripper.append(torch.abs(zero_prediction[:, 6]).cpu())
        if residual_policy.model.gate_enabled:
            gate_probabilities.append(prediction["gate_probability"].cpu())
            gate_labels.append(batch["correction_label"].cpu())

    def concatenate(values):
        return torch.cat(values) if values else torch.empty(0)

    base_position = concatenate(correction_base_position)
    pred_position = concatenate(correction_pred_position)
    base_rotation = concatenate(correction_base_rotation)
    pred_rotation = concatenate(correction_pred_rotation)
    base_gripper = concatenate(correction_base_gripper)
    pred_gripper = concatenate(correction_pred_gripper)
    zero_pos = concatenate(zero_position)
    zero_rot = concatenate(zero_rotation)
    zero_grip = concatenate(zero_gripper)
    def mean(value):
        return float(value.mean()) if len(value) else 0.0

    def quantile(value, q):
        return float(torch.quantile(value.float(), q)) if len(value) else 0.0

    result = dataset.audit_summary()
    result.update(
        {
            "base_checkpoint_sha256": frozen.checkpoint_sha256,
            "correction_base_position_error_mm": 1000.0 * mean(base_position),
            "correction_predicted_position_error_mm": 1000.0
            * mean(pred_position),
            "correction_position_error_ratio": mean(pred_position)
            / max(mean(base_position), 1e-12),
            "correction_base_rotation_error_deg": 180.0
            / np.pi
            * mean(base_rotation),
            "correction_predicted_rotation_error_deg": 180.0
            / np.pi
            * mean(pred_rotation),
            "correction_rotation_error_ratio": mean(pred_rotation)
            / max(mean(base_rotation), 1e-12),
            "correction_base_gripper_error_mm": 1000.0
            * mean(base_gripper),
            "correction_predicted_gripper_error_mm": 1000.0
            * mean(pred_gripper),
            "correction_gripper_error_ratio": mean(pred_gripper)
            / max(mean(base_gripper), 1e-12),
            "zero_position_p95_mm": 1000.0 * quantile(zero_pos, 0.95),
            "zero_rotation_p95_deg": 180.0
            / np.pi
            * quantile(zero_rot, 0.95),
            "zero_gripper_p95_mm": 1000.0 * quantile(zero_grip, 0.95),
        }
    )
    if gate_probabilities:
        probabilities = torch.cat(gate_probabilities)
        labels = torch.cat(gate_labels) > 0.5
        gate_predicted = probabilities >= args.gate_threshold
        result.update(
            {
                "gate_precision": float(
                    torch.count_nonzero(gate_predicted & labels)
                )
                / max(1, int(torch.count_nonzero(gate_predicted))),
                "gate_recall": float(
                    torch.count_nonzero(gate_predicted & labels)
                )
                / max(1, int(torch.count_nonzero(labels))),
                "gate_false_positive_rate": float(
                    torch.count_nonzero(gate_predicted & ~labels)
                )
                / max(1, int(torch.count_nonzero(~labels))),
            }
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("residual_checkpoint", type=Path)
    parser.add_argument("--base-checkpoint", type=Path, default=None)
    parser.add_argument("--sample-mode", choices=("all", "correction_only"), default="all")
    parser.add_argument("--gate-mode", choices=("hard", "soft", "disabled"), default="hard")
    parser.add_argument("--gate-threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = evaluate(args)
    text = json.dumps(result, indent=2)
    print(text)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
