"""Load the exact frozen Diffusion Policy used by a residual model."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import dill
import hydra
import torch


@dataclass
class FrozenBasePolicy:
    policy: torch.nn.Module
    checkpoint_path: str
    checkpoint_sha256: str
    checkpoint_cfg: Any
    state_key: str


def sha256_file(path: str | Path, block_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            block = stream.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def load_frozen_base_policy(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
    prefer_ema: bool = True,
) -> FrozenBasePolicy:
    """Instantiate only the policy from a data-scaling-laws checkpoint."""
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Base checkpoint does not exist: {checkpoint_path}")
    with checkpoint_path.open("rb") as checkpoint_file:
        payload = torch.load(
            checkpoint_file,
            map_location="cpu",
            pickle_module=dill,
        )
    if "cfg" not in payload or "state_dicts" not in payload:
        raise ValueError("Base checkpoint has no cfg/state_dicts payload")
    cfg = payload["cfg"]
    if not hasattr(cfg, "policy"):
        raise ValueError("Base checkpoint config has no policy section")
    policy = hydra.utils.instantiate(cfg.policy)
    state_dicts = payload["state_dicts"]
    if prefer_ema and "ema_model" in state_dicts:
        state_key = "ema_model"
    elif "model" in state_dicts:
        state_key = "model"
    else:
        raise ValueError("Base checkpoint contains neither model nor ema_model")
    policy.load_state_dict(state_dicts[state_key], strict=True)
    policy.requires_grad_(False)
    policy.eval()
    policy.to(device)
    return FrozenBasePolicy(
        policy=policy,
        checkpoint_path=str(checkpoint_path),
        checkpoint_sha256=sha256_file(checkpoint_path),
        checkpoint_cfg=cfg,
        state_key=state_key,
    )
