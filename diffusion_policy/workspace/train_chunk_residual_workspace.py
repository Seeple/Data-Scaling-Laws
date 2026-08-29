"""Training workspace for event-level active-plan chunk residual policies."""

from __future__ import annotations

if __name__ == "__main__":
    import pathlib
    import sys

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)

import copy
import dill
import json
import os
import pathlib
import pickle
import random
from typing import Dict

import hydra
import numpy as np
import torch
import tqdm
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, WeightedRandomSampler

from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.frozen_base_policy_loader import (
    load_frozen_base_policy,
)
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.umi_residual_dataset import (
    UmiResidualDataset,
    UmiResidualDatasetCollection,
)
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.residual.chunk_residual_mlp import ChunkResidualMLP
from diffusion_policy.policy.chunk_residual_policy import ChunkResidualPolicy
from diffusion_policy.workspace.base_workspace import BaseWorkspace


OmegaConf.register_new_resolver("eval", eval, replace=True)


class TrainChunkResidualWorkspace(BaseWorkspace):
    include_keys = (
        "global_step",
        "epoch",
        "base_checkpoint_path",
        "base_checkpoint_sha256",
        "base_checkpoint_state_key",
        "residual_schema_version",
        "residual_action_alignment",
        "residual_valid_action_mask_layout",
        "residual_base_action_horizon",
        "residual_prediction_horizon",
        "initial_residual_checkpoint",
        "initial_residual_state_key",
    )
    exclude_keys = ("base_policy",)

    def __init__(self, cfg: OmegaConf, output_dir=None) -> None:
        super().__init__(cfg, output_dir=output_dir)
        seed = int(cfg.training.seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

        frozen = load_frozen_base_policy(
            cfg.base_policy.ckpt_path,
            device="cpu",
            prefer_ema=bool(cfg.base_policy.prefer_ema),
        )
        self.base_policy = frozen.policy
        self.base_checkpoint_path = frozen.checkpoint_path
        self.base_checkpoint_sha256 = frozen.checkpoint_sha256
        self.base_checkpoint_state_key = frozen.state_key
        self.residual_schema_version = int(
            cfg.task.dataset.expected_schema_version
        )
        self.residual_action_alignment = "unknown"
        self.residual_valid_action_mask_layout = "unknown"
        if not hasattr(self.base_policy, "obs_encoder"):
            raise TypeError("Base checkpoint policy has no obs_encoder")
        if not hasattr(self.base_policy, "normalizer"):
            raise TypeError("Base checkpoint policy has no normalizer")

        expected_horizon = int(cfg.shape_meta.action.horizon)
        expected_action_dim = int(cfg.shape_meta.action.shape[0])
        if int(self.base_policy.action_horizon) != expected_horizon:
            raise ValueError(
                "Base policy action horizon mismatch: expected "
                f"{expected_horizon}, got {self.base_policy.action_horizon}"
            )
        if int(self.base_policy.action_dim) != expected_action_dim:
            raise ValueError(
                "Base policy action dim mismatch: expected "
                f"{expected_action_dim}, got {self.base_policy.action_dim}"
            )

        obs_feature_dim = int(np.prod(self.base_policy.obs_encoder.output_shape()))
        model_cfg = OmegaConf.to_container(cfg.policy.model, resolve=True)
        model_cfg.pop("_target_", None)
        residual_model = ChunkResidualMLP(
            obs_feature_dim=obs_feature_dim,
            **model_cfg,
        )
        policy_cfg = OmegaConf.to_container(cfg.policy, resolve=True)
        policy_cfg.pop("_target_", None)
        policy_cfg.pop("model", None)
        self.model = ChunkResidualPolicy(model=residual_model, **policy_cfg)
        self.residual_base_action_horizon = self.model.base_action_horizon
        self.residual_prediction_horizon = self.model.residual_horizon
        self.ema_model = (
            copy.deepcopy(self.model) if bool(cfg.training.use_ema) else None
        )
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters()
        )
        self.global_step = 0
        self.epoch = 0
        self.initial_residual_checkpoint = ""
        self.initial_residual_state_key = ""

    @staticmethod
    def _load_pickled_metadata(payload: dict, key: str):
        value = payload.get("pickles", {}).get(key)
        return None if value is None else dill.loads(value)

    def _initialize_from_residual_checkpoint(
        self,
        checkpoint_path: str,
        state_key: str,
        expected_alignment: str,
    ) -> None:
        """Warm-start model/EMA weights while keeping a fresh optimizer/epoch."""
        path = pathlib.Path(checkpoint_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"Initial residual checkpoint does not exist: {path}"
            )
        with path.open("rb") as stream:
            payload = torch.load(
                stream, map_location="cpu", pickle_module=dill
            )
        state_dicts = payload.get("state_dicts", {})
        resolved_state_key = str(state_key)
        if resolved_state_key == "auto":
            resolved_state_key = (
                "ema_model" if "ema_model" in state_dicts else "model"
            )
        if resolved_state_key not in state_dicts:
            raise KeyError(
                f"Initial residual checkpoint has no {resolved_state_key!r} state"
            )
        expected_hash = self._load_pickled_metadata(
            payload, "base_checkpoint_sha256"
        )
        if expected_hash and expected_hash != self.base_checkpoint_sha256:
            raise ValueError(
                "Initial residual checkpoint uses a different frozen base "
                f"policy: {expected_hash} != {self.base_checkpoint_sha256}"
            )
        checkpoint_alignment = self._load_pickled_metadata(
            payload, "residual_action_alignment"
        )
        if (
            checkpoint_alignment is not None
            and str(checkpoint_alignment) != str(expected_alignment)
        ):
            raise ValueError(
                "Initial residual checkpoint alignment mismatch: "
                f"{checkpoint_alignment!r} != {expected_alignment!r}"
            )
        self.model.load_state_dict(
            state_dicts[resolved_state_key], strict=True
        )
        if self.ema_model is not None:
            self.ema_model.load_state_dict(
                state_dicts[resolved_state_key], strict=True
            )
        self.initial_residual_checkpoint = str(path)
        self.initial_residual_state_key = resolved_state_key

    def _encode_batch(
        self, batch: Dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            normalized_obs = self.base_policy.normalizer.normalize(batch["obs"])
            obs_feature = self.base_policy.obs_encoder(normalized_obs)
            normalized_base_action = self.base_policy.normalizer[
                "action"
            ].normalize(batch["base_action"])
        return obs_feature, normalized_base_action

    @staticmethod
    def _mean_across_processes(
        accelerator: Accelerator, value: torch.Tensor
    ) -> float:
        gathered = accelerator.gather_for_metrics(value.detach().reshape(1))
        return float(gathered.float().mean().cpu())

    @torch.no_grad()
    def _evaluate(
        self,
        accelerator: Accelerator,
        dataloader: DataLoader,
        policy: ChunkResidualPolicy,
        max_steps: int | None,
    ) -> Dict[str, float]:
        policy.eval()
        encoder_was_training = self.base_policy.obs_encoder.training
        self.base_policy.obs_encoder.eval()
        loss_values = []
        correction_loss_values = []
        zero_loss_values = []
        gate_loss_values = []
        correction_position_errors = []
        correction_rotation_errors = []
        correction_gripper_errors = []
        zero_position_norms = []
        zero_rotation_norms = []
        gate_probabilities = []
        gate_labels = []

        for batch_index, batch in enumerate(dataloader):
            if max_steps is not None and batch_index >= max_steps:
                break
            batch = dict_apply(
                batch, lambda value: value.to(accelerator.device, non_blocking=True)
            )
            obs_feature, normalized_base_action = self._encode_batch(batch)
            output = policy(
                obs_feature,
                normalized_base_action,
                batch["residual_action"],
                batch["valid_action_mask"],
                batch["correction_label"],
                batch.get("residual_supervision_label"),
                batch.get("gate_target"),
            )
            loss_values.append(output["loss"].detach().reshape(1))
            correction_loss_values.append(
                output["correction_loss"].detach().reshape(1)
            )
            zero_loss_values.append(output["zero_loss"].detach().reshape(1))
            gate_loss_values.append(output["gate_loss"].detach().reshape(1))

            prediction = output["predicted_residual"]
            target = batch["residual_action"]
            supervision_label = batch.get(
                "residual_supervision_label", batch["correction_label"]
            )
            correction_mask = (
                batch["valid_action_mask"].bool()
                & (supervision_label > 0.5)[:, None]
            )
            zero_mask = (
                torch.ones_like(batch["valid_action_mask"], dtype=torch.bool)
                & (supervision_label <= 0.5)[:, None]
            )
            if correction_mask.any():
                delta = prediction[correction_mask] - target[correction_mask]
                correction_position_errors.append(
                    torch.linalg.vector_norm(delta[:, :3], dim=-1)
                )
                correction_rotation_errors.append(
                    torch.linalg.vector_norm(delta[:, 3:6], dim=-1)
                )
                correction_gripper_errors.append(torch.abs(delta[:, 6]))
            if zero_mask.any():
                zero_prediction = prediction[zero_mask]
                zero_position_norms.append(
                    torch.linalg.vector_norm(zero_prediction[:, :3], dim=-1)
                )
                zero_rotation_norms.append(
                    torch.linalg.vector_norm(zero_prediction[:, 3:6], dim=-1)
                )
            if policy.model.gate_enabled:
                gate_probabilities.append(
                    output["gate_probability"].detach()
                )
                gate_labels.append(
                    batch.get("gate_target", supervision_label).detach()
                )

        def gathered_mean(values, default=0.0):
            if not values:
                return float(default)
            local = torch.cat(values)
            gathered = accelerator.gather_for_metrics(local)
            return float(gathered.float().mean().cpu())

        def gathered_quantile(values, quantile, default=0.0):
            if not values:
                return float(default)
            local = torch.cat(values)
            gathered = accelerator.gather_for_metrics(local)
            return float(torch.quantile(gathered.float().cpu(), quantile))

        metrics = {
            "val_loss": gathered_mean(loss_values),
            "val_correction_loss": gathered_mean(correction_loss_values),
            "val_zero_loss": gathered_mean(zero_loss_values),
            "val_gate_loss": gathered_mean(gate_loss_values),
            "val_correction_position_error_mm": 1000.0
            * gathered_mean(correction_position_errors),
            "val_correction_rotation_error_deg": 180.0
            / np.pi
            * gathered_mean(correction_rotation_errors),
            "val_correction_gripper_error_mm": 1000.0
            * gathered_mean(correction_gripper_errors),
            "val_zero_position_p95_mm": 1000.0
            * gathered_quantile(zero_position_norms, 0.95),
            "val_zero_rotation_p95_deg": 180.0
            / np.pi
            * gathered_quantile(zero_rotation_norms, 0.95),
        }
        if gate_probabilities:
            probabilities = accelerator.gather_for_metrics(
                torch.cat(gate_probabilities)
            ).float().cpu()
            labels = accelerator.gather_for_metrics(torch.cat(gate_labels)).cpu()
            predicted = probabilities >= float(policy.gate_threshold)
            positive = labels > 0.5
            negative = ~positive
            true_positive = int(torch.count_nonzero(predicted & positive))
            predicted_positive = int(torch.count_nonzero(predicted))
            positive_count = int(torch.count_nonzero(positive))
            false_positive = int(torch.count_nonzero(predicted & negative))
            negative_count = int(torch.count_nonzero(negative))
            metrics.update(
                {
                    "val_gate_precision": true_positive
                    / max(1, predicted_positive),
                    "val_gate_recall": true_positive / max(1, positive_count),
                    "val_gate_false_positive_rate": false_positive
                    / max(1, negative_count),
                }
            )
        metrics["val_composite_objective"] = (
            metrics["val_correction_loss"]
            + metrics["val_zero_loss"]
            + metrics["val_gate_loss"]
        )
        self.base_policy.obs_encoder.train(encoder_was_training)
        return metrics

    def run(self) -> None:
        cfg = copy.deepcopy(self.cfg)
        use_wandb = bool(cfg.logging.get("use_wandb", True))
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
        accelerator = Accelerator(
            gradient_accumulation_steps=int(
                cfg.training.gradient_accumulate_every
            ),
            log_with="wandb" if use_wandb else None,
            kwargs_handlers=[ddp_kwargs],
        )
        if use_wandb:
            wandb_cfg = OmegaConf.to_container(cfg.logging, resolve=True)
            project = wandb_cfg.pop("project")
            wandb_cfg.pop("use_wandb", None)
            accelerator.init_trackers(
                project_name=project,
                config=OmegaConf.to_container(cfg, resolve=True),
                init_kwargs={"wandb": wandb_cfg},
            )

        init_checkpoint = cfg.training.get(
            "init_residual_checkpoint", None
        )
        if bool(cfg.training.resume) and init_checkpoint not in {
            None,
            "",
            "null",
        }:
            raise ValueError(
                "training.resume and init_residual_checkpoint are mutually "
                "exclusive: resume restores optimizer/epoch state, whereas "
                "a DAgger warm start intentionally creates a fresh run"
            )
        if bool(cfg.training.resume):
            checkpoint = self.get_checkpoint_path()
            if checkpoint.is_file():
                self.load_checkpoint(path=checkpoint)

        dataset = hydra.utils.instantiate(cfg.task.dataset)
        if not isinstance(
            dataset, (UmiResidualDataset, UmiResidualDatasetCollection)
        ):
            raise TypeError(
                "Residual workspace requires a residual dataset or collection"
            )
        self.residual_action_alignment = dataset.action_alignment
        self.residual_valid_action_mask_layout = (
            dataset.valid_action_mask_layout
        )
        if dataset.action_horizon != self.model.base_action_horizon:
            raise ValueError(
                "Residual dataset/model base horizon mismatch: dataset="
                f"{dataset.action_horizon}, model="
                f"{self.model.base_action_horizon}"
            )
        if dataset.action_horizon != self.model.residual_horizon:
            raise ValueError(
                "Residual dataset/model prediction horizon mismatch: dataset="
                f"{dataset.action_horizon}, model="
                f"{self.model.residual_horizon}"
            )
        if (
            dataset.recorded_base_checkpoint_sha256
            and dataset.recorded_base_checkpoint_sha256
            != self.base_checkpoint_sha256
        ):
            raise ValueError(
                "Residual dataset was recorded with a different base "
                "checkpoint: dataset SHA256="
                f"{dataset.recorded_base_checkpoint_sha256}, training "
                f"SHA256={self.base_checkpoint_sha256}"
            )
        if (
            accelerator.is_main_process
            and not dataset.recorded_base_checkpoint_sha256
        ):
            print(
                "WARNING: residual dataset predates base-checkpoint hashing; "
                "verify its recorded base checkpoint manually. Newly "
                "converted datasets enforce this automatically."
            )
        if init_checkpoint not in {None, "", "null"}:
            self._initialize_from_residual_checkpoint(
                str(init_checkpoint),
                str(cfg.training.get("init_residual_state_key", "auto")),
                dataset.action_alignment,
            )
        validation_dataset = dataset.get_validation_dataset()
        if accelerator.is_main_process:
            print(json.dumps(dataset.audit_summary(), indent=2))
            print(json.dumps(validation_dataset.audit_summary(), indent=2))

        sampling_weights = dataset.get_sampling_weights()
        train_loader_cfg = OmegaConf.to_container(cfg.dataloader, resolve=True)
        if sampling_weights is not None:
            train_loader_cfg["shuffle"] = False
            sampler = WeightedRandomSampler(
                sampling_weights,
                num_samples=len(dataset),
                replacement=True,
                generator=torch.Generator().manual_seed(int(cfg.training.seed)),
            )
            train_dataloader = DataLoader(
                dataset, sampler=sampler, **train_loader_cfg
            )
        else:
            train_dataloader = DataLoader(dataset, **train_loader_cfg)
        val_dataloader = DataLoader(
            validation_dataset,
            **OmegaConf.to_container(cfg.val_dataloader, resolve=True),
        )

        normalizer_path = os.path.join(
            self.output_dir, "residual_normalizer.pkl"
        )
        use_initial_normalizer = bool(
            self.initial_residual_checkpoint
            and cfg.training.get("init_use_checkpoint_normalizer", True)
        )
        if accelerator.is_main_process:
            residual_normalizer = (
                copy.deepcopy(self.model.residual_normalizer)
                if use_initial_normalizer
                else dataset.get_residual_normalizer()
            )
            with open(normalizer_path, "wb") as normalizer_file:
                pickle.dump(residual_normalizer, normalizer_file)
        accelerator.wait_for_everyone()
        with open(normalizer_path, "rb") as normalizer_file:
            residual_normalizer = pickle.load(normalizer_file)
        self.model.set_residual_normalizer(residual_normalizer)
        if self.ema_model is not None:
            self.ema_model.set_residual_normalizer(residual_normalizer)

        total_steps = (
            len(train_dataloader) * int(cfg.training.num_epochs)
        ) // int(cfg.training.gradient_accumulate_every)
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=int(cfg.training.lr_warmup_steps),
            num_training_steps=max(1, total_steps),
            last_epoch=self.global_step - 1,
        )
        prepared = accelerator.prepare(
            train_dataloader,
            val_dataloader,
            self.model,
            self.optimizer,
            lr_scheduler,
        )
        (
            train_dataloader,
            val_dataloader,
            self.model,
            self.optimizer,
            lr_scheduler,
        ) = prepared
        self.base_policy.to(accelerator.device)
        self.base_policy.requires_grad_(False)
        encoder_mode = str(cfg.base_policy.encoder_mode)
        if encoder_mode == "eval":
            self.base_policy.obs_encoder.eval()
        elif encoder_mode == "train_transforms":
            self.base_policy.obs_encoder.train()
        else:
            raise ValueError(
                "base_policy.encoder_mode must be eval or train_transforms"
            )
        if self.ema_model is not None:
            self.ema_model.to(accelerator.device)
            ema: EMAModel | None = hydra.utils.instantiate(
                cfg.ema, model=self.ema_model
            )
        else:
            ema = None

        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, "checkpoints"),
            **cfg.checkpoint.topk,
        )
        log_path = os.path.join(self.output_dir, "logs.json.txt")
        max_train_steps = cfg.training.get("max_train_steps", None)
        max_val_steps = cfg.training.get("max_val_steps", None)
        if bool(cfg.training.debug):
            cfg.training.num_epochs = 2
            max_train_steps = 3
            max_val_steps = 3
            cfg.training.val_every = 1
            cfg.training.checkpoint_every = 1

        with JsonLogger(log_path) as json_logger:
            while self.epoch < int(cfg.training.num_epochs):
                self.model.train()
                train_losses = []
                progress = tqdm.tqdm(
                    train_dataloader,
                    desc=f"Residual epoch {self.epoch}",
                    disable=not accelerator.is_local_main_process,
                    mininterval=float(cfg.training.tqdm_interval_sec),
                )
                for batch_index, batch in enumerate(progress):
                    if max_train_steps is not None and batch_index >= int(
                        max_train_steps
                    ):
                        break
                    batch = dict_apply(
                        batch,
                        lambda value: value.to(
                            accelerator.device, non_blocking=True
                        ),
                    )
                    with accelerator.accumulate(self.model):
                        with accelerator.autocast():
                            obs_feature, normalized_base_action = self._encode_batch(
                                batch
                            )
                            output = self.model(
                                obs_feature,
                                normalized_base_action,
                                batch["residual_action"],
                                batch["valid_action_mask"],
                                batch["correction_label"],
                                batch.get("residual_supervision_label"),
                                batch.get("gate_target"),
                            )
                            loss = output["loss"]
                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(
                                self.model.parameters(),
                                float(cfg.training.max_grad_norm),
                            )
                        self.optimizer.step()
                        lr_scheduler.step()
                        self.optimizer.zero_grad(set_to_none=True)
                        if ema is not None and accelerator.sync_gradients:
                            ema.step(accelerator.unwrap_model(self.model))
                    train_losses.append(loss.detach())
                    progress.set_postfix(loss=float(loss.detach().cpu()))
                    self.global_step += 1

                train_loss = self._mean_across_processes(
                    accelerator,
                    torch.stack(train_losses).mean()
                    if train_losses
                    else torch.tensor(0.0, device=accelerator.device),
                )
                step_log = {
                    "epoch": self.epoch,
                    "global_step": self.global_step,
                    "train_loss": train_loss,
                    "lr": float(lr_scheduler.get_last_lr()[0]),
                }
                if (
                    self.epoch % int(cfg.training.val_every) == 0
                    and len(validation_dataset) > 0
                ):
                    eval_policy = (
                        self.ema_model
                        if self.ema_model is not None
                        else accelerator.unwrap_model(self.model)
                    )
                    step_log.update(
                        self._evaluate(
                            accelerator,
                            val_dataloader,
                            eval_policy,
                            None
                            if max_val_steps is None
                            else int(max_val_steps),
                        )
                    )
                else:
                    step_log["val_composite_objective"] = train_loss

                accelerator.wait_for_everyone()
                should_checkpoint = (
                    self.epoch % int(cfg.training.checkpoint_every) == 0
                    or self.epoch == int(cfg.training.num_epochs) - 1
                )
                if should_checkpoint and accelerator.is_main_process:
                    wrapped_model = self.model
                    self.model = accelerator.unwrap_model(self.model)
                    if bool(cfg.checkpoint.save_last_ckpt):
                        self.save_checkpoint()
                    topk_path = topk_manager.get_ckpt_path(step_log)
                    if topk_path is not None:
                        self.save_checkpoint(path=topk_path)
                    self.model = wrapped_model

                if use_wandb:
                    accelerator.log(step_log, step=self.global_step)
                if accelerator.is_main_process:
                    json_logger.log(step_log)
                self.epoch += 1

        accelerator.end_training()


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name="train_chunk_residual_mlp_workspace",
)
def main(cfg):
    workspace = TrainChunkResidualWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
