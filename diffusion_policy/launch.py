"""Portable launcher for the ARISE training workflows; importing it loads no model."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import importlib.util
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
TASKS = ("hang_chinese_knot", "manage_table", "cable_routing")
METHODS = ("pretrain", "dagger", "sirius", "residual", "step-residual")
CONFIGS = {
    "pretrain": "train_diffusion_unet_timm_umi_workspace",
    "dagger": "train_diffusion_unet_timm_umi_dagger_workspace",
    "sirius": "train_diffusion_unet_timm_umi_dagger_workspace",
    "residual": "train_chunk_residual_mlp_workspace",
    "step-residual": "train_step_active_plan_residual_mlp_workspace",
}
PATH_OPTIONS = {"dataset", "teleop_dataset", "checkpoint", "base_checkpoint",
                "init_residual", "run_dir", "output_root", "cache_dir", "accelerate_config"}
LIST_PATH_OPTIONS = {"online_datasets", "replay_datasets"}


@dataclass
class LaunchPlan:
    command: list[str]
    cwd: Path
    environment: dict[str, str]
    config: dict
    run_dir: Path
    resume: bool


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, help="Local YAML; relative paths use its directory")
    p.add_argument("--task", choices=TASKS)
    p.add_argument("--method", choices=METHODS)
    p.add_argument("--dataset", help="Demonstrations for pretrain, event Zarr for residual")
    p.add_argument("--teleop-dataset")
    p.add_argument("--online-dataset", dest="online_datasets", action="append")
    p.add_argument("--checkpoint", help="DP weights for DAgger/Sirius finetuning")
    p.add_argument("--base-checkpoint", help="Frozen base DP for residual training")
    p.add_argument("--replay-dataset", dest="replay_datasets", action="append")
    p.add_argument("--init-residual", help="Residual warm start; does not resume optimizer")
    p.add_argument("--variant", choices=("correction_only", "mixed_no_gate", "mixed_gate"), default="mixed_no_gate")
    p.add_argument("--action-alignment", choices=("recorded_chunk", "active_suffix_left_aligned"), default="recorded_chunk")
    p.add_argument("--residual-horizon", type=int, choices=(1, 5), default=5)
    p.add_argument("--gpus", help="Visible CUDA IDs, e.g. 0,1")
    p.add_argument("--num-processes", type=int)
    p.add_argument("--port", type=int, default=29501)
    p.add_argument("--precision", choices=("no", "fp16", "bf16"), default="bf16")
    p.add_argument("--accelerate-config")
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--num-workers", type=int)
    p.add_argument("--val-num-workers", type=int, default=4)
    p.add_argument("--lr", type=float)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=None,
                   help="Download/use vision initialization, separate from finetune weights")
    p.add_argument("--cache-dir", help="Optional LMDB cache; default loads Zarr in memory")
    p.add_argument("--run-dir")
    p.add_argument("--output-root")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="Hydra override, repeatable; applied after workflow defaults")
    p.add_argument("--dry-run", action="store_true", help="Compose and print; no files, devices or model loads")
    p.add_argument("--print-config", action="store_true", help="Also print composed workspace YAML")
    p.add_argument("--check", action="store_true", help="Check input files and training dependencies, then exit")
    return p


def parse_args(argv=None):
    p = parser()
    argv = sys.argv[1:] if argv is None else list(argv)
    preliminary, _ = p.parse_known_args(argv)
    if preliminary.config:
        path = preliminary.config.expanduser().resolve()
        values = yaml.safe_load(path.read_text()) or {}
        if not isinstance(values, dict):
            p.error("Local YAML must be a mapping")
        allowed = {action.dest for action in p._actions} - {"help", "config", "dry_run", "check", "print_config"}
        unknown = set(values) - allowed
        if unknown:
            p.error(f"Unknown local settings: {sorted(unknown)}")
        for key in PATH_OPTIONS & values.keys():
            if values[key] is not None:
                values[key] = str(resolve_path(values[key], path.parent))
        for key in LIST_PATH_OPTIONS & values.keys():
            if not isinstance(values[key], list):
                p.error(f"{key} must be a YAML list")
            values[key] = [str(resolve_path(value, path.parent)) for value in values[key]]
        p.set_defaults(**values)
    options = p.parse_args(argv)
    # Repeated CLI lists replace the YAML list instead of silently appending to it.
    explicit = parser().parse_args(argv)
    for key, flag in (("online_datasets", "--online-dataset"), ("replay_datasets", "--replay-dataset"),
                      ("overrides", "--set")):
        if any(arg == flag or arg.startswith(flag + "=") for arg in argv):
            setattr(options, key, getattr(explicit, key))
    return options


def resolve_path(value, base=None):
    path = Path(os.path.expandvars(str(value))).expanduser()
    return ((base or Path.cwd()) / path).resolve()


def literal(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def build_plan(options, root=ROOT):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    o = options
    root = Path(root).resolve()
    if o.task not in TASKS or o.method not in METHODS:
        raise ValueError("Select --task and --method (or put them in --config)")
    for key in ("epochs", "batch_size", "num_processes"):
        value = getattr(o, key)
        if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value <= 0):
            raise ValueError(f"{key} must be a positive integer")
    for key in ("num_workers", "val_num_workers", "seed"):
        value = getattr(o, key)
        if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
            raise ValueError(f"{key} must be a nonnegative integer")
    if o.lr is not None and (not isinstance(o.lr, (int, float)) or not 0 < o.lr < float("inf")):
        raise ValueError("lr must be positive and finite")
    if o.precision not in ("no", "fp16", "bf16") or not isinstance(o.port, int) or not 0 < o.port < 65536:
        raise ValueError("Invalid precision or port")
    if not isinstance(o.overrides, list) or not all(isinstance(value, str) for value in o.overrides):
        raise ValueError("overrides must be a list of Hydra override strings")
    for key in ("wandb", "resume"):
        if not isinstance(getattr(o, key), bool):
            raise ValueError(f"{key} must be a boolean")
    if o.pretrained is not None and not isinstance(o.pretrained, bool):
        raise ValueError("pretrained must be a boolean")

    is_residual = o.method in ("residual", "step-residual")
    is_mixed = o.method in ("dagger", "sirius")
    if is_residual:
        if not o.dataset or not o.base_checkpoint:
            raise ValueError("Residual training requires --dataset and --base-checkpoint")
        if o.checkpoint or o.teleop_dataset or o.online_datasets:
            raise ValueError("Use --base-checkpoint/--init-residual for residual training")
    elif is_mixed:
        if not o.teleop_dataset or not o.online_datasets or not o.checkpoint:
            raise ValueError("DAgger/Sirius requires --teleop-dataset, --online-dataset and --checkpoint")
        if o.dataset:
            raise ValueError("Use --teleop-dataset and --online-dataset for DAgger/Sirius")
    elif not o.dataset:
        raise ValueError("Pretraining requires --dataset")
    if not is_residual and (o.base_checkpoint or o.init_residual or o.replay_datasets):
        raise ValueError("Residual inputs require a residual method")
    if o.method == "pretrain" and (o.checkpoint or o.teleop_dataset or o.online_datasets):
        raise ValueError("Use dagger/sirius for checkpoint finetuning")
    if o.variant not in ("correction_only", "mixed_no_gate", "mixed_gate"):
        raise ValueError("Unknown residual variant")
    if o.action_alignment not in ("recorded_chunk", "active_suffix_left_aligned"):
        raise ValueError("Unknown action alignment")
    if o.residual_horizon not in (1, 5):
        raise ValueError("Residual horizon must be 1 or 5")
    stamp = datetime.now().strftime("%Y.%m.%d-%H.%M.%S.%f")
    run_dir = resolve_path(o.run_dir) if o.run_dir else resolve_path(o.output_root or root / "outputs") / o.task / o.method / stamp
    if o.resume and (not o.run_dir or o.init_residual):
        raise ValueError("Resume requires --run-dir and cannot also warm-start residual weights")

    overrides = []
    def put(key, value):
        overrides.append(f"{key}={literal(value)}")
    put("hydra.run.dir", str(run_dir))
    put("task.name", o.task)
    put("logging.name", f"{stamp}_{o.task}_{o.method}")
    put("logging.use_wandb", o.wandb)
    put("training.seed", o.seed)
    put("training.resume", o.resume)
    task_pretrain = {"hang_chinese_knot": (200, 32, 8), "manage_table": (500, 64, 16), "cable_routing": (200, 64, 8)}
    epochs, batch, workers = task_pretrain[o.task] if o.method == "pretrain" else ((200, 32, 4) if is_residual else (400, 64, 8 if o.task == "cable_routing" else 16))
    workers = o.num_workers if o.num_workers is not None else workers
    put("training.num_epochs", o.epochs or epochs)
    put("dataloader.batch_size", o.batch_size or batch)
    put("dataloader.num_workers", workers)
    put("dataloader.persistent_workers", workers > 0)
    put("val_dataloader.num_workers", o.val_num_workers)
    put("val_dataloader.persistent_workers", o.val_num_workers > 0)
    put("optimizer.lr", o.lr or (3e-4 if is_residual or o.method == "pretrain" else 5e-5))
    if not is_residual:
        put("policy.obs_encoder.model_name", "vit_large_patch14_dinov2.lvd142m")
        put("policy.obs_encoder.pretrained", o.pretrained if o.pretrained is not None else o.method == "pretrain")
        put("task.dataset.cache_dir", str(resolve_path(o.cache_dir)) if o.cache_dir else None)
        put("task.dataset.val_ratio", 0.1)
        put("task.dataset_frequeny", 15 if o.method == "sirius" else 0)
        put("task.obs_down_sample_steps", 3)
        put("task.action_down_sample_steps", 3)
        put("training.rollout_every", 1000)
        put("multi_run.run_dir", str(run_dir))
        put("multi_run.wandb_name_base", f"{stamp}_{o.task}_{o.method}")
    if o.method == "pretrain":
        put("task.dataset_path", str(resolve_path(o.dataset)))
    if is_mixed:
        online = [str(resolve_path(value)) for value in o.online_datasets]
        put("task.teleop_dataset_path", str(resolve_path(o.teleop_dataset)))
        put("task.hitl_dataset_path", online[0])
        put("task.online_dataset_paths", online)
        put("task.dataset.rlpd_ratio", [0.5] + [0.5 / len(online)] * len(online))
        put("training.finetune_from", None if o.resume else str(resolve_path(o.checkpoint)))
        put("training.reset_steps_on_finetune", True)
        put("training.lr_warmup_steps", 500)
        put("training.freeze_encoder_on_finetune", True)
        put("training.freeze_encoder_epochs", 400 if o.task == "cable_routing" and o.method == "dagger" else 3)
    if o.method == "sirius":
        put("task.dataset.sirius_weighting_enabled", True)
        put("task.dataset.lowdim_obs_normalizer_source", "teleop")
        put("checkpoint.topk.k", 2)
    if is_residual:
        put("base_policy.ckpt_path", str(resolve_path(o.base_checkpoint)))
        put("task.residual_dataset_path", str(resolve_path(o.dataset)))
        put("task.dataset.sample_mode", "correction_only" if o.variant == "correction_only" else "all")
        put("task.dataset.correction_fraction", None if o.variant == "correction_only" else 0.5)
        put("policy.zero_loss_weight", 0.0 if o.variant == "correction_only" else 1.0)
        put("policy.model.gate_enabled", o.variant == "mixed_gate")
        if o.method == "residual":
            put("task.dataset.expected_action_alignment", o.action_alignment)
        else:
            put("task.residual_horizon", o.residual_horizon)
        if o.replay_datasets:
            paths = [str(resolve_path(value)) for value in o.replay_datasets] + [str(resolve_path(o.dataset))]
            put("task.dataset._target_", "diffusion_policy.dataset.umi_residual_dataset.UmiResidualDatasetCollection")
            put("+task.dataset.dataset_paths", paths)
            put("+task.dataset.dataset_mixture_weights", [1 / len(paths)] * len(paths))
            put("+task.dataset.val_episode_indices_by_dataset", None)
        if o.init_residual:
            put("training.init_residual_checkpoint", str(resolve_path(o.init_residual)))
    if any(value.split("=", 1)[0].lstrip("+") in ("hydra", "hydra.run", "hydra.run.dir", "training", "training.resume") for value in o.overrides):
        raise ValueError("Use --run-dir/--resume for launch-owned run settings")
    overrides.extend(o.overrides)
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", lambda expression: eval(expression, {"__builtins__": {}}, {}))
    with initialize_config_dir(config_dir=str(root / "diffusion_policy/config"), version_base="1.3"):
        config = compose(config_name=CONFIGS[o.method], overrides=overrides)
        config = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    # Validate the effective settings, including raw Hydra overrides.
    if o.method == "sirius":
        dataset = config["task"]["dataset"]
        incompatible = ("hitl_only_tag", "hitl_require_full_action_tag", "hitl_action_mask", "hitl_contiguous_action_mask", "hitl_invalid_tail_padding", "hitl_min_valid_steps_enabled", "hitl_skip_rising_edge", "hitl_treat_segments_as_episodes", "hitl_disable_downsample")
        if not dataset["sirius_weighting_enabled"] or any(dataset[key] for key in incompatible):
            raise ValueError("Sirius requires complete online rollouts without correction-only filters or masks")
    for name in ("dataloader", "val_dataloader"):
        loader = config[name]
        if loader["num_workers"] == 0 and loader["persistent_workers"]:
            raise ValueError(f"{name}: persistent_workers requires num_workers > 0")
    gpus = [value.strip() for value in str(o.gpus).split(",")] if o.gpus is not None else []
    if gpus and (any(not item.strip().isdigit() for item in gpus) or len(set(gpus)) != len(gpus)):
        raise ValueError("--gpus must be distinct comma-separated CUDA IDs")
    processes = o.num_processes or len(gpus) or 1
    if gpus and processes > len(gpus):
        raise ValueError("num_processes exceeds the number of visible GPUs")
    accelerator = resolve_path(o.accelerate_config) if o.accelerate_config else root / "train_scripts/configs/accelerate.yaml"
    if not accelerator.is_file():
        raise ValueError(f"Accelerate config not found: {accelerator}")
    command = [sys.executable, "-m", "accelerate.commands.launch", "--config_file", str(accelerator),
               "--num_processes", str(processes), "--num_machines", "1", "--main_process_port", str(o.port),
               "--mixed_precision", o.precision]
    if processes > 1:
        command.append("--multi_gpu")
    command.extend([str(root / "train.py"), "--config-name=" + CONFIGS[o.method], *overrides])
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root) + (os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else "")
    environment.update(HYDRA_FULL_ERROR="1", PYTHONFAULTHANDLER="1")
    if gpus:
        environment["CUDA_VISIBLE_DEVICES"] = ",".join(item.strip() for item in gpus)
    return LaunchPlan(command, root, environment, config, run_dir, o.resume)


def check_plan(plan):
    """Validate effective input paths and import availability without importing models."""
    config = plan.config
    dataset = config["task"]["dataset"]
    paths = []
    if "dataset_paths" in dataset:
        paths.extend(dataset["dataset_paths"])
    elif "teleop_dataset_path" in dataset:
        paths.append(dataset["teleop_dataset_path"])
        paths.extend(dataset.get("online_dataset_paths") or [dataset["hitl_dataset_path"]])
    else:
        paths.append(dataset["dataset_path"])
    for value in (config.get("base_policy", {}).get("ckpt_path"),
                  config["training"].get("finetune_from"), config["training"].get("init_residual_checkpoint")):
        if value:
            paths.append(value)
    if plan.resume:
        paths.append(str(plan.run_dir / "checkpoints/latest.ckpt"))
    errors = [f"Input not found: {value}" for value in paths if not Path(value).is_file()]
    modules = ["torch", "torchvision", "timm", "peft", "diffusers", "accelerate", "dill", "wandb", "zarr", "numcodecs", "imagecodecs", "einops", "scipy", "threadpoolctl", "filelock"]
    if dataset.get("cache_dir"):
        modules.append("lmdb")
    errors.extend(f"Missing dependency: {name}" for name in modules if importlib.util.find_spec(name) is None)
    if errors:
        raise ValueError("\n".join(errors))


def run_plan(plan):
    """Stream logs and preserve the actual training process's exit status."""
    plan.run_dir.mkdir(parents=True, exist_ok=plan.resume)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    metadata = {"command": plan.command, "cwd": str(plan.cwd),
                "CUDA_VISIBLE_DEVICES": plan.environment.get("CUDA_VISIBLE_DEVICES"),
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=plan.cwd, text=True).strip(),
                "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=plan.cwd, text=True))}
    (plan.run_dir / f"launch-{stamp}.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (plan.run_dir / f"resolved-{stamp}.yaml").write_text(yaml.safe_dump(plan.config, sort_keys=False))
    with (plan.run_dir / "train.log").open("a") as log:
        process = subprocess.Popen(plan.command, cwd=plan.cwd, env=plan.environment,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1, start_new_session=True)
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            return process.wait()
        except KeyboardInterrupt:
            os.killpg(process.pid, signal.SIGINT)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            return 130


def main(argv=None):
    from hydra.errors import HydraException
    from omegaconf.errors import OmegaConfBaseException

    try:
        options = parse_args(argv)
        plan = build_plan(options)
        print("Working directory:", plan.cwd)
        print("Run directory:", plan.run_dir)
        print("Command:", shlex.join(plan.command))
        if options.print_config:
            print(yaml.safe_dump(plan.config, sort_keys=False))
        if options.check:
            check_plan(plan)
            print("Input files and training imports are available; no model was loaded.")
            return 0
        if options.dry_run:
            return 0
        check_plan(plan)
        return run_plan(plan)
    except (ValueError, OSError, HydraException, OmegaConfBaseException) as error:
        print(f"Launch error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
