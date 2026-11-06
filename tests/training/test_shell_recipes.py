"""Run public shell recipes from another cwd; checks never load model files."""
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest
import yaml

from diffusion_policy.launch import ROOT

RECIPES = {
    "base-model.sh": "train_diffusion_unet_timm_umi_workspace",
    "hg-dagger.sh": "train_diffusion_unet_timm_umi_dagger_workspace",
    "sirius.sh": "train_diffusion_unet_timm_umi_dagger_workspace",
    "residual-policy.sh": "train_chunk_residual_mlp_workspace",
    "residual-policy-round2.sh": "train_chunk_residual_mlp_workspace",
    "residual-policy-step.sh": "train_step_active_plan_residual_mlp_workspace",
}


def invoke(recipe, cwd, *args, settings=None):
    environment = os.environ.copy()
    # Keep the test independent of a developer's training environment settings.
    for name in ("TASK_NAME", "DATA_ROOT", "TELEOP_DATASET_PATH", "HITL_DATASET_PATH",
                 "FINETUNE_CKPT", "BASE_POLICY_CKPT", "RESIDUAL_DATASET_PATH",
                 "REPLAY_RESIDUAL_DATASET_PATH", "INIT_RESIDUAL_CKPT", "RLPD_RATIO",
                 "NUM_EPOCHS", "BATCH_SIZE", "NUM_WORKERS", "VAL_NUM_WORKERS",
                 "GPU_LIST", "NUM_PROCESSES", "MAIN_PROCESS_PORT", "MODEL_PRETRAINED",
                 "RUN_DIR", "CACHE_DIR", "ACCELERATE_CONFIG_FILE", "VARIANT",
                 "RESIDUAL_HORIZON", "ACTION_ALIGNMENT", "CORRECTION_FRACTION"):
        environment.pop(name, None)
    environment.update(PYTHON=sys.executable, USE_WANDB="false")
    environment.update(settings or {})
    return subprocess.run(
        ["bash", str(ROOT / "train_scripts/configs" / recipe), *args],
        cwd=cwd, env=environment, capture_output=True, text=True, timeout=30,
    )


def composed(result):
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    command_index = next(index for index, line in enumerate(lines) if line.startswith("Command: "))
    command = shlex.split(lines[command_index].removeprefix("Command: "))
    config = yaml.safe_load("\n".join(lines[command_index + 1:]))
    return command, config


@pytest.mark.parametrize("recipe", RECIPES)
def test_recipe_finds_training_entry_and_configs_from_another_cwd(tmp_path, recipe):
    path = ROOT / "train_scripts/configs" / recipe
    syntax = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
    assert syntax.returncode == 0, syntax.stderr
    result = invoke(recipe, tmp_path, "--dry-run", "--print-config")
    command, config = composed(result)
    assert f"Working directory: {ROOT}" in result.stdout
    assert str(ROOT / "train.py") in command
    assert Path(command[command.index("--config_file") + 1]).is_file()
    assert (ROOT / "diffusion_policy/config" / (RECIPES[recipe] + ".yaml")).is_file()
    assert config["task"]["shape_meta"]["action"]["horizon"] == 16
    # Check the exact argv at the actual training entry, without running a job.
    train_index = command.index(str(ROOT / "train.py"))
    actual = subprocess.run(
        [sys.executable, *command[train_index:], "--cfg", "job", "--resolve"],
        cwd=ROOT, capture_output=True, text=True, timeout=30,
    )
    assert actual.returncode == 0, actual.stderr
    assert yaml.safe_load(actual.stdout) == config


@pytest.mark.parametrize("recipe", RECIPES)
def test_check_finds_relative_input_paths_with_spaces(tmp_path, recipe):
    # Stubs exercise lookup only; check mode must not deserialize any of them.
    data = tmp_path / "assets with spaces"
    for relative in ("dataset/teleop.zarr.zip", "dataset/hitl.zarr.zip",
                     "dataset/residual-round1.zarr.zip", "dataset/residual-round2.zarr.zip",
                     "dataset/step-residual-h5.zarr.zip", "ckpts/base.ckpt",
                     "ckpts/residual-round1.ckpt"):
        path = data / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    result = invoke(recipe, tmp_path, "--check", settings={"DATA_ROOT": os.path.relpath(data, ROOT)})
    assert result.returncode == 0, result.stderr
    assert "no model was loaded" in result.stdout


def test_missing_inputs_fail_without_starting_training(tmp_path):
    data = tmp_path / "not-present"
    result = invoke("base-model.sh", tmp_path, "--check", settings={"DATA_ROOT": os.path.relpath(data, ROOT)})
    assert result.returncode == 2
    assert f"Input not found: {data / 'dataset/teleop.zarr.zip'}" in result.stderr
    assert not data.exists()


@pytest.mark.parametrize("recipe", ["hg-dagger.sh", "sirius.sh"])
def test_explicit_online_buffers_replace_recipe_default(tmp_path, recipe):
    _, config = composed(invoke(
        recipe, tmp_path, "--online-dataset", "data/one.zip", "--online-dataset=data/two.zip",
        "--dry-run", "--print-config",
    ))
    assert config["task"]["dataset"]["online_dataset_paths"] == [str(ROOT / "data/one.zip"), str(ROOT / "data/two.zip")]
    assert config["task"]["dataset"]["rlpd_ratio"] == [0.5, 0.25, 0.25]
    if recipe == "hg-dagger.sh":
        assert config["task"]["dataset"]["hitl_only_tag"]
        assert config["task"]["dataset"]["hitl_action_mask"]
        assert config["task"]["dataset"]["hitl_downsample_multiplier"] == 2
    else:
        assert config["task"]["dataset"]["sirius_weighting_enabled"]
        assert config["task"]["dataset"]["hitl_action_mask"] is False


def test_round_two_replay_and_checkpoint_paths_survive_hydra_delimiters(tmp_path):
    checkpoint = "data/ckpts/epoch=0010 score=0.1.ckpt"
    _, config = composed(invoke(
        "residual-policy-round2.sh", tmp_path, "--init-residual", checkpoint,
        "--replay-dataset", "data/replay one.zip", "--replay-dataset", "data/replay two.zip",
        "--dry-run", "--print-config",
    ))
    assert config["training"]["init_residual_checkpoint"] == str(ROOT / checkpoint)
    assert config["training"]["init_ema_min_decay"] == 0.99
    assert config["task"]["dataset"]["dataset_paths"] == [str(ROOT / "data/replay one.zip"), str(ROOT / "data/replay two.zip"), str(ROOT / "data/dataset/residual-round2.zarr.zip")]


@pytest.mark.parametrize("recipe", ["residual-policy.sh", "residual-policy-step.sh"])
def test_correction_only_keeps_null_mixture_and_zero_loss(tmp_path, recipe):
    _, config = composed(invoke(recipe, tmp_path, "--variant", "correction_only", "--dry-run", "--print-config"))
    assert config["task"]["dataset"]["correction_fraction"] is None
    assert config["policy"]["zero_loss_weight"] == 0
    assert config["policy"]["model"]["gate_enabled"] is False


def test_step_horizon_one_uses_matching_default_input(tmp_path):
    _, config = composed(invoke("residual-policy-step.sh", tmp_path, "--dry-run", "--print-config", settings={"RESIDUAL_HORIZON": "1"}))
    assert config["task"]["residual_horizon"] == 1
    assert config["task"]["dataset"]["dataset_path"] == str(ROOT / "data/dataset/step-residual-h1.zarr.zip")


def test_round_two_resume_does_not_apply_warm_start_again(tmp_path):
    _, config = composed(invoke("residual-policy-round2.sh", tmp_path, "--resume", "--run-dir", "outputs/existing", "--dry-run", "--print-config"))
    assert config["training"]["resume"] is True
    assert config["training"]["init_residual_checkpoint"] is None
