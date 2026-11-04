"""Exercise composed workflows and process behavior without GPUs or model files."""
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from diffusion_policy.launch import (
    ROOT, TASKS, METHODS, build_plan, check_plan, parse_args, run_plan,
)


def arguments(method="pretrain", task="manage_table"):
    args = ["--task", task, "--method", method, "--num-workers", "0", "--val-num-workers", "0"]
    if method in ("dagger", "sirius"):
        args += ["--teleop-dataset", "/tmp/expert data.zip", "--online-dataset", "/tmp/online data.zip", "--checkpoint", "/tmp/base weights.ckpt"]
    elif method in ("residual", "step-residual"):
        args += ["--dataset", "/tmp/residual events.zip", "--base-checkpoint", "/tmp/base weights.ckpt"]
    else:
        args += ["--dataset", "/tmp/expert data.zip"]
    return args


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("method", METHODS)
def test_all_task_method_combinations_compose(task, method):
    plan = build_plan(parse_args(arguments(method, task)))
    assert not plan.run_dir.exists()
    assert plan.config["task"]["name"] == task
    assert plan.config["task"]["shape_meta"]["action"]["shape"] == [10]
    assert plan.config["dataloader"]["persistent_workers"] is False
    assert plan.config["val_dataloader"]["persistent_workers"] is False
    assert plan.command[:3] == [sys.executable, "-m", "accelerate.commands.launch"]
    if method == "pretrain":
        assert plan.config["policy"]["obs_encoder"]["pretrained"] is True
        assert plan.config["task"]["dataset_frequeny"] == 0
    if method == "sirius":
        assert plan.config["task"]["dataset"]["sirius_weighting_enabled"]
        assert plan.config["training"]["finetune_from"] == "/tmp/base weights.ckpt"
        assert plan.config["task"]["dataset_frequeny"] == 15
    if method == "residual":
        assert plan.config["task"]["dataset"]["expected_action_alignment"] == "recorded_chunk"


def test_multibuffer_and_warm_start_are_composable():
    dagger = build_plan(parse_args(arguments("dagger") + ["--online-dataset", "/tmp/round two.zip", "--gpus", "0,1"]))
    assert dagger.config["task"]["dataset"]["rlpd_ratio"] == [0.5, 0.25, 0.25]
    assert "--multi_gpu" in dagger.command
    assert dagger.environment["CUDA_VISIBLE_DEVICES"] == "0,1"
    residual = build_plan(parse_args(arguments("residual") + ["--replay-dataset", "/tmp/round one.zip", "--init-residual", "/tmp/first residual.ckpt", "--variant", "mixed_gate"]))
    dataset = residual.config["task"]["dataset"]
    assert dataset["_target_"].endswith("UmiResidualDatasetCollection")
    assert dataset["dataset_paths"] == ["/tmp/round one.zip", "/tmp/residual events.zip"]
    assert residual.config["training"]["init_residual_checkpoint"] == "/tmp/first residual.ckpt"
    assert residual.config["policy"]["model"]["gate_enabled"]
    step = build_plan(parse_args(arguments("step-residual") + ["--residual-horizon", "1"]))
    assert step.config["policy"]["model"]["residual_horizon"] == 1
    assert step.config["task"]["dataset"]["expected_action_alignment"] == "step_active_plan_left_aligned"


def test_yaml_paths_and_cli_precedence(tmp_path, monkeypatch):
    config = tmp_path / "local.yaml"
    config.write_text(yaml.safe_dump({"task": "manage_table", "method": "dagger", "teleop_dataset": "expert.zip", "online_datasets": ["old.zip"], "checkpoint": "base.ckpt", "batch_size": 9, "overrides": ["training.num_epochs=3"]}))
    invocation = tmp_path / "invocation"
    invocation.mkdir()
    monkeypatch.chdir(invocation)
    plan = build_plan(parse_args(["--config", str(config), "--batch-size", "17", "--online-dataset", "new.zip", "--set", "training.num_epochs=4"]))
    assert plan.config["task"]["dataset"]["teleop_dataset_path"] == str(tmp_path / "expert.zip")
    assert plan.config["task"]["dataset"]["online_dataset_paths"] == [str(invocation / "new.zip")]
    assert plan.config["dataloader"]["batch_size"] == 17
    assert plan.config["training"]["num_epochs"] == 4


@pytest.mark.parametrize("extra,message", [
    (["--num-processes", "0"], "positive integer"),
    (["--num-workers", "-1"], "nonnegative integer"),
    (["--gpus", "0, 0"], "distinct"),
    (["--gpus", "0", "--num-processes", "2"], "exceeds"),
    (["--set", "dataloader.persistent_workers=true"], "persistent_workers"),
])
def test_reject_invalid_resources_before_launch(extra, message):
    with pytest.raises(ValueError, match=message):
        build_plan(parse_args(arguments() + extra))


def test_sirius_rejects_correction_only_masks():
    with pytest.raises(ValueError, match="complete online rollouts"):
        build_plan(parse_args(arguments("sirius") + ["--set", "task.dataset.hitl_action_mask=true"]))


def test_check_reads_effective_overridden_paths(tmp_path, monkeypatch):
    monkeypatch.setattr("diffusion_policy.launch.importlib.util.find_spec", lambda name: object())
    dataset = tmp_path / "valid.zip"
    dataset.touch()
    plan = build_plan(parse_args(arguments() + ["--set", 'task.dataset_path=' + json.dumps(str(dataset))]))
    check_plan(plan)
    dataset.unlink()
    with pytest.raises(ValueError, match="Input not found"):
        check_plan(plan)


def test_process_failure_is_logged_and_returned(tmp_path):
    plan = build_plan(parse_args(arguments() + ["--run-dir", str(tmp_path / "run")]))
    script = "import os,sys; print(os.getcwd(),flush=True); print('synthetic failure',flush=True); sys.exit(17)"
    plan = replace(plan, command=[sys.executable, "-c", script])
    assert run_plan(plan) == 17
    assert "synthetic failure" in (plan.run_dir / "train.log").read_text()
    manifest = json.loads(next(plan.run_dir.glob("launch-*.json")).read_text())
    assert manifest["cwd"] == str(ROOT)
    assert "environment" not in manifest
    assert next(plan.run_dir.glob("resolved-*.yaml")).is_file()
    with pytest.raises(FileExistsError):
        run_plan(plan)


def test_resume_does_not_load_finetune_weights_again(tmp_path):
    plan = build_plan(parse_args(arguments("dagger") + ["--run-dir", str(tmp_path), "--resume"]))
    assert plan.config["training"]["resume"] is True
    assert plan.config["training"]["finetune_from"] is None


def test_accelerate_accepts_generated_single_and_multi_gpu_arguments():
    from accelerate.commands.launch import launch_command_parser
    from accelerate.commands.config.config_args import load_config_from_file

    for gpus in ("0", "0,1"):
        plan = build_plan(parse_args(arguments() + ["--gpus", gpus]))
        parsed = launch_command_parser().parse_args(plan.command[3:])
        config = load_config_from_file(parsed.config_file)
        assert config.distributed_type.value == "NO"
        assert parsed.num_processes == len(gpus.split(","))
        assert parsed.multi_gpu == (parsed.num_processes > 1)
        assert parsed.training_script == str(ROOT / "train.py")


@pytest.mark.parametrize("method", METHODS)
def test_actual_train_entry_composes_without_loading_a_model(tmp_path, method):
    plan = build_plan(parse_args(arguments(method)))
    index = plan.command.index(str(ROOT / "train.py"))
    command = [sys.executable, *plan.command[index:], "--cfg", "job", "--resolve"]
    result = subprocess.run(command, cwd=tmp_path, env=plan.environment, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert yaml.safe_load(result.stdout) == plan.config
    assert not plan.run_dir.exists()


def test_wrapper_from_other_cwd_has_no_model_import_or_output(tmp_path):
    environment = os.environ.copy()
    environment["PYTHON"] = sys.executable
    result = subprocess.run(["bash", str(ROOT / "train_scripts/launch.sh"), *arguments(), "--dry-run", "--run-dir", str(tmp_path / "not-created")], cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "not-created").exists()
    result = subprocess.run([sys.executable, "-c", "import sys; from diffusion_policy.launch import *; build_plan(parse_args(" + repr(arguments()) + ")); assert 'torch' not in sys.modules"], cwd=ROOT, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
