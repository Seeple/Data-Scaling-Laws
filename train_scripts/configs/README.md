# Method-level training recipes

These executable shell files replace per-task copies of launch settings. They
call the shared Python launcher and compose the existing training workspaces.
Edit the settings near the top of a recipe, supply environment values, or pass
launcher arguments after its filename. No local YAML is required.

| Script | Training workflow |
| --- | --- |
| `base-model.sh` | Ordinary DP training from demonstrations |
| `hg-dagger.sh` | Checkpoint finetuning using human-only online anchors, action masks and the original pipeline's HITL stride multiplier |
| `sirius.sh` | Checkpoint finetuning using complete online rollouts and Sirius token weights |
| `residual-policy.sh` | First-round full-chunk residual, default `mixed_no_gate` and `recorded_chunk` alignment |
| `residual-policy-round2.sh` | Full-chunk residual with replay, previous residual weights and initial EMA decay protection |
| `residual-policy-step.sh` | Dense active-plan residual, horizon 1 or 5; accepts replay/warm-start arguments for later rounds |

`_common.sh` is a sourced helper, not a training recipe. The selected task name
defaults to `manage_table`; `TASK_NAME` changes it without a separate script.
The recipes use the same explicit batch/epoch settings across task names.

## Run and inspect

From the repository root:

```bash
bash train_scripts/configs/base-model.sh --dry-run --print-config
bash train_scripts/configs/hg-dagger.sh --check
bash train_scripts/configs/sirius.sh
bash train_scripts/configs/residual-policy.sh --variant mixed_gate
bash train_scripts/configs/residual-policy-round2.sh
RESIDUAL_HORIZON=1 bash train_scripts/configs/residual-policy-step.sh
```

An absolute script path also works from another directory. Each recipe changes
to the repository root first: **all relative paths in shell settings and
forwarded CLI arguments are relative to the repository root**. The standalone
Python launcher's YAML-relative behavior is unchanged.

`--dry-run` composes and prints the workspace without checking input existence
or starting training. `--check` additionally verifies input-file presence and
dependency availability, without deserializing checkpoints. It does not certify
dataset schema, checkpoint compatibility or CUDA readiness. Training begins
when neither flag is supplied. `PYTHON` selects the interpreter if needed.

## Relative data layout

The portable defaults expect the following **external, untracked** files:

```text
data/
  dataset/
    teleop.zarr.zip
    hitl.zarr.zip
    residual-round1.zarr.zip
    residual-round2.zarr.zip
    step-residual-h1.zarr.zip
    step-residual-h5.zarr.zip
  ckpts/
    base.ckpt
    residual-round1.ckpt
```

Only the inputs used by a selected method are required. Place or link the real
artifacts into this layout, or override the paths. These names are aliases for
your experiment artifacts, not datasets or weights distributed with this repo.
For Sirius, `hitl.zarr.zip` must retain both policy and takeover frames.
Residual Zarr files must match their declared alignment/horizon.

```bash
DATA_ROOT=data/my_experiment GPU_LIST=0,1 NUM_PROCESSES=2 \
  bash train_scripts/configs/hg-dagger.sh --dry-run

TELEOP_DATASET_PATH=data/dataset/expert_original_name.zarr.zip \
FINETUNE_CKPT=data/ckpts/my_base.ckpt \
  bash train_scripts/configs/hg-dagger.sh \
    --online-dataset data/dataset/round1.zarr.zip \
    --online-dataset data/dataset/round2.zarr.zip \
    --set 'task.dataset.rlpd_ratio=[0.4,0.2,0.4]' --check
```

Explicit `--online-dataset`/`--replay-dataset` lists replace a recipe's default
input list. Other repeated scalar options use the final CLI value. `--set`
settings passed after the recipe override its earlier settings while retaining
the other recipe settings. GPU, epoch, batch, worker, precision and logging
environment values use the original scripts' names (`GPU_LIST`, `NUM_EPOCHS`,
`BATCH_SIZE`, `NUM_WORKERS`, `MIXED_PRECISION`, `USE_WANDB`, etc.). Vision
initialization defaults to pretrained for base training and disabled for
checkpoint finetuning. Recipes default to W&B logging; set `USE_WANDB=false`
or pass `--no-wandb` to disable it. Optional `CACHE_DIR` requires LMDB.

HG-DAgger defaults follow the old `pipeline.sh`: human anchors/action masks
and HITL downsample multiplier 2. Contiguous masks and invalid-tail padding
remain opt-in. The corresponding `HITL_*` variables can change these switches;
other ablations remain available through `--set`.

Round two defaults to `residual-round2.zarr.zip` plus replay from round one and
`residual-round1.ckpt`. Set `RESIDUAL_DATASET_PATH`,
`REPLAY_RESIDUAL_DATASET_PATH` and `INIT_RESIDUAL_CKPT` to use other files.
The step recipe accepts these same environment names. For an interrupted run,
pass `--resume --run-dir outputs/your_run`; automatic weight-only warm start is
disabled during resume. Existing normalizer, frozen-base hash and alignment
checks remain in the training workspace.

The source and config paths have been tested from outside the repository.
Input-lookup tests use temporary stubs (never loaded as models). Real experiment
data and weights must be supplied separately; a missing artifact is reported
with its resolved path before training can start.
