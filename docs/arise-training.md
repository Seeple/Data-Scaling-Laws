# ARISE training workflows

This fork combines the ordinary DP/DAgger, Sirius and residual training
branches. Selected legacy task scripts remain as experiment records; unused
task scripts have been removed. Use the method-level shell recipes or the
shared Python launcher for new runs.

## Environment and entry point

Use a training environment with the repository's dependencies from
`conda_environment.yaml`. An existing inference environment can run config
composition, but is not automatically a complete training environment.
The repository root must be on `PYTHONPATH` when importing it elsewhere.

From the repository root:

```bash
python -m diffusion_policy.launch --help
```

Or invoke `bash /absolute/path/to/train_scripts/launch.sh ...` from any
directory. It uses the active Python interpreter (`PYTHON` can select another).
The launcher runs the existing `train.py` through Accelerate, without loading
models in the launcher process.

| Method | Input and behavior |
| --- | --- |
| `pretrain` | Demonstration Zarr; ordinary diffusion-policy training |
| `dagger` | DP checkpoint, offline demonstrations and one or more online buffers; original policy finetuning and HITL ablations |
| `sirius` | Same inputs as DAgger, with complete online rollouts and intervention-aware token loss weights |
| `residual` | Frozen DP checkpoint and event Zarr; full-chunk residual training |
| `step-residual` | Frozen DP checkpoint and dense active-plan Zarr; residual horizon 1 or 5 |

All methods accept `hang_chinese_knot`, `manage_table` and `cable_routing`.
Task selection controls run naming and ordinary training batch/epoch defaults;
the same single-arm UMI shape contract is used. The new cable pretraining
recipe is generic: the original master supplied only its DAgger script.

## Method-level shell recipes

For editable, version-controlled shell settings, use
[`train_scripts/configs/`](../train_scripts/configs/README.md):

```bash
bash train_scripts/configs/base-model.sh --dry-run --print-config
bash train_scripts/configs/hg-dagger.sh --check
bash train_scripts/configs/sirius.sh --dry-run
bash train_scripts/configs/residual-policy.sh --dry-run
bash train_scripts/configs/residual-policy-round2.sh --dry-run
RESIDUAL_HORIZON=1 bash train_scripts/configs/residual-policy-step.sh --dry-run
```

Each method has one recipe across task names. The shell recipes resolve paths
from the repository root even when invoked from another working directory.
They default to a portable `data/dataset/` and `data/ckpts/` layout; supply the
actual artifacts or change their path settings before training. Their W&B and
HG-DAgger switches follow the older shell entry points. The independent Python
launcher keeps its CLI/YAML path and logging conventions described below.

## Launch examples

Start with `--dry-run` to compose the actual workspace config and print the
exact argument vector. It neither checks that input files exist nor creates
output directories. Add `--print-config` to inspect the resolved config.
Use `--check` to check input files and import availability, without loading
a checkpoint or model. Remove `--dry-run` to train.

```bash
python -m diffusion_policy.launch --task manage_table --method pretrain \
  --dataset /data/expert.zarr.zip --gpus 0 --dry-run

python -m diffusion_policy.launch --task hang_chinese_knot --method dagger \
  --teleop-dataset /data/expert.zarr.zip \
  --online-dataset /data/round1.zarr.zip --checkpoint /models/base.ckpt \
  --set task.dataset.hitl_only_tag=true \
  --set task.dataset.hitl_action_mask=true --dry-run

python -m diffusion_policy.launch --task cable_routing --method sirius \
  --teleop-dataset /data/expert.zarr.zip \
  --online-dataset /data/complete-online-rollouts.zarr.zip \
  --checkpoint /models/base.ckpt --gpus 0,1 --dry-run

python -m diffusion_policy.launch --task manage_table --method residual \
  --dataset /data/residual-events.zarr.zip --base-checkpoint /models/base.ckpt \
  --variant mixed_gate --action-alignment recorded_chunk --dry-run

python -m diffusion_policy.launch --task cable_routing --method step-residual \
  --dataset /data/dense-step-events.zarr.zip --base-checkpoint /models/base.ckpt \
  --residual-horizon 5 --dry-run
```

`--gpus 0,1` defaults to two processes. `--num-processes` and `--port` can
override launch resources; use a separate port for concurrent runs. A single
process uses the single-GPU Accelerate path. `--precision` defaults to `bf16`.
`--num-workers 0 --val-num-workers 0` also disables persistent workers.

## Local recipes and exact overrides

Copy `train_scripts/configs/local.example.yaml` to
`train_scripts/configs/local.yaml` (ignored), edit it, and run:

```bash
python -m diffusion_policy.launch --config train_scripts/configs/local.yaml --dry-run
```

YAML paths resolve relative to the YAML file; CLI paths resolve relative to
the invoking directory. CLI settings override YAML. A repeated list option
replaces the corresponding YAML list. No launch behavior depends on starting
inside `train_scripts/`.

`--set KEY=VALUE` exposes the existing Hydra configuration, including HITL
mask/padding/downsampling ablations, sampling ratios, normalization source,
Sirius class weights and residual losses. Repeat it for several overrides.
Use `--run-dir` and `--resume` for run ownership rather than overriding those
two Hydra keys. Unknown Hydra keys are rejected during composition.

The defaults preserve ordinary scripts' DINOv2-L backbone, 16-step action
horizon and stride 3. Pretraining uses pretrained vision weights by default;
choose `--no-pretrained` explicitly for training without downloading them.
DAgger/Sirius obtain weights from the supplied DP checkpoint and default to
no vision-weight download. Weights & Biases is opt-in with `--wandb`.
Ordinary recipes retain `task.dataset_frequeny=0` (the historical spelling)
so they do not silently introduce sensor latency offsets; Sirius retains its
source setting of 15. These rates and masks should match the input conversion.

The cache is disabled by default; `--cache-dir` opts into the existing LMDB
cache and requires `lmdb`. There are no hard-coded home, SSD or HF-cache paths.
Use standard HF environment variables when pretrained weights are needed.

## Later rounds, alignment and resume

Repeat `--online-dataset` for multiple DAgger/Sirius online buffers. Defaults
allocate half the sampling mass to offline data and share the other half
among online buffers. Override `task.dataset.rlpd_ratio` for an ablation.
Sirius requires both policy and takeover frames (`hitl_tag=0/1`); correction-only
filters and masks are rejected in that method.

Residual methods accept repeated `--replay-dataset` plus the current
`--dataset`. Each input gets an equal mixture weight by default. Use
`--init-residual` to warm-start the previous residual, retaining the original
normalizer/hash/alignment checks. For full-chunk experiments, the retained
normalizer reparameterization and `training.init_ema_min_decay` controls are
available through `--set`. This is a fresh optimizer run, distinct from resume.

`--variant` selects `correction_only`, `mixed_no_gate` or `mixed_gate`.
Full-chunk alignment defaults to `recorded_chunk`, which matches the ARISE
deployment preset. `active_suffix_left_aligned` must use a matching deployment
adapter. Step residual training is preserved here; ARISE's existing deployment
preset does not deploy a step residual policy.

To resume an interrupted run, retain the same recipe, pass its existing
`--run-dir` and add `--resume`. This requires `checkpoints/latest.ckpt`.
For DAgger/Sirius the original `--checkpoint` field stays in the recipe, but
`training.finetune_from` is cleared so resume does not overwrite resumed weights.

## Outputs and validation

New runs get unique `outputs/TASK/METHOD/TIMESTAMP/` directories. An explicit
existing directory is rejected unless resuming. Each launch saves the exact
command, source commit/dirty flag and resolved workspace YAML, and streams
stdout/stderr into `train.log`. Training failures retain their exit status.
Ctrl+C is forwarded to the whole training process group. Launch manifests
do not serialize the process environment or its credentials.

The focused offline suite is:

```bash
python -m pytest -q tests/common tests/dataset tests/model tests/training
```

It covers all 15 task/method combinations, Sirius weights and masks, residual
geometry/datasets/model gradients, both merged DP APIs, path precedence,
multi-GPU argument construction, replay/warm starts and failed-process logs.
It does not run the hardware camera tests, a full training job or a model download.
