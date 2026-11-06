# ARISE Training — Data-Scaling-Laws fork

This fork supplies the training backend used by
[ARISE](https://github.com/Seeple/ARISE): ordinary Diffusion Policy training,
human-gated DAgger, Sirius weighted finetuning and residual-policy updates.
It is based on [Data-Scaling-Laws](https://github.com/Fanqi-Lin/Data-Scaling-Laws),
which builds on UMI and Diffusion Policy. The reusable intervention, VR editing
and real-robot deployment code lives in ARISE.

The release code is maintained on `arise-release`. ARISE consumes this fork as
`third_party/data-scaling-laws`, pinned to a tested commit. A branch name is
for development; the parent repository's gitlink selects the reproducible version.

## What this fork adds

- DP finetuning with offline/online replay, human-only anchors, action masks,
  contiguous-prefix and invalid-tail options, and observation/action strides.
- Sirius intervention-aware token weights over complete online rollouts.
- Frozen-base geometric residual policies, full-chunk/active-suffix and dense
  step datasets, later-round replay and warm starts, and multibase augmentation.
- Shared launch arguments and method-level shell recipes, local overrides,
  config-only checks, reproducible launch metadata and training logs.

These extensions retain the upstream `diffusion_policy` and `umi` namespaces
for checkpoint and deployment compatibility. No robot SDK or model weights are
distributed here.

## Get the code and environment

As part of ARISE, run from the ARISE repository root:

```bash
git submodule update --init --recursive third_party/data-scaling-laws
```

Or clone this fork directly and check out `arise-release`.
Use a compatible training environment based on `conda_environment.yaml`.
An existing inference environment can often be reused after checking training
dependencies. From this repository's root, Python can import `diffusion_policy`
and `umi` directly. For ARISE deployment from its parent directory, expose this
submodule root on `PYTHONPATH` as described in the ARISE deployment guide.

## Start a training run

The method-level scripts share one set of settings across task names:

| Script under `train_scripts/configs/` | Workflow |
| --- | --- |
| `base-model.sh` | Ordinary DP training |
| `hg-dagger.sh` | Human-gated policy finetuning |
| `sirius.sh` | Complete-rollout Sirius weighted finetuning |
| `residual-policy.sh` | First-round full-chunk residual |
| `residual-policy-round2.sh` | Residual replay and weight warm start |
| `residual-policy-step.sh` | Dense step residual, horizon 1 or 5 |

For example, from this repository's root:

```bash
bash train_scripts/configs/base-model.sh --dry-run --print-config
bash train_scripts/configs/hg-dagger.sh --check
bash train_scripts/configs/residual-policy.sh --variant mixed_gate --dry-run
```

`--dry-run` composes the workspace and prints the command without starting a
job. `--check` also checks input-file presence and import availability, without
loading checkpoints. Remove these flags to train after supplying real inputs.
Recipes use repository-relative `data/dataset/` and `data/ckpts/` paths; edit
the settings or override them with environment values/launcher arguments.
The data directory and weights are ignored by Git.

See the [shell recipe guide](train_scripts/configs/README.md) for the expected
file layout, switches, GPU settings and overrides, and the
[training guide](docs/arise-training.md) for Python/YAML launch recipes,
sampling contracts, replay, normalization and resume.

## Data and deployment contracts

Ordinary DP/DAgger use fixed-rate UMI Zarr data. Sirius needs complete policy
and takeover rollouts with `hitl_tag`, rather than only extracted corrections.
The ARISE HDF5/MP4 converters prepare these experiment datasets.

Residual inputs are event datasets containing the exact frozen-base reference,
human/composite targets and validity masks. `recorded_chunk`,
`active_suffix_left_aligned` and `step_active_plan_left_aligned` are different
contracts. Match the checkpoint's action alignment, frozen-base SHA256 and
state key to deployment. ARISE's default residual preset selects full-chunk
`recorded_chunk`; another alignment requires an explicit matching configuration.
Keeping step residual training here does not add a step deployment adapter.

Checkpoints, normalization assets and example datasets must be supplied
separately. Paths in the historical task scripts describe their original
experiments; new runs should use the method-level recipes above.

## Validation

```bash
python -m pytest -q tests/common tests/dataset tests/model tests/training
```

This focused suite uses temporary data, CPU tensors and configuration
composition. It includes model loss/gradient and feature-return regressions,
residual dataset contracts, launch errors and shell path handling. Hardware
camera tests and full training jobs are separate.

## Upstream attribution

The underlying training framework is the work of the
[Data-Scaling-Laws authors](https://data-scaling-laws.github.io/), based on
[UMI](https://github.com/real-stanford/universal_manipulation_interface) and
[Diffusion Policy](https://github.com/real-stanford/diffusion_policy).
Please cite those works when using their methods, together with ARISE when
using its contributions. The upstream paper, authors, original models/data
links and reproduction instructions are retained in the
[upstream project README](docs/UPSTREAM_README.md).

The existing [MIT license and copyright notice](LICENSE) are preserved.
