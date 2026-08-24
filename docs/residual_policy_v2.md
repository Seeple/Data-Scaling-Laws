# Residual Policy v2 experiments

The residual event schema supports two strict action alignments:

- `recorded_chunk`: the original frozen chunk and suffix mask. This reproduces
  the first residual-policy experiments.
- `active_suffix_left_aligned`: the unexecuted active suffix is moved to index
  zero, the invalid padded tail is masked, and the observation is taken at the
  intervention/execution boundary. This makes the learned correction visible
  in the next five-step execution window.

Always pass `ACTION_ALIGNMENT` to `train_scripts/manage_table_residual.sh`.
The dataset is checked at startup and the alignment is stored in every
checkpoint.

## Training

```bash
BASE_POLICY_CKPT=/absolute/path/to/base.ckpt \
RESIDUAL_DATASET_PATH=/absolute/path/to/residual.zarr.zip \
TASK_NAME=cable_routing \
VARIANT=mixed_no_gate \
ACTION_ALIGNMENT=active_suffix_left_aligned \
NUM_EPOCHS=200 BATCH_SIZE=32 NUM_WORKERS=16 \
bash train_scripts/manage_table_residual.sh
```

`VARIANT` remains one of `correction_only`, `mixed_no_gate`, or `mixed_gate`.
The dataset converter, rather than the trainer, controls ambiguous-zero
filtering. Correction/zero balancing remains controlled by
`CORRECTION_FRACTION` and defaults to 0.5.

## Filtered multi-base materialization

Run this once before training; do not resample independently in every DDP
worker:

```bash
python diffusion_policy/scripts/augment_residual_multibase.py \
  INPUT_ACTIVE_SUFFIX.zarr.zip \
  OUTPUT_ACTIVE_SUFFIX_MULTIBASE.zarr.zip \
  --base-policy-ckpt /absolute/path/to/base.ckpt \
  --candidates-per-correction 2 \
  --max-draw-multiplier 8 \
  --sample-batch-size 4 \
  --device cuda \
  --overwrite
```

The tool verifies the base checkpoint SHA256, samples the frozen DP at the
same correction observation, converts each candidate to the absolute robot
frame, and accepts it only if:

1. its first/next-five waypoint geometry is close to the recorded active plan;
2. the residual to the edited goal stays inside configured safety bounds; and
3. it is not a duplicate of an already accepted base sample.

Augmented rows retain `source_episode_index`, so train/validation splitting
cannot leak resamples of one correction across splits. The output Zarr stores
`multibase_source_sample_index`, `multibase_resample_index`, filter attributes,
and a sibling `.multibase.json` report. The balanced sampler also assigns equal
total correction mass to each original event family; a correction with two
accepted candidates therefore has the same family-level probability as one
with no accepted candidate.
