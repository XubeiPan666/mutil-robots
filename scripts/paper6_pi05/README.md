# Six-task standard Pi0.5 baseline on NHR Alex

This uses the existing `pi05_two_arms_multi_task` configuration: one standard
Pi0.5 policy controls both arms. No SkillLoRA, skill router or arm-wise attention
is added. The public dataset's phase-derived atomic instructions remain inputs;
this is an architecture baseline, not a change to whole-task instructions.

## Matched settings

Matches the six-task AE-VLA run: cached Pi0.5 base initialization, six families /
600 episodes, seed 42, global batch 32, 30000 steps, one data-loader worker,
two A100 80 GB GPUs, FSDP 2, 32 CPUs, 24-hour limit, logs every 50 steps,
saving/retaining every 5000 steps and online W&B. The existing optimizer,
learning-rate schedule and EMA are checked against the AE-VLA configuration.
This is a new run, with neither resume nor overwrite. Runtime is not guaranteed
to match AE-VLA or fit the allocation until measured on the cluster.

The six-task dataset and its existing `norm_stats.json` remain at
`$PROJECT_WS/datasets/paper6/multi_task_combined`. The stats program uses repack
and data transforms before model tokenization. Both configurations use identical
two-arm atom state/action processing, including 32-dimensional padding and delta
joint actions; the PI0 versus PI05 override selects the same image/mask branch
in `DroidMultiInputs_atom`. Token grouping and skill labels do not enter these
statistics. Reuse is deliberate for the comparison, not an assumption that
arbitrary configurations can share normalization. Preflight rejects changed
data factories, paths, dimensions and optimization settings. It does not prove
the statistics' historical provenance or perform a full-dataset numerical audit.

The preflight reuses the existing six-task metadata checker, located in either
`scripts/paper6_AE-VLA` (preferred after rename) or `scripts/paper6`. Both directory
names work; the shared metadata includes skill labels even though Pi0.5 does not
use them for training. Do not rename the actual dataset or checkpoint directories.

## Read-only validation on the server

After pushing the local changes to your GitHub repository, pull them on Alex.
Then from the training repository:

```bash
source "$(ws_find mutil-robots)/load_env.sh"
export PYTHONNOUSERSITE=1
cd "$PROJECT_WS/code/mutil-robots"
export FUTURE_DATA_ROOT="$PROJECT_WS/datasets/paper6"
export FUTURE_DATASET_DIR="$FUTURE_DATA_ROOT/multi_task_combined"
export FUTURE_ASSETS_DIR="$FUTURE_DATA_ROOT"
export FUTURE_PI05_PARAMS="$PROJECT_WS/cache/openpi/openpi-assets/checkpoints/pi05_base/params"
bash -n scripts/paper6_pi05/train.sh
JAX_PLATFORMS=cpu uv run --frozen python scripts/paper6_pi05/check_training.py \
  --dataset "$FUTURE_DATASET_DIR"
```

This loads configuration and checks metadata/files; no training model or W&B
run is started. The batch repeats this preflight before loading the GPU model.

## W&B and submission

The trainer uploads configuration, scalar metrics and up to five first-batch
camera samples. Code upload is disabled; this script adds no checkpoint artifact
upload. Set your own destination and a session-only key. Never put the key in Git,
logs, command arguments, shared `~/.netrc` or chat.

```bash
export WANDB_ENTITY="xubei-sun-yat-sen-university"
export WANDB_PROJECT="mutil-robots"
if [ -z "${WANDB_API_KEY:-}" ]; then
    read -rsp 'W&B API Key (hidden input): ' WANDB_API_KEY
    echo
fi
export WANDB_API_KEY

mkdir -p "$PROJECT_WS/logs"
sbatch --export=ALL \
  --output="$PROJECT_WS/logs/train_pi05_paper6_%j.out" \
  --error="$PROJECT_WS/logs/train_pi05_paper6_%j.err" \
  scripts/paper6_pi05/train.sh
```

Submit yourself after validation; creating these files submits no job. Do not
run the training script directly on a login node. The wrapper rejects additional
arguments so attempted overrides cannot be silently ignored. Record any future
recipe changes explicitly. Slurm continues after disconnecting WindTerm.

Paths for a submitted `JOBID`:

- Experiment: `pi05_paper6_JOBID`
- Checkpoints: `$PROJECT_WS/checkpoints/pi05_two_arms_multi_task/pi05_paper6_JOBID/`
- Provenance: `$PROJECT_WS/runs/pi05_paper6_JOBID/` (Git commit/status, manifest,
  normalization SHA256, preflight report)
- Logs: `$PROJECT_WS/logs/train_pi05_paper6_JOBID.out` and `.err`
- Final checkpoint normally uses step `29999`; confirm successful completion.

For later evaluation select `pi05_two_arms_multi_task` in the policy server too.
An evaluation launcher fixed to the AE-VLA model must be adapted before serving
these weights; changing the checkpoint path alone is insufficient.

## Local validation

Run `python -m unittest discover -s tests -p 'test_paper6*.py' -v` with Python 3.11+.
Parquet integration tests need PyArrow. The Pi0.5 preflight guard tests use small
configuration fixtures and do not replace running the real config preflight on
Alex. Shell syntax checks and local tests do not establish GPU training success.

Verified locally on 2026-11-05: 13 tests passed, one Parquet integration test
skipped because PyArrow was unavailable. The same result was obtained in a
temporary copy with `paper6` renamed to `paper6_AE-VLA`. All three affected Bash
scripts passed `bash -n`. Actual OpenPI configuration imports, GPU allocation,
training and W&B authentication remain to be checked on the server.
