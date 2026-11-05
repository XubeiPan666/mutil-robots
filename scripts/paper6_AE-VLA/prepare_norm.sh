#!/bin/bash -l
#SBATCH --job-name=aevla-paper6-data
#SBATCH --partition=a100mig
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00

# CPU-only conversion and stats. Does not start training or W&B.
set -eo pipefail
source "$(ws_find mutil-robots)/load_env.sh"
export PYTHONNOUSERSITE=1
export JAX_PLATFORMS=cpu
cd "$PROJECT_WS/code/mutil-robots"
AEVLA_SCRIPTS=scripts/paper6
if [[ -d scripts/paper6_AE-VLA ]]; then AEVLA_SCRIPTS=scripts/paper6_AE-VLA; fi
export FUTURE_DATA_ROOT="$PROJECT_WS/datasets/paper6"
export FUTURE_DATASET_DIR="$FUTURE_DATA_ROOT/multi_task_combined"
export FUTURE_ASSETS_DIR="$FUTURE_DATA_ROOT"
SOURCE="$PWD/dataset/multi_task_combined"
REVISION="$(cat "$PROJECT_WS/macg_data_revision.txt")"
test "$REVISION" = 447112d1c5b2ec10f7ad34ed438532c062bc51bc

echo "Node: $(hostname -f); job: $SLURM_JOB_ID; started: $(date -Is)"
uv run --frozen python "$AEVLA_SCRIPTS/prepare_dataset.py" \
    --source "$SOURCE" --output "$FUTURE_DATASET_DIR" --revision "$REVISION" --apply
uv run --frozen python scripts/compute_norm_stats.py \
    --config-name aevla_pi05_skill_lora_two_arms_multitask
test -s "$FUTURE_DATASET_DIR/norm_stats.json"
echo "Paper-six dataset and normalization complete: $(date -Is)"
