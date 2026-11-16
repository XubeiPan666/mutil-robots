#!/bin/bash -l
#SBATCH --job-name=pi05-singlearm-norm
#SBATCH --partition=a100mig
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00

# CPU-only statistics. No GPU, model, training, or W&B run.
set -eo pipefail
source "$(ws_find mutil-robots)/load_env.sh"
export PYTHONNOUSERSITE=1
export JAX_PLATFORMS=cpu
export PYTHONUNBUFFERED=1
cd "$PROJECT_WS/code/mutil-robots"
DATASET="$PROJECT_WS/datasets/paper6/multi_task_combined"
OUTPUT="$PROJECT_WS/assets/intent_ar_single_arm/paper6_${SLURM_JOB_ID:?Submit with sbatch}"

echo "Node: $(hostname -f); job: $SLURM_JOB_ID; started: $(date -Is)"
echo "Dataset: $DATASET"
echo "Output: $OUTPUT"
uv run --frozen python scripts/intent_ar/compute_norm_stats.py \
    --dataset "$DATASET" --output "$OUTPUT" --apply
test -s "$OUTPUT/norm_stats.json"
echo "Shared single-arm stats job complete: $(date -Is)"
