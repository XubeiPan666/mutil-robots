#!/bin/bash -l
#SBATCH --job-name=pi0-paper6
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a100:2
#SBATCH --time=24:00:00

# Plain Pi0 six-task baseline. Submission and W&B destination are chosen by the user.
set -eo pipefail
if [[ $# -ne 0 ]]; then echo "This matched-baseline script takes no arguments." >&2; exit 2; fi
source "$(ws_find mutil-robots)/load_env.sh"
export PYTHONNOUSERSITE=1
unset JAX_PLATFORMS
# Keep Slurm's CUDA_VISIBLE_DEVICES unchanged.
cd "$PROJECT_WS/code/mutil-robots"

export FUTURE_DATA_ROOT="$PROJECT_WS/datasets/paper6"
export FUTURE_DATASET_DIR="$FUTURE_DATA_ROOT/multi_task_combined"
export FUTURE_ASSETS_DIR="$FUTURE_DATA_ROOT"
export FUTURE_CHECKPOINT_ROOT="$PROJECT_WS/checkpoints"
export FUTURE_PI05_PARAMS="$PROJECT_WS/cache/openpi/openpi-assets/checkpoints/pi05_base/params"
export FUTURE_PI0_PARAMS="$PROJECT_WS/cache/openpi/openpi-assets/checkpoints/pi0_base/params"

: "${SLURM_JOB_ID:?Submit this script with sbatch, not bash on a login node}"
: "${WANDB_API_KEY:?Set your session WANDB_API_KEY before submitting}"
: "${WANDB_ENTITY:?Choose your own W&B username or team before submitting}"
: "${WANDB_PROJECT:?Choose a W&B project before submitting}"
export WANDB_MODE=online
export WANDB_DIR="$PROJECT_WS/logs/wandb"
export WANDB_CACHE_DIR="$PROJECT_WS/cache/wandb"
export WANDB_CONFIG_DIR="$PROJECT_WS/tools/wandb-config"
export WANDB_DATA_DIR="$PROJECT_WS/cache/wandb-data"
export WANDB_DISABLE_CODE=true
unset WANDB_RUN_ID WANDB_RESUME
CONFIG=pi0_two_arms_multi_task
EXP_NAME="pi0_paper6_${SLURM_JOB_ID}"
RUN_DIR="$PROJECT_WS/runs/$EXP_NAME"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR" "$WANDB_DATA_DIR" "$RUN_DIR"
test ! -e "$FUTURE_CHECKPOINT_ROOT/$CONFIG/$EXP_NAME"

echo "Node: $(hostname -f); job: $SLURM_JOB_ID; experiment: $EXP_NAME"
echo "Started: $(date -Is); W&B: $WANDB_ENTITY/$WANDB_PROJECT"
git rev-parse HEAD | tee "$RUN_DIR/git_commit.txt"
git status --short > "$RUN_DIR/git_status.txt"
cp "$FUTURE_DATASET_DIR/paper6_manifest.json" "$RUN_DIR/paper6_manifest.json"
sha256sum "$FUTURE_DATASET_DIR/norm_stats.json" > "$RUN_DIR/norm_stats.sha256"

JAX_PLATFORMS=cpu uv run --frozen python scripts/paper6_pi0/check_training.py \
  --dataset "$FUTURE_DATASET_DIR" | tee "$RUN_DIR/preflight.log"
nvidia-smi
uv run --frozen python - <<'PY'
import jax
devices = jax.devices()
print("JAX devices:", devices)
assert len(devices) == 2 and all(d.platform == "gpu" for d in devices), "Expected two allocated GPUs"
PY

# No resume/overwrite: train from Pi0 base, never from the seven-task run.
uv run --frozen python scripts/train.py "$CONFIG" \
  --exp-name "$EXP_NAME" \
  --checkpoint-base-dir "$FUTURE_CHECKPOINT_ROOT" \
  --seed 42 --batch-size 32 --num-train-steps 30000 --num-workers 1 --fsdp-devices 2 \
  --log-interval 50 --save-interval 5000 --keep-period 5000 \
  --wandb-enabled --project-name "$WANDB_PROJECT"
echo "Six-task plain Pi0 training complete: $(date -Is)"
