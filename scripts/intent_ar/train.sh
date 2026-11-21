#!/bin/bash -l
#SBATCH --job-name=pi05-intent-ar
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:a100:4
#SBATCH --time=24:00:00

# Shared launcher for the six-task R1/R2/R3 experiments. Submit manually.
set -eo pipefail

usage() {
    cat <<'EOF'
Usage: sbatch [Slurm options] scripts/intent_ar/train.sh CONFIG [--smoke] [--gpus 2|4]
CONFIG: r1 | r2 | r3 (or the corresponding full config name)
  r1 = pi05_intent_ar_r1_no_comm_fixed_order
  r2 = pi05_intent_ar_r2_comm_fixed_order
  r3 = pi05_intent_ar_r3_comm_random_order
Default: batch 32, FSDP 4, 30000 steps, save/keep every 5000 steps, W&B enabled.
--smoke: batch 32, 2 steps and per-step logging; validates the real target batch.
--gpus: FSDP shard count. The sbatch --gres request must match this value.
Required exported variables: FUTURE_INTENT_STATS_DIR, WANDB_API_KEY,
WANDB_ENTITY, WANDB_PROJECT. Statistics must match the six-task dataset.
Use bash scripts/intent_ar/train.sh --help to view help without Slurm.
EOF
}

if [[ ${1:-} == --help || ${1:-} == -h ]]; then usage; exit 0; fi
case ${1:-} in
    r1|pi05_intent_ar_r1_no_comm_fixed_order) CONFIG=pi05_intent_ar_r1_no_comm_fixed_order ;;
    r2|pi05_intent_ar_r2_comm_fixed_order) CONFIG=pi05_intent_ar_r2_comm_fixed_order ;;
    r3|pi05_intent_ar_r3_comm_random_order) CONFIG=pi05_intent_ar_r3_comm_random_order ;;
    *) usage >&2; exit 2 ;;
esac
shift
MODE=full
BATCH_SIZE=32
FSDP_DEVICES=4
EXPECTED_GPUS=4
TRAIN_STEPS=30000
LOG_INTERVAL=50
SAVE_INTERVAL=5000
KEEP_PERIOD=5000
while [[ $# -gt 0 ]]; do
    case $1 in
        --smoke)
            MODE=smoke
            TRAIN_STEPS=2
            LOG_INTERVAL=1
            SAVE_INTERVAL=2
            KEEP_PERIOD=2
            shift
            ;;
        --gpus)
            [[ ${2:-} == 2 || ${2:-} == 4 ]] || { usage >&2; exit 2; }
            FSDP_DEVICES=$2
            EXPECTED_GPUS=$2
            shift 2
            ;;
        *) usage >&2; exit 2 ;;
    esac
done

: "${SLURM_JOB_ID:?Submit with sbatch, not bash on a login node}"
: "${FUTURE_INTENT_STATS_DIR:?Export the completed shared single-arm statistics directory}"
: "${WANDB_API_KEY:?Set your session W&B API Key before submitting}"
: "${WANDB_ENTITY:?Choose your W&B username or team before submitting}"
: "${WANDB_PROJECT:?Choose a W&B project before submitting}"
source "$(ws_find mutil-robots)/load_env.sh"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
unset JAX_PLATFORMS
# Preserve Slurm's CUDA_VISIBLE_DEVICES; both arms share one model sharded over all requested GPUs.
cd "$PROJECT_WS/code/mutil-robots"
export FUTURE_DATA_ROOT="$PROJECT_WS/datasets/paper6"
export FUTURE_DATASET_DIR="$FUTURE_DATA_ROOT/multi_task_combined"
export FUTURE_ASSETS_DIR="$FUTURE_DATA_ROOT"
export FUTURE_INTENT_STATS_DIR
export FUTURE_CHECKPOINT_ROOT="$PROJECT_WS/checkpoints"
export FUTURE_PI05_PARAMS="$PROJECT_WS/cache/openpi/openpi-assets/checkpoints/pi05_base/params"
test -d "$FUTURE_PI05_PARAMS"
test -f "$FUTURE_DATASET_DIR/paper6_manifest.json"

export WANDB_MODE=online
export WANDB_DIR="$PROJECT_WS/logs/wandb"
export WANDB_CACHE_DIR="$PROJECT_WS/cache/wandb"
export WANDB_CONFIG_DIR="$PROJECT_WS/tools/wandb-config"
export WANDB_DATA_DIR="$PROJECT_WS/cache/wandb-data"
export WANDB_DISABLE_CODE=true
unset WANDB_RUN_ID WANDB_RESUME
EXP_NAME="${CONFIG}_paper6_${MODE}_${SLURM_JOB_ID}"
RUN_DIR="$PROJECT_WS/runs/$EXP_NAME"
test ! -e "$FUTURE_CHECKPOINT_ROOT/$CONFIG/$EXP_NAME"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR" "$WANDB_DATA_DIR" "$RUN_DIR"

echo "Node: $(hostname -f); job: $SLURM_JOB_ID; experiment: $EXP_NAME"
echo "Started: $(date -Is); config: $CONFIG; mode: $MODE"
echo "W&B: $WANDB_ENTITY/$WANDB_PROJECT (configuration, metrics, up to five first-batch camera samples)"
git rev-parse HEAD | tee "$RUN_DIR/git_commit.txt"
git status --short > "$RUN_DIR/git_status.txt"
cp "$0" "$RUN_DIR/train.sh"
cp "$FUTURE_DATASET_DIR/paper6_manifest.json" "$RUN_DIR/paper6_manifest.json"

# Validate the selected config and shared statistics before allocating model weights.
JAX_PLATFORMS=cpu uv run --frozen python scripts/intent_ar/check_model.py \
    --config "$CONFIG" --dataset "$FUTURE_DATASET_DIR" \
    --stats-dir "$FUTURE_INTENT_STATS_DIR" | tee "$RUN_DIR/preflight.log"
cp "$FUTURE_INTENT_STATS_DIR/stats_manifest.json" "$RUN_DIR/stats_manifest.json"
cp "$FUTURE_INTENT_STATS_DIR/norm_stats.json" "$RUN_DIR/norm_stats.json"
sha256sum "$FUTURE_INTENT_STATS_DIR/norm_stats.json" > "$RUN_DIR/norm_stats.sha256"
nvidia-smi
export EXPECTED_GPUS
uv run --frozen python - <<'PY'
import os
import jax

expected = int(os.environ["EXPECTED_GPUS"])
devices = jax.devices()
print("JAX devices:", devices)
assert len(devices) == expected and all(d.platform == "gpu" for d in devices), (
    f"Expected {expected} allocated GPUs, got {devices}"
)
PY

# Start from Pi0.5 base. Do not resume or overwrite an earlier experiment.
TRAIN_COMMAND=(uv run --frozen python scripts/train.py "$CONFIG"
    --exp-name "$EXP_NAME" --checkpoint-base-dir "$FUTURE_CHECKPOINT_ROOT"
    --seed 42 --batch-size "$BATCH_SIZE" --num-train-steps "$TRAIN_STEPS"
    --num-workers 1 --fsdp-devices "$FSDP_DEVICES" --log-interval "$LOG_INTERVAL"
    --save-interval "$SAVE_INTERVAL" --keep-period "$KEEP_PERIOD"
    --wandb-enabled --project-name "$WANDB_PROJECT")
printf '%q ' "${TRAIN_COMMAND[@]}" > "$RUN_DIR/train_command.txt"
printf '\n' >> "$RUN_DIR/train_command.txt"
cat "$RUN_DIR/train_command.txt"
"${TRAIN_COMMAND[@]}"
echo "Intent AR six-task $MODE training complete: $(date -Is)"
