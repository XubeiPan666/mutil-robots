#!/bin/bash -l
# Download/cache Pi0 base and perform CPU-only validation; no training submission.
set -eo pipefail
source "$(ws_find mutil-robots)/load_env.sh"
export PYTHONNOUSERSITE=1
export JAX_PLATFORMS=cpu
cd "$PROJECT_WS/code/mutil-robots"
export OPENPI_DATA_HOME="$PROJECT_WS/cache/openpi"
export FUTURE_DATA_ROOT="$PROJECT_WS/datasets/paper6"
export FUTURE_DATASET_DIR="$FUTURE_DATA_ROOT/multi_task_combined"
export FUTURE_ASSETS_DIR="$FUTURE_DATA_ROOT"
export FUTURE_PI0_PARAMS="$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi0_base/params"
export FUTURE_PI05_PARAMS="$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_base/params"
uv run --frozen python - <<'PY'
import os
from pathlib import Path
from openpi.shared import download

path = download.maybe_download("gs://openpi-assets/checkpoints/pi0_base/params", gs={"token": "anon"})
assert path.resolve() == Path(os.environ["FUTURE_PI0_PARAMS"]).resolve()
assert path.is_dir() and any(path.iterdir()), "Pi0 parameter directory is empty"
print("Pi0 base ready:", path, flush=True)
PY
uv run --frozen python scripts/paper6_pi0/check_training.py --dataset "$FUTURE_DATASET_DIR"
