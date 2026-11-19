"""Exercise launcher control flow with mocked cluster/ML commands; never train."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/intent_ar/train.sh"
BASH = "C:/Program Files/Git/bin/bash.exe" if os.name == "nt" else shutil.which("bash")
pytestmark = pytest.mark.skipif(not BASH or not Path(BASH).is_file(), reason="Bash required")
CONFIGS = [
    "pi05_intent_ar_r1_no_comm_fixed_order",
    "pi05_intent_ar_r2_comm_fixed_order",
    "pi05_intent_ar_r3_comm_random_order",
]


def shell_path(path):
    value = str(path.resolve()).replace("\\", "/")
    return f"/{value[0].lower()}{value[2:]}" if os.name == "nt" else value


@pytest.fixture
def launch(tmp_path):
    workspace = tmp_path / "workspace"
    repo = workspace / "code/mutil-robots"
    repo.mkdir(parents=True)
    copied = repo / "train.sh"
    shutil.copyfile(SCRIPT, copied)
    dataset = workspace / "datasets/paper6/multi_task_combined"
    dataset.mkdir(parents=True)
    (dataset / "paper6_manifest.json").write_text("{}")
    stats = workspace / "stats"
    stats.mkdir()
    for name in ("stats_manifest.json", "norm_stats.json"):
        (stats / name).write_text("{}")
    (workspace / "cache/openpi/openpi-assets/checkpoints/pi05_base/params").mkdir(parents=True)
    (workspace / "load_env.sh").write_text('export PROJECT_WS="$TEST_WORKSPACE"\n')
    mock = tmp_path / "mock.sh"
    mock.write_text(
        """
ws_find() { printf '%s\n' "$TEST_WORKSPACE"; }
git() { echo mock-git; }
nvidia-smi() { echo mock-gpu; }
uv() {
    case "$*" in
        *scripts/intent_ar/check_model.py*) return "${TEST_PREFLIGHT_EXIT:-0}" ;;
        *scripts/train.py*) printf '%s\n' "$@" > "$TEST_CALLS" ;;
        'run --frozen python -') cat >/dev/null; return "${TEST_GPU_EXIT:-0}" ;;
        *) echo "Unexpected uv invocation" >&2; return 99 ;;
    esac
}
""",
        encoding="utf-8",
    )
    calls = tmp_path / "calls.txt"
    env = dict(
        os.environ,
        BASH_ENV=shell_path(mock),
        TEST_WORKSPACE=shell_path(workspace),
        TEST_CALLS=shell_path(calls),
        SLURM_JOB_ID="123",
        FUTURE_INTENT_STATS_DIR=shell_path(stats),
        WANDB_API_KEY="test-placeholder",
        WANDB_ENTITY="test-team",
        WANDB_PROJECT="test-project",
    )

    def run(*args, overrides=None):
        current = env | (overrides or {})
        result = subprocess.run(
            [BASH, shell_path(copied), *args],
            check=False,
            env=current,
            capture_output=True,
            text=True,
            timeout=30,
            encoding="utf-8",
        )
        return result, calls.read_text(encoding="utf-8").splitlines() if calls.exists() else []

    return run


@pytest.mark.parametrize("index", range(3))
@pytest.mark.parametrize("smoke", [False, True])
def test_select_config_and_training_mode(launch, index, smoke):
    selection = CONFIGS[index] if smoke else f"r{index + 1}"
    result, args = launch(selection, *(["--smoke"] if smoke else []))
    assert result.returncode == 0, result.stderr
    assert args[4] == CONFIGS[index]
    for flag, value in {
        "--batch-size": "2" if smoke else "32",
        "--num-train-steps": "20" if smoke else "30000",
        "--save-interval": "20" if smoke else "5000",
        "--keep-period": "20" if smoke else "5000",
        "--fsdp-devices": "2",
        "--seed": "42",
    }.items():
        assert args[args.index(flag) + 1] == value
    assert "--wandb-enabled" in args
    assert "--resume" not in args
    assert "--overwrite" not in args
    assert "test-placeholder" not in result.stdout + result.stderr


@pytest.mark.parametrize(
    "overrides",
    [
        {"SLURM_JOB_ID": ""},
        {"FUTURE_INTENT_STATS_DIR": ""},
        {"WANDB_API_KEY": ""},
        {"TEST_PREFLIGHT_EXIT": "7"},
        {"TEST_GPU_EXIT": "8"},
    ],
)
def test_failed_prerequisite_never_starts_training(launch, overrides):
    result, args = launch("r3", overrides=overrides)
    assert result.returncode != 0
    assert not args


@pytest.mark.parametrize("args", [(), ("r4",), ("r3", "--overwrite"), ("r1", "--smoke", "extra")])
def test_invalid_arguments_never_start_training(launch, args):
    result, calls = launch(*args)
    assert result.returncode == 2
    assert not calls


def test_help_without_cluster(launch):
    result, calls = launch("--help", overrides={"SLURM_JOB_ID": ""})
    assert result.returncode == 0
    assert "Usage:" in result.stdout
    assert not calls


def test_bash_syntax():
    subprocess.run([BASH, "-n", shell_path(SCRIPT)], check=True, timeout=10)
