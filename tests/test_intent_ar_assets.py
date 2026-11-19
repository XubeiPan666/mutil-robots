import hashlib
import json

import pytest

from openpi.shared.intent_ar_assets import check_shared_stats


@pytest.fixture
def assets(tmp_path):
    dataset, directory = tmp_path / "dataset", tmp_path / "stats"
    (dataset / "meta").mkdir(parents=True)
    directory.mkdir()
    info = dataset / "meta/info.json"
    info.write_text(json.dumps({"total_frames": 3}), encoding="utf-8")
    stats = {
        key: {
            "mean": [0.0] * 32,
            "std": [1.0] * 8 + [0.0] * 24,
            "q01": [-1.0] * 8 + [0.0] * 24,
            "q99": [1.0] * 8 + [0.0] * 24,
        }
        for key in ("state", "actions")
    }
    path = directory / "norm_stats.json"
    path.write_text(json.dumps({"norm_stats": stats}), encoding="utf-8")
    manifest = {
        "status": "complete",
        "dataset": str(dataset),
        "action_horizon": 50,
        "processed_frames": 3,
        "state_vectors": 6,
        "action_vectors": 300,
        "metadata_sha256": {str(info): hashlib.sha256(info.read_bytes()).hexdigest()},
        "norm_stats_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    (directory / "stats_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return dataset, directory, manifest


def test_accepts_completed_shared_stats(assets):
    dataset, directory, _ = assets
    assert check_shared_stats(directory, dataset)["processed_frames"] == 3


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("status", "running"),
        ("action_horizon", 25),
        ("state_vectors", 3),
        ("processed_frames", 2),
        ("norm_stats_sha256", "wrong"),
    ],
)
def test_rejects_incomplete_or_mismatched_assets(assets, key, value):
    dataset, directory, manifest = assets
    manifest[key] = value
    (directory / "stats_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="Shared"):
        check_shared_stats(directory, dataset)


def test_rejects_metadata_drift(assets):
    dataset, directory, _ = assets
    (dataset / "meta/info.json").write_text('{"total_frames": 3, "changed": true}', encoding="utf-8")
    with pytest.raises(ValueError, match="metadata changed"):
        check_shared_stats(directory, dataset)


def test_rejects_old_two_arm_stats_without_manifest(tmp_path):
    with pytest.raises(ValueError, match="FUTURE_INTENT_STATS_DIR"):
        check_shared_stats(tmp_path, tmp_path)
