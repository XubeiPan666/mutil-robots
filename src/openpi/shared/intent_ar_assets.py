"""Validate completed shared-arm normalization before new training (no ML imports)."""

import hashlib
import json
from pathlib import Path

import numpy as np


def check_shared_stats(directory, dataset, horizon=50):
    directory, dataset = Path(directory), Path(dataset)
    manifest_path = directory / "stats_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("Set FUTURE_INTENT_STATS_DIR to a completed shared single-arm stats directory")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest.get("action_horizon") != horizon:
        raise ValueError("Shared stats are incomplete or use a different action horizon")
    if Path(manifest["dataset"]).resolve() != dataset.resolve():
        raise ValueError("Shared stats belong to another dataset")
    info = json.loads((dataset / "meta/info.json").read_text(encoding="utf-8"))
    frames = info["total_frames"]
    if (manifest.get("processed_frames"), manifest.get("state_vectors"), manifest.get("action_vectors")) != (
        frames,
        2 * frames,
        2 * frames * horizon,
    ):
        raise ValueError("Shared stats frame/vector counts do not match this dataset")
    for name, expected in manifest["metadata_sha256"].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Dataset metadata changed since normalization: {name}")
    path = directory / "norm_stats.json"
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["norm_stats_sha256"]:
        raise ValueError("Shared normalization file hash mismatch")
    stats = json.loads(path.read_text(encoding="utf-8"))["norm_stats"]
    for key in ("state", "actions"):
        for field in ("mean", "std", "q01", "q99"):
            values = np.asarray(stats[key][field])
            if values.shape != (32,) or not np.isfinite(values).all() or np.any(values[8:] != 0):
                raise ValueError("Expected finite shared 8D statistics with 24 zero padding slots")
        if np.any(np.asarray(stats[key]["std"]) < 0) or np.any(
            np.asarray(stats[key]["q01"]) > np.asarray(stats[key]["q99"])
        ):
            raise ValueError("Invalid standard deviation or quantile ordering")
    return manifest
