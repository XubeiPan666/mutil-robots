"""Pool both arms of paper-six data after the existing AE-VLA delta transform.

Read-only preflight by default. --apply writes to a NEW output directory only.
Uses the project's LeRobot reader, horizon/padding, image-type fix, transforms,
and RunningStats; does not construct a model, tokenizer or W&B run.
"""

import argparse
import dataclasses
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from openpi.shared.single_arm import collate_pairs, pad_stat_vector, split_delta_pair


REFERENCE = "aevla_pi05_skill_lora_two_arms_multitask"
REPO = Path(__file__).resolve().parents[2]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_reference(cfg, root):
    """Reject interface drift rather than silently generating incompatible assets."""
    require(Path(cfg.data.repo_id).resolve() == root, "Reference dataset path does not match --dataset")
    require(cfg.model.action_dim == 32 and cfg.model.action_horizon == 50, "Expected 32D / horizon 50")
    factory = cfg.data.data_transforms
    expected = {
        "num_agents": 2, "joints_per_agent": 7, "action_dim_per_agent": 8,
        "use_atom_version": True, "single_arm_id": None,
        "use_delta_joint_actions": True, "use_phase_aware_delta": False,
        "use_robotwin_format": False, "use_biso100_format": False, "needs_biso100_to_atom": False,
        "shuffle_prob": 0.0, "image_mask_prob": 0.0, "image_noise_prob": 0.0, "wait_augment_prob": 0.0,
    }
    for key, value in expected.items():
        require(getattr(factory, key) == value, f"Unsupported reference transform: {key}")
    require(tuple(cfg.data.base_config.action_sequence_keys) == ("actions",), "Unexpected action sequence keys")


def validate_output(root, output):
    require(output != root and root not in output.parents, "Output must be outside the source dataset")
    require(not output.exists(), f"Output already exists; choose a new directory: {output}")


def check_paper6(root):
    # Reuse the dataset audit, including revision, six families and episode map.
    path = REPO / "scripts/paper6_AE-VLA/check_training.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("intent_ar_paper6_check", path)
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    return checker.check_dataset(root)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class NumericArmDataset:
    """Drop images after the trusted transforms; never substitute corrupt frames."""

    def __init__(self, dataset, transform, horizon):
        self.dataset, self.transform, self.horizon = dataset, transform, horizon

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        row = self.transform(self.dataset[index])
        return split_delta_pair(row["state"], row["actions"], horizon=self.horizon)


def accumulate(loader, accumulators, horizon):
    frames = 0
    for batch in loader:
        n = len(batch["state"])
        require(batch["state"].shape == (n, 2, 8), "Bad pooled state batch")
        require(batch["actions"].shape == (n, 2, horizon, 8), "Bad pooled action batch")
        for key, accumulator in accumulators.items():
            # float64 avoids cancellation in running variance of joint positions.
            values = np.asarray(batch[key], dtype=np.float64)
            require(np.isfinite(values).all(), f"Non-finite {key} batch")
            accumulator.update(values)
        frames += n
    return frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=1)
    args = parser.parse_args()
    root, output = args.dataset.resolve(), args.output.resolve()
    validate_output(root, output)
    require(args.batch_size > 0 and args.num_workers >= 0, "Invalid batch size / workers")
    require(root.name == "multi_task_combined", "Expected the prepared paper-six directory")

    # Configuration paths are evaluated at import time. Do not inherit another dataset.
    os.environ["FUTURE_DATA_ROOT"] = str(root.parent)
    os.environ["FUTURE_DATASET_DIR"] = str(root)
    os.environ["FUTURE_ASSETS_DIR"] = str(root.parent)
    os.environ["JAX_PLATFORMS"] = "cpu"
    from openpi.training import config
    from openpi.training.config_utils import without_augmentation

    summary = check_paper6(root)
    cfg = without_augmentation(config.get_config(REFERENCE))
    validate_reference(cfg, root)
    summary.update({
        "output": str(output), "reference_config": REFERENCE, "action_horizon": cfg.model.action_horizon,
        "state_vectors": summary["frames"] * 2,
        "action_vectors": summary["frames"] * 2 * cfg.model.action_horizon,
        "layout": "slots 0:7 joint delta actions; slot 7 absolute gripper; slots 8:32 zero padding",
        "state_semantics": "current absolute joint positions and gripper; shared eight-slot normalization",
        "chunk_semantics": "existing LeRobot future chunks and end-of-episode padding; all H positions counted",
        "coverage": "all frames, both arms, no shuffle, no augmentation, drop_last=False",
        "quantiles": "project RunningStats 5000-bin approximate quantiles (not exact sorted quantiles)",
    })
    print(json.dumps(summary, indent=2), flush=True)
    if not args.apply:
        print("Preflight passed; no dataset decoded, output created, model loaded or W&B run started.", flush=True)
        return

    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from openpi.shared import normalize
    from openpi.training import data_loader
    from openpi import transforms

    # Do not call cfg.data.create(): it also constructs model/tokenizer transforms.
    # Numeric stats need only the existing repack + data transforms. Prompt is not
    # needed when all stochastic transforms are off; omit its corruption-skipping wrapper.
    data_cfg = dataclasses.replace(cfg.data.base_config, repo_id=str(root), prompt_from_task=False, norm_stats=None)
    group = cfg.data.data_transforms(cfg.model)
    delta = [item for item in group.inputs if isinstance(item, transforms.DeltaActions)]
    expected_mask = [True] * 7 + [False] + [True] * 7 + [False]
    require(len(delta) == 1 and list(delta[0].mask) == expected_mask, "Expected exactly one standard joint-delta transform")

    tracked = [root / "paper6_manifest.json", root / "norm_stats.json", *sorted((root / "meta").glob("*.json*"))]
    before = {str(path): digest(path) for path in tracked}
    manifest = dict(summary, status="running", started_at=datetime.now(timezone.utc).isoformat(), metadata_sha256=before)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=False)
    manifest["training_repo_commit"] = revision.stdout.strip() if revision.returncode == 0 else "unknown"
    manifest["implementation_sha256"] = {
        name: digest(REPO / name) for name in (
            "scripts/intent_ar/compute_norm_stats.py", "src/openpi/shared/single_arm.py",
            "src/openpi/shared/normalize.py", "src/openpi/transforms.py",
            "src/openpi/policies/droid_policy.py", "src/openpi/training/data_loader.py",
            "src/openpi/training/config.py", "uv.lock",
        )
    }
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = output / "stats_manifest.json"

    def write_manifest():
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    write_manifest()
    try:
        dataset = data_loader.create_torch_dataset(data_cfg, cfg.model.action_horizon, cfg.model)
        require(len(dataset) == summary["frames"], "Loaded dataset frame count disagrees with metadata")
        numeric = NumericArmDataset(
            dataset, transforms.compose([*data_cfg.repack_transforms.inputs, *group.inputs]), cfg.model.action_horizon
        )
        loader = DataLoader(numeric, batch_size=args.batch_size, shuffle=False, drop_last=False,
                            num_workers=args.num_workers, collate_fn=collate_pairs)
        accumulators = {key: normalize.RunningStats() for key in ("state", "actions")}
        seen = accumulate(tqdm(loader, desc="Shared single-arm stats", total=len(loader)),
                          accumulators, cfg.model.action_horizon)
        require(seen == summary["frames"], "Not all source frames were processed")
        result = {}
        for key, accumulator in accumulators.items():
            stat = accumulator.get_statistics()
            fields = {field: pad_stat_vector(getattr(stat, field)) for field in ("mean", "std", "q01", "q99")}
            require(np.all(fields["std"] >= 0) and np.all(fields["q01"] <= fields["q99"]), "Invalid statistics")
            result[key] = normalize.NormStats(**fields)
        require(before == {str(path): digest(path) for path in tracked}, "Source metadata/statistics changed during run")
        # Publish a complete file only, after all numerical and source checks pass.
        temp = output / "norm_stats.json.tmp"
        temp.write_text(normalize.serialize_json(result), encoding="utf-8")
        normalize.deserialize_json(temp.read_text(encoding="utf-8"))
        temp.replace(output / "norm_stats.json")
        manifest.update(status="complete", processed_frames=seen, finished_at=datetime.now(timezone.utc).isoformat(),
                        norm_stats_sha256=digest(output / "norm_stats.json"))
        write_manifest()
        print(f"Shared single-arm normalization complete: {output / 'norm_stats.json'}", flush=True)
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_manifest()
        raise


if __name__ == "__main__":
    main()
