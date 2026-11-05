"""Read-only six-task preflight; no model allocation or W&B run is started."""

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path

from prepare_dataset import ANCHORS, load_lines


CONFIG = "aevla_pi05_skill_lora_two_arms_multitask"
REVISION = "447112d1c5b2ec10f7ad34ed438532c062bc51bc"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_dataset(root):
    root = Path(root).resolve()
    info = json.loads((root / "meta/info.json").read_text())
    manifest = json.loads((root / "paper6_manifest.json").read_text())
    episodes = load_lines(root / "meta/episodes.jsonl")
    require(info["total_episodes"] == len(episodes) == 600, "Expected 600 episodes")
    require(sorted(e["episode_index"] for e in episodes) == list(range(600)), "Bad episode IDs")
    require(sum(e["length"] for e in episodes) == info["total_frames"], "Frame totals disagree")
    families = Counter()
    for episode in episodes:
        text = "\n".join(episode["tasks"])
        matches = [name for name, anchor in ANCHORS.items() if anchor in text]
        require(len(matches) == 1 and matches[0] != "handover", "Unexpected episode family")
        families[matches[0]] += 1
        path = root / "data/chunk-000" / f"episode_{episode['episode_index']:06d}.parquet"
        require(path.is_file() and path.stat().st_size > 0, f"Missing/empty episode: {path}")
    expected = {name: 100 for name in ANCHORS if name != "handover"}
    require(dict(families) == manifest["families"] == expected, "Six-family counts disagree")
    require(manifest["source_revision"] == REVISION, "Unexpected public data revision")
    require(manifest["selected_episodes"] == 600, "Wrong selection manifest")
    require(manifest["total_frames"] == info["total_frames"], "Manifest frame count disagrees")
    require(manifest["episode_map"] == {str(i + 100): i for i in range(600)}, "Wrong episode mapping")

    tasks = load_lines(root / "meta/tasks.jsonl")
    labels = load_lines(root / "meta/skill_labels.jsonl")
    ids = list(range(info["total_tasks"]))
    require(sorted(t["task_index"] for t in tasks) == ids, "Bad task IDs")
    require(sorted(t["task_index"] for t in labels) == ids, "Bad skill-label IDs")
    for label in labels:
        skills = label["skill_ids"]
        require(len(skills) == 2 and all(type(i) is int and 0 <= i < 10 for i in skills),
                "Invalid two-arm skill labels")

    stats = json.loads((root / "norm_stats.json").read_text())["norm_stats"]
    for group in ("state", "actions"):
        for field in ("mean", "std", "q01", "q99"):
            values = stats[group][field]
            require(len(values) == 32 and all(isinstance(x, (int, float)) and math.isfinite(x) for x in values),
                    f"Invalid {group}/{field} normalization")
        require(all(x >= 0 for x in stats[group]["std"]), f"Negative {group} std")
        require(all(a <= b for a, b in zip(stats[group]["q01"], stats[group]["q99"])),
                f"Invalid {group} quantiles")
    return {"dataset": str(root), "episodes": 600, "frames": info["total_frames"],
            "families": dict(families), "norm_stats": str(root / "norm_stats.json")}


def check_config(root):
    # Resolve factories without creating tokenizers, datasets, models or a W&B run.
    from openpi.training import config

    cfg = config.get_config(CONFIG)
    root = Path(root).resolve()
    require(Path(cfg.data.repo_id).resolve() == root, "Config still points at another dataset")
    require(Path(cfg.data.model_transforms.skill_labels_path).resolve() == root / "meta/skill_labels.jsonl",
            "Config still points at other skill labels")
    assets = Path(cfg.data.assets.assets_dir) / cfg.data.assets.asset_id
    require(assets.resolve() == root, "Config still points at other normalization assets")
    params = Path(os.environ["FUTURE_PI05_PARAMS"]).resolve()
    require(Path(cfg.weight_loader.params_path).resolve() == params and params.is_dir(),
            "Config must initialize from the cached Pi0.5 base")
    require(cfg.batch_size == 32 and cfg.num_train_steps == 30000 and cfg.fsdp_devices == 2,
            "Unexpected training defaults")
    require(cfg.model.use_arm_wise_prefix_attn and cfg.model.use_arm_wise_suffix_attn
            and not cfg.model.use_latent_collab_tokens, "Unexpected AE-VLA architecture")
    print("Resolved config:", cfg.name)
    print("Skill labels:", cfg.data.model_transforms.skill_labels_path)
    print("Base parameters:", params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--check-config", action="store_true")
    args = parser.parse_args()
    print(json.dumps(check_dataset(args.dataset), indent=2))
    if args.check_config:
        check_config(args.dataset)
    print("Six-task training preflight passed (metadata, files, labels, normalization).")
