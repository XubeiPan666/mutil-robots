"""Validate the plain Pi0.5 six-task baseline without loading a model or W&B."""

import argparse
import dataclasses
import importlib.util
import json
import os
from pathlib import Path
import sys


CONFIG = "pi05_two_arms_multi_task"
REFERENCE = "aevla_pi05_skill_lora_two_arms_multitask"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_dataset_checker():
    scripts = Path(__file__).resolve().parents[1]
    for name in ("paper6_AE-VLA", "paper6"):
        path = scripts / name / "check_training.py"
        if path.is_file():
            sys.path.insert(0, str(path.parent))
            spec = importlib.util.spec_from_file_location("paper6_dataset_checker", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module.check_dataset
    raise FileNotFoundError("Missing six-task dataset checker in paper6_AE-VLA or paper6")


def check_pair(cfg, reference, root, params, *, reference_params=None):
    """Fail closed if settings affecting the baseline comparison have drifted."""
    root, params = Path(root).resolve(), Path(params).resolve()
    reference_params = params if reference_params is None else Path(reference_params).resolve()
    for candidate, expected_params in ((cfg, params), (reference, reference_params)):
        require(Path(candidate.data.repo_id).resolve() == root, "Wrong six-task dataset")
        assets = Path(candidate.data.assets.assets_dir) / candidate.data.assets.asset_id
        require(assets.resolve() == root, "Wrong normalization assets")
        require(Path(candidate.weight_loader.params_path).resolve() == expected_params and expected_params.is_dir(),
                "Expected the matching cached base parameters")

    expected = dict(seed=42, batch_size=32, num_train_steps=30000, num_workers=1, fsdp_devices=2)
    for name, value in expected.items():
        require(getattr(cfg, name) == getattr(reference, name) == value, f"Changed setting: {name}")
    for name in ("lr_schedule", "optimizer", "ema_decay", "freeze_filter"):
        require(getattr(cfg, name) == getattr(reference, name), f"Unmatched setting: {name}")
    for name, value in dict(action_dim=32, action_horizon=50, max_token_len=200).items():
        require(getattr(cfg.model, name) == getattr(reference.model, name) == value,
                f"Changed model dimensions: {name}")

    # Stats computation uses repack + data transforms, NOT model transforms
    # (tokenization/SkillLoRA labels). For the two-arm atom path, PI0/PI05
    # select the same image/mask branch and do not change state/actions.
    left, right = cfg.data.data_transforms, reference.data.data_transforms
    require(type(left) is type(right), "Different data transform factories")
    require(left.use_atom_version and left.num_agents == 2 and left.single_arm_id is None,
            "Expected the unified two-arm atom data path")
    require(not left.use_robotwin_format and not left.use_biso100_format
            and not left.needs_biso100_to_atom, "Unexpected data format")
    require(left.model_type_override.name == "PI0" and right.model_type_override.name == "PI05",
            "Unexpected model type overrides; review normalization compatibility")
    require(dataclasses.replace(left, model_type_override=right.model_type_override) == right,
            "Data transforms differ; do not reuse normalization without review")
    for name in ("shuffle_prob", "image_mask_prob", "image_noise_prob", "wait_augment_prob"):
        require(getattr(left, name) == 0, f"Unexpected augmentation: {name}")
    require(cfg.data.base_config == reference.data.base_config, "Base data settings differ")
    return expected


def check_config(root):
    from openpi.models.pi0_config import Pi0Config
    from openpi.training import config

    cfg, reference = config.get_config(CONFIG), config.get_config(REFERENCE)
    require(type(cfg.model) is Pi0Config and cfg.model.pi05 and cfg.model.discrete_state_input,
            "Expected standard Pi0.5, not an AE-VLA variant")
    require(type(cfg.data.model_transforms) is config.ModelTransformFactory,
            "Expected standard Pi0.5 tokenization without SkillLoRA labels")
    settings = check_pair(cfg, reference, root, os.environ["FUTURE_PI05_PARAMS"])
    # Exercise the actual factory too: comparing its type alone misses broken
    # names in __call__ until the GPU training process starts.
    cfg.data.create(cfg.assets_dirs, cfg.model)
    print(json.dumps({"config": CONFIG, "reference_config": REFERENCE, "settings": settings,
                      "normalization": "Reuse six-task stats: equivalent state/action transforms",
                      "data_config_creation": "passed",
                      "save_interval_cli": 5000, "keep_period_cli": 5000}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(load_dataset_checker()(args.dataset), indent=2))
    check_config(args.dataset)
    print("Six-task plain Pi0.5 preflight passed; no model or W&B run started.")
