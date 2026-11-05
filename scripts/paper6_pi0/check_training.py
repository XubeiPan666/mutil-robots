"""Read-only preflight for the standard Pi0 six-task baseline."""

import argparse
import importlib.util
import json
import os
from pathlib import Path


CONFIG = "pi0_two_arms_multi_task"


def require_pi0(model, model_class, params, reference_params):
    if type(model) is not model_class or model.pi05 or model.discrete_state_input:
        raise ValueError("Expected standard Pi0 with continuous state, not Pi0.5 or AE-VLA")
    if Path(params).resolve() == Path(reference_params).resolve():
        raise ValueError("Pi0 must use its own base weights, not the Pi0.5 base")


def main(root):
    # Reuse the matched-baseline guards; keep existing Pi0.5 defaults unchanged.
    path = Path(__file__).resolve().parents[1] / "paper6_pi05/check_training.py"
    spec = importlib.util.spec_from_file_location("pi05_baseline_checks", path)
    checks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checks)
    print(json.dumps(checks.load_dataset_checker()(root), indent=2))

    from openpi.models.pi0_config import Pi0Config
    from openpi.training import config

    cfg = config.get_config(CONFIG)
    reference = config.get_config(checks.REFERENCE)
    params, ref_params = os.environ["FUTURE_PI0_PARAMS"], os.environ["FUTURE_PI05_PARAMS"]
    require_pi0(cfg.model, Pi0Config, params, ref_params)
    checks.require(type(cfg.data.model_transforms) is config.ModelTransformFactory,
                   "Expected standard Pi0 tokenization without SkillLoRA")
    settings = checks.check_pair(cfg, reference, root, params, reference_params=ref_params)
    print(json.dumps({"config": CONFIG, "reference_config": checks.REFERENCE,
                      "settings": settings, "base_params": params,
                      "reference_base_params": ref_params,
                      "normalization": "Reuse six-task stats: equivalent state/action transforms",
                      "normalization_method": "Pi0 uses mean/std; Pi0.5 uses quantiles",
                      "save_interval_cli": 5000, "keep_period_cli": 5000}, indent=2))
    print("Six-task plain Pi0 preflight passed; no model or W&B run started.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    main(parser.parse_args().dataset)
