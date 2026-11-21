"""Read-only V1 configuration/assets check; no model allocation or training."""

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        choices=[
            "pi05_intent_ar_r1_no_comm_fixed_order",
            "pi05_intent_ar_r2_comm_fixed_order",
            "pi05_intent_ar_r3_comm_random_order",
        ],
        required=True,
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--stats-dir", type=Path, required=True)
    args = parser.parse_args()
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["FUTURE_DATASET_DIR"] = str(args.dataset.resolve())
    os.environ["FUTURE_INTENT_STATS_DIR"] = str(args.stats_dir.resolve())
    # Path environment variables must be set before configuration import.
    from openpi.shared.intent_ar_assets import check_shared_stats  # noqa: PLC0415
    from openpi.training.config import get_config  # noqa: PLC0415

    config = get_config(args.config)
    manifest = check_shared_stats(args.stats_dir, args.dataset, config.model.action_horizon)
    observation, actions = config.model.inputs_spec(batch_size=2)
    print(
        json.dumps(
            {
                "config": config.name,
                "communicate": config.model.communicate,
                "random_training_order": config.model.random_order,
                "inference_order": config.model.inference_order,
                "message_tokens": config.model.message_tokens,
                "paligemma_variant": config.model.paligemma_variant,
                "action_expert_variant": config.model.action_expert_variant,
                "vision_tower_frozen": config.freeze_filter(
                    ("PaliGemma", "img", "encoder", "kernel"), object()
                ),
                "ema_decay": config.ema_decay,
                "fsdp_devices": config.fsdp_devices,
                "global_batch_size": config.batch_size,
                "per_arm_action_dim": config.model.action_dim,
                "grouped_state_spec": observation.state.shape,
                "grouped_action_spec": actions.shape,
                "dataset": str(args.dataset.resolve()),
                "stats_dir": str(args.stats_dir.resolve()),
                "stats_sha256": manifest["norm_stats_sha256"],
            },
            indent=2,
        )
    )
    print("Intent AR config/assets check passed; no model, tokenizer or W&B run started.")


if __name__ == "__main__":
    main()
