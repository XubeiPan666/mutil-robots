"""Check that the baseline preflight rejects changes invalidating comparison."""

from dataclasses import dataclass, replace
from enum import Enum
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import unittest


PATH = Path(__file__).resolve().parents[1] / "scripts/paper6_pi05/check_training.py"
SPEC = importlib.util.spec_from_file_location("pi05_check", PATH)
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


class ModelType(Enum):
    PI0 = 1
    PI05 = 2


@dataclass(frozen=True)
class Transforms:
    model_type_override: ModelType = ModelType.PI0
    use_atom_version: bool = True
    num_agents: int = 2
    single_arm_id: object = None
    use_robotwin_format: bool = False
    use_biso100_format: bool = False
    needs_biso100_to_atom: bool = False
    shuffle_prob: float = 0
    image_mask_prob: float = 0
    image_noise_prob: float = 0
    wait_augment_prob: float = 0
    use_delta_joint_actions: bool = True


class Pi05PreflightTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "multi_task_combined"
        self.params = Path(self.tmp.name) / "params"
        self.params.mkdir()

        def config(model_type):
            return NS(seed=42, batch_size=32, num_train_steps=30000, num_workers=1,
                      fsdp_devices=2, lr_schedule=(1000, 5e-5), optimizer="adamw",
                      ema_decay=0.99, freeze_filter=None,
                      model=NS(action_dim=32, action_horizon=50, max_token_len=200),
                      data=NS(repo_id=str(self.root),
                              assets=NS(assets_dir=str(self.root.parent), asset_id=self.root.name),
                              data_transforms=Transforms(model_type_override=model_type),
                              base_config={"prompt_from_task": True}),
                      weight_loader=NS(params_path=str(self.params)))
        self.cfg = config(ModelType.PI0)
        self.ref = config(ModelType.PI05)

    def check(self):
        return CHECK.check_pair(self.cfg, self.ref, self.root, self.params)

    def test_matched_config_accepts_image_branch_override(self):
        self.assertEqual(self.check()["seed"], 42)

    def test_explicit_separate_backbone_parameters(self):
        reference_params = Path(self.tmp.name) / "pi05_params"
        reference_params.mkdir()
        self.ref.weight_loader.params_path = str(reference_params)
        # Default still requires the same backbone for the original Pi0.5 run.
        with self.assertRaisesRegex(ValueError, "base parameters"):
            self.check()
        self.assertEqual(CHECK.check_pair(self.cfg, self.ref, self.root, self.params,
                                         reference_params=reference_params)["seed"], 42)
        self.cfg.weight_loader.params_path = str(reference_params)
        with self.assertRaisesRegex(ValueError, "base parameters"):
            CHECK.check_pair(self.cfg, self.ref, self.root, self.params,
                             reference_params=reference_params)

    def test_changed_delta_actions_rejected(self):
        self.cfg.data.data_transforms = replace(self.cfg.data.data_transforms, use_delta_joint_actions=False)
        with self.assertRaisesRegex(ValueError, "transforms differ"):
            self.check()

    def test_wrong_dataset_assets_and_base_weights_rejected(self):
        for obj, key in ((self.cfg.data, "repo_id"), (self.cfg.data.assets, "assets_dir"),
                         (self.cfg.weight_loader, "params_path")):
            with self.subTest(key=key):
                old = getattr(obj, key)
                setattr(obj, key, str(self.root / "wrong"))
                with self.assertRaises(ValueError):
                    self.check()
                setattr(obj, key, old)

    def test_seed_and_optimizer_drift_rejected(self):
        for key, value in (("seed", 43), ("lr_schedule", (1000, 1e-4)), ("num_workers", 2)):
            with self.subTest(key=key):
                old = getattr(self.cfg, key)
                setattr(self.cfg, key, value)
                with self.assertRaises(ValueError):
                    self.check()
                setattr(self.cfg, key, old)

    def test_different_horizon_rejected(self):
        self.cfg.model.action_horizon = 25
        with self.assertRaisesRegex(ValueError, "dimensions"):
            self.check()

    def test_shared_metadata_checker_resolves(self):
        self.assertTrue(callable(CHECK.load_dataset_checker()))


if __name__ == "__main__":
    unittest.main()
