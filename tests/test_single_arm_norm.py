"""Numeric mapping and accounting tests; no JAX/GPU/server dataset required."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import unittest

import numpy as np

from openpi.shared.single_arm import collate_pairs, pad_stat_vector, split_delta_pair


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("single_arm_stats", ROOT / "scripts/intent_ar/compute_norm_stats.py")
SCRIPT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCRIPT)


class Recorder:
    def __init__(self):
        self.batches = []

    def update(self, batch):
        self.batches.append(batch.reshape(-1, 8))

    @property
    def values(self):
        return np.concatenate(self.batches)


class SharedArmTests(unittest.TestCase):
    def sample(self, offset=0):
        # Already delta-transformed joints, absolute grippers, zero padding.
        state = np.zeros(32, dtype=np.float32)
        state[:8] = np.arange(8) + offset
        state[8:16] = np.arange(8) + 100 + offset
        actions = np.zeros((3, 32), dtype=np.float32)
        actions[:, :7] = np.arange(3)[:, None] + 1 + offset
        actions[:, 8:15] = np.arange(3)[:, None] + 11 + offset
        actions[:, 7] = [0.1, 0.2, 0.3]
        actions[:, 15] = [0.7, 0.8, 0.9]
        return state, actions

    def test_arm_order_and_absolute_gripper_preserved_without_double_delta(self):
        state, actions = self.sample()
        original = actions.copy()
        split = split_delta_pair(state, actions, horizon=3)
        np.testing.assert_array_equal(split["state"][0], np.arange(8))
        np.testing.assert_array_equal(split["state"][1], np.arange(8) + 100)
        np.testing.assert_array_equal(split["actions"][0, :, 0], [1, 2, 3])
        np.testing.assert_array_equal(split["actions"][1, :, 0], [11, 12, 13])
        np.testing.assert_allclose(split["actions"][:, :, 7], [[0.1, 0.2, 0.3], [0.7, 0.8, 0.9]])
        split["actions"][0, 0, 0] = -999
        np.testing.assert_array_equal(actions, original)

    def test_bad_shape_padding_and_nonfinite_rejected(self):
        state, actions = self.sample()
        with self.assertRaises(ValueError):
            split_delta_pair(state[:16], actions, horizon=3)
        with self.assertRaises(ValueError):
            split_delta_pair(state, actions, horizon=50)
        state[16] = 1
        with self.assertRaisesRegex(ValueError, "padding"):
            split_delta_pair(state, actions, horizon=3)
        state[16] = 0
        actions[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            split_delta_pair(state, actions, horizon=3)

    def test_pooling_counts_and_partial_batch(self):
        samples = [split_delta_pair(*self.sample(i), horizon=3) for i in range(3)]
        accumulators = {key: Recorder() for key in ("state", "actions")}
        seen = SCRIPT.accumulate([collate_pairs(samples[:2]), collate_pairs(samples[2:])], accumulators, 3)
        self.assertEqual(seen, 3)
        self.assertEqual(accumulators["state"].values.shape, (6, 8))
        self.assertEqual(accumulators["actions"].values.shape, (18, 8))
        # Equal pooling of both physical arms, including the final one-scene batch.
        np.testing.assert_array_equal(accumulators["state"].values[:, 0], [0, 100, 1, 101, 2, 102])
        self.assertEqual(accumulators["state"].values[:, 0].mean(), 51)
        self.assertEqual(accumulators["actions"].values[:, 0].mean(), 8)
        self.assertEqual(accumulators["actions"].values.dtype, np.float64)

    def test_padding_does_not_pollute_meaningful_slots(self):
        vector = np.arange(8, dtype=float) + 0.5
        padded = pad_stat_vector(vector)
        self.assertEqual(padded.shape, (32,))
        np.testing.assert_array_equal(padded[:8], vector)
        np.testing.assert_array_equal(padded[8:], np.zeros(24))
        with self.assertRaises(ValueError):
            pad_stat_vector(np.zeros(16))

    def test_numeric_reader_propagates_corruption(self):
        class Broken:
            def __len__(self):
                return 1

            def __getitem__(self, index):
                raise OSError("corrupt frame")

        dataset = SCRIPT.NumericArmDataset(Broken(), lambda row: row, 3)
        with self.assertRaisesRegex(OSError, "corrupt frame"):
            dataset[0]

    def test_output_refuses_source_and_existing_directories(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            root.mkdir()
            for destination in (root, root / "new_stats", Path(temporary)):
                with self.subTest(destination=destination), self.assertRaises(ValueError):
                    SCRIPT.validate_output(root, destination)
            destination = Path(temporary) / "separate_stats"
            SCRIPT.validate_output(root, destination)
            self.assertFalse(destination.exists())

    def test_reference_rejects_delta_and_horizon_drift(self):
        root = Path("dataset").resolve()
        factory = NS(num_agents=2, joints_per_agent=7, action_dim_per_agent=8,
                     use_atom_version=True, single_arm_id=None, use_delta_joint_actions=True,
                     use_phase_aware_delta=False, use_robotwin_format=False, use_biso100_format=False,
                     needs_biso100_to_atom=False, shuffle_prob=0.0, image_mask_prob=0.0,
                     image_noise_prob=0.0, wait_augment_prob=0.0)
        cfg = NS(model=NS(action_dim=32, action_horizon=50),
                 data=NS(repo_id=str(root), data_transforms=factory,
                         base_config=NS(action_sequence_keys=("actions",))))
        SCRIPT.validate_reference(cfg, root)
        for key, value in (("use_delta_joint_actions", False), ("use_phase_aware_delta", True),
                           ("single_arm_id", 0), ("shuffle_prob", 0.5)):
            old = getattr(factory, key)
            setattr(factory, key, value)
            with self.subTest(key=key), self.assertRaises(ValueError):
                SCRIPT.validate_reference(cfg, root)
            setattr(factory, key, old)
        cfg.model.action_horizon = 25
        with self.assertRaises(ValueError):
            SCRIPT.validate_reference(cfg, root)


if __name__ == "__main__":
    unittest.main()
