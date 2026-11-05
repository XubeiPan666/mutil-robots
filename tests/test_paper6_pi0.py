"""Prevent accidental Pi0.5 initialization in the Pi0 comparison."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

PATH = Path(__file__).resolve().parents[1] / "scripts/paper6_pi0/check_training.py"
SPEC = importlib.util.spec_from_file_location("pi0_check", PATH)
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


class Pi0Guards(unittest.TestCase):
    def test_standard_pi0_allowed(self):
        CHECK.require_pi0(SimpleNamespace(pi05=False, discrete_state_input=False),
                          SimpleNamespace, "pi0/params", "pi05/params")

    def test_pi05_model_rejected(self):
        for pi05, discrete in ((True, True), (False, True)):
            with self.subTest(pi05=pi05, discrete=discrete), self.assertRaisesRegex(ValueError, "standard Pi0"):
                CHECK.require_pi0(SimpleNamespace(pi05=pi05, discrete_state_input=discrete),
                                  SimpleNamespace, "pi0/params", "pi05/params")

    def test_pi05_weights_rejected(self):
        with self.assertRaisesRegex(ValueError, "own base weights"):
            CHECK.require_pi0(SimpleNamespace(pi05=False, discrete_state_input=False),
                              SimpleNamespace, "pi05/params", "pi05/params")


if __name__ == "__main__":
    unittest.main()
