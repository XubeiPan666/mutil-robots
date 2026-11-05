"""Exercise the real transform factory without importing the GPU training stack."""

import ast
import dataclasses
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


class ModelType(Enum):
    PI0 = 1
    PI05 = 2


class Pi0Config:
    model_type = ModelType.PI05
    max_token_len = 200
    action_dim = 32
    discrete_state_input = True


class MultiAgentConfig(Pi0Config):
    pass


class ModelTransformFactoryTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1] / "src/openpi/training/config.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        factory = next(node for node in tree.body
                       if isinstance(node, ast.ClassDef) and node.name == "ModelTransformFactory")
        self.transforms = Mock()
        self.tokenizers = Mock()
        namespace = {
            "dataclasses": dataclasses,
            "GroupFactory": object,
            "_model": SimpleNamespace(BaseModelConfig=object, ModelType=ModelType),
            "_transforms": self.transforms,
            "_tokenizer": self.tokenizers,
            "pi0_config": SimpleNamespace(Pi0Config=Pi0Config),
            "Pi0MultiAgentAttnConfig": MultiAgentConfig,
        }
        # No TwinVLA symbol is supplied: this reproduces the released module's
        # globals while exercising its actual class body and PI05 branch.
        module = ast.Module(body=[factory], type_ignores=[])
        exec(compile(module, str(path), "exec"), namespace)
        self.factory = namespace["ModelTransformFactory"]()

    def test_plain_pi05_builds_discrete_state_tokenization(self):
        result = self.factory(Pi0Config())
        self.assertIs(result, self.transforms.Group.return_value)
        self.tokenizers.PaligemmaTokenizer.assert_called_once_with(200)
        self.transforms.TokenizePrompt.assert_called_once_with(
            self.tokenizers.PaligemmaTokenizer.return_value, discrete_state_input=True,
        )
        self.transforms.ResizeImages.assert_called_once_with(224, 224)
        self.transforms.PadStatesAndActions.assert_called_once_with(32)

    def test_multi_agent_pi05_is_still_supported(self):
        self.factory(MultiAgentConfig())
        self.transforms.TokenizePrompt.assert_called_once()

    def test_unrelated_pi05_config_is_rejected(self):
        with self.assertRaises(AssertionError):
            self.factory(SimpleNamespace(model_type=ModelType.PI05))


if __name__ == "__main__":
    unittest.main()
