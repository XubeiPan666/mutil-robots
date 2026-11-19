"""Paired preprocessing, shared normalization and output reconstruction."""

import numpy as np
import pytest

from openpi import transforms
from openpi.policies.intent_ar_policy import JoinOutputs
from openpi.policies.intent_ar_policy import PairInputs
from openpi.policies.intent_ar_policy import TokenizePair
from openpi.policies.intent_ar_policy import UnflattenPair
from openpi.shared.normalize import NormStats


class RecordingTokenizer:
    def __init__(self):
        self.calls = []

    def tokenize(self, prompt, state):
        self.calls.append((prompt, state.copy()))
        return np.full(4, len(self.calls), dtype=np.int32), np.ones(4, dtype=bool)


def test_full_delta_normalize_tokenize_inverse_roundtrip():
    state = np.zeros(32, np.float32)
    state[:16] = np.arange(16) / 10
    target = np.zeros((3, 32), np.float32)
    target[:, :16] = np.arange(3)[:, None] / 20 + state[:16] + 0.2
    original = target.copy()
    row = {"state": state.copy(), "actions": target, "prompt": "Pick blue. | Wait."}
    mask = [True] * 7 + [False] + [True] * 7 + [False]
    row = transforms.DeltaActions(mask)(row)
    row = PairInputs(horizon=3)(row)
    stats = {
        key: NormStats(
            mean=np.zeros(32),
            std=np.ones(32),
            q01=np.r_[np.full(8, -2.0), np.zeros(24)],
            q99=np.r_[np.full(8, 2.0), np.zeros(24)],
        )
        for key in ("state", "actions")
    }
    row = transforms.Normalize(stats, use_quantiles=True)(row)
    tokenizer = RecordingTokenizer()
    row = TokenizePair(tokenizer)(row)
    assert row["state"].shape == (64,)
    assert row["actions"].shape == (3, 64)
    assert [call[0] for call in tokenizer.calls] == ["Pick blue.", "Wait."]
    assert np.all(row["state"].reshape(2, 32)[:, 8:] == 0)
    assert not np.array_equal(tokenizer.calls[0][1][:8], tokenizer.calls[1][1][:8])
    for _, private_state in tokenizer.calls:
        assert private_state.shape == (32,)
        assert np.all(private_state[8:] == 0)
    output = UnflattenPair()({"state": row["state"], "actions": row["actions"]})
    output = transforms.Unnormalize(stats, use_quantiles=True)(output)
    output = JoinOutputs()(output)
    output = transforms.AbsoluteActions(mask)(output)
    np.testing.assert_allclose(output["actions"][:, :16], original[:, :16], atol=1e-6)
    np.testing.assert_allclose(output["state"][:16], state[:16], atol=1e-6)


def test_inference_without_action_targets():
    state = np.zeros(32)
    state[8:16] = 1
    paired = PairInputs()({"state": state, "prompt": "Wait. | Pick blue."})
    result = TokenizePair(RecordingTokenizer())(paired)
    assert "actions" not in result
    assert result["state"].shape == (64,)


def test_registered_droid_transform_keeps_original_16d_output():
    from openpi.training.config import get_config  # noqa: PLC0415

    cfg = get_config("pi05_intent_ar_r3_comm_random_order")
    group = cfg.data.data_transforms(cfg.model)
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    target = np.tile(np.arange(16, dtype=np.float32) / 10 + 0.05, (50, 1))
    source = {
        "global_image": image,
        "image_agent0": image.copy(),
        "image_agent1": image.copy(),
        "joint_position_agent0": np.arange(7, dtype=np.float32) / 10,
        "joint_position_agent1": np.arange(8, 15, dtype=np.float32) / 10,
        "gripper_position_agent0": np.float32(0.7),
        "gripper_position_agent1": np.float32(1.5),
        "task": "Pick blue. | Wait.",
        "actions": target.copy(),
    }
    converted = transforms.compose(group.inputs)(source)
    assert converted["state"].shape == (2, 32)
    assert converted["actions"].shape == (50, 2, 32)
    np.testing.assert_allclose(converted["actions"][..., :7], 0.05, atol=1e-6)
    np.testing.assert_allclose(converted["actions"][0, :, 7], [0.75, 1.55], atol=1e-6)
    restored = transforms.compose(group.outputs)({"state": converted["state"], "actions": converted["actions"]})
    assert restored["actions"].shape == (50, 16)
    np.testing.assert_allclose(restored["actions"], target, atol=1e-6)


@pytest.mark.parametrize("prompt", ["global only", "A|B|C", "A|", "|B"])
def test_ambiguous_prompts_rejected(prompt):
    with pytest.raises(ValueError, match="instruction"):
        TokenizePair(RecordingTokenizer())({"state": np.zeros((2, 32)), "prompt": prompt})
