"""Paired numeric/token transforms for the shared single-arm intent policy.

Run PairInputs AFTER DroidMultiAgentInputs and DeltaActions, BEFORE Normalize.
TokenizePair runs AFTER Normalize. Output transforms reverse this exact sequence.
No dataset files are rewritten, and no physical-arm identity token is introduced.
"""

import dataclasses

import numpy as np

from openpi.shared.single_arm import split_delta_pair


@dataclasses.dataclass(frozen=True)
class PairInputs:
    horizon: int = 50

    def __call__(self, data):
        result = dict(data)
        state = np.asarray(data["state"])
        if "actions" in data:
            pair = split_delta_pair(state, data["actions"], horizon=self.horizon)
            result["state"] = np.pad(pair["state"], ((0, 0), (0, 24)))
            result["actions"] = np.pad(pair["actions"].transpose(1, 0, 2), ((0, 0), (0, 0), (0, 24)))
        else:
            if state.shape != (32,) or not np.isfinite(state).all() or np.any(state[16:] != 0):
                raise ValueError("Expected two 8D absolute states plus zero padding")
            result["state"] = np.pad(state[:16].reshape(2, 8), ((0, 0), (0, 24)))
        return result


@dataclasses.dataclass(frozen=True)
class JoinOutputs:
    """After shared Unnormalize, restore dual-arm slots for AbsoluteActions."""

    def __call__(self, data):
        result = dict(data)
        state, actions = np.asarray(data["state"]), np.asarray(data["actions"])
        if state.shape != (2, 32) or actions.ndim != 3 or actions.shape[1:] != (2, 32):
            raise ValueError("Expected unnormalized paired state/actions")
        result["state"] = np.pad(state[:, :8].reshape(16), (0, 16))
        result["actions"] = np.pad(actions[:, :, :8].reshape(-1, 16), ((0, 0), (0, 16)))
        return result


@dataclasses.dataclass(frozen=True)
class TokenizePair:
    tokenizer: object

    def __call__(self, data):
        result = dict(data)
        prompt = data.get("prompt")
        if isinstance(prompt, bytes):
            prompt = prompt.decode("utf-8")
        if not isinstance(prompt, str) or len(prompt.split("|")) != 2:
            raise ValueError("Intent AR requires exactly two private instructions separated by '|'")
        parts = [part.strip() for part in prompt.split("|")]
        if not all(parts):
            raise ValueError("Each arm needs its own nonempty instruction")
        state = np.asarray(data["state"]).copy()
        if state.shape != (2, 32):
            raise ValueError("Expected paired normalized state (2,32)")
        # Quantile normalization maps zero padding to -1. Canonicalize it before
        # tokenization so padding cannot become a physical-arm signal.
        state[:, 8:] = 0
        encoded = [self.tokenizer.tokenize(parts[i], state[i]) for i in range(2)]
        result["tokenized_prompt"] = np.concatenate([item[0] for item in encoded])
        result["tokenized_prompt_mask"] = np.concatenate([item[1] for item in encoded])
        result["state"] = state.reshape(64)
        if "actions" in data:
            actions = np.asarray(data["actions"]).copy()
            if actions.ndim != 3 or actions.shape[1:] != (2, 32):
                raise ValueError("Expected normalized actions (H,2,32)")
            actions[:, :, 8:] = 0
            result["actions"] = actions.reshape(-1, 64)
        result.pop("prompt", None)
        result.pop("skill_labels", None)
        return result


@dataclasses.dataclass(frozen=True)
class UnflattenPair:
    """Before Unnormalize: expose the per-arm 32D axis for shared statistics."""

    def __call__(self, data):
        result = dict(data)
        state, actions = np.asarray(data["state"]), np.asarray(data["actions"])
        if state.shape != (64,) or actions.ndim != 2 or actions.shape[-1] != 64:
            raise ValueError("Expected packed state (64,) and actions (H,64)")
        result["state"] = state.reshape(2, 32)
        result["actions"] = actions.reshape(-1, 2, 32)
        return result
