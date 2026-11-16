"""Canonical per-arm numeric views AFTER the existing joint-delta transform.

No action conversion is performed here: callers must apply DeltaActions once,
using the current observation state for every action in the future chunk.
"""

import numpy as np


def split_delta_pair(state, actions, *, horizon: int):
    """Return state (2, 8) and actions (2, H, 8), without losing frame pairing.

    Inputs use the existing 32-dimensional dual-arm layout: two blocks of
    seven joints plus an absolute gripper, then sixteen padding dimensions.
    The output uses the same eight slots for both arms. It is NOT normalized.
    """
    state = np.asarray(state)
    actions = np.asarray(actions)
    if state.shape != (32,) or actions.shape != (horizon, 32) or horizon < 1:
        raise ValueError(f"Expected state (32,) and actions ({horizon}, 32); got {state.shape}, {actions.shape}")
    if not np.isfinite(state).all() or not np.isfinite(actions).all():
        raise ValueError("Non-finite state/actions")
    if np.any(state[16:] != 0) or np.any(actions[:, 16:] != 0):
        raise ValueError("Expected zero padding after the two 8D arms")
    return {
        "state": state[:16].reshape(2, 8).copy(),
        "actions": actions[:, :16].reshape(horizon, 2, 8).transpose(1, 0, 2).copy(),
    }


def collate_pairs(samples):
    """Keep the scene and arm axes until the statistics accumulator pools them."""
    return {key: np.stack([sample[key] for sample in samples]) for key in ("state", "actions")}


def pad_stat_vector(values):
    """Store eight meaningful slots plus exact-zero padding for the 32D interface."""
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (8,) or not np.isfinite(values).all():
        raise ValueError("Expected eight finite statistics")
    return np.pad(values, (0, 24))
