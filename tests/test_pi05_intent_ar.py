"""Real JAX/Gemma/NNX tests with a tiny image encoder and reduced test vocabulary.

No pretrained checkpoint, tokenizer download, simulator or GPU is used. These
tests validate V1 computation/gradients, not full-size memory or training quality.
"""

# These tests intentionally inspect private causal/gradient boundaries.
# ruff: noqa: SLF001

import dataclasses

import flax.linen as linen
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import gemma
from openpi.models import pi0
from openpi.models.pi05_intent_ar import Pi05IntentARConfig
from openpi.shared.nnx_utils import module_jit
from openpi.training import weight_loaders
from openpi.training.config import get_config


class TinyImage(linen.Module):
    num_classes: int
    variant: str
    pool_type: str
    scan: bool
    dtype_mm: str

    @linen.compact
    def __call__(self, image, *, train=False):
        value = jnp.mean(image, axis=(1, 2))[:, None, :]
        return linen.Dense(self.num_classes)(value).astype(self.dtype_mm), {}


@pytest.fixture
def tiny(monkeypatch):
    monkeypatch.setattr(pi0._siglip, "Module", TinyImage)
    monkeypatch.setattr(gemma, "PALIGEMMA_VOCAB_SIZE", 128)
    original = gemma.get_config
    monkeypatch.setattr(gemma, "get_config", lambda variant: dataclasses.replace(original(variant), depth=1))
    cfg = Pi05IntentARConfig(
        dtype="float32",
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_horizon=2,
        max_token_len=4,
        message_tokens=2,
    )
    model = cfg.create(jax.random.key(7))
    # Fresh Pi0.5 adaRMS residual gates are zero, so a randomly initialized action
    # expert initially ignores *all* prefix context. Real runs load trained base
    # weights. Open those gates only in this tiny fixture to test the actual
    # downstream communication path, without changing production initialization.
    state = nnx.state(model)
    for path, variable in state.flat_state().items():
        if "Dense_0" in path and path[-1] == "bias" and any(str(part).endswith("_1") for part in path):
            width = variable.value.shape[-1] // 3
            variable.value = variable.value.at[..., 2 * width :].set(1)
    nnx.update(model, state)
    obs = cfg.fake_obs(batch_size=2)
    obs = dataclasses.replace(
        obs,
        images={name: jnp.full((2, 224, 224, 3), i / 4, dtype=jnp.float32) for i, name in enumerate(obs.images)},
        state=jnp.arange(128, dtype=jnp.float32).reshape(2, 64) / 100,
        tokenized_prompt=jnp.asarray([[1, 2, 3, 4, 5, 6, 7, 8]] * 2, dtype=jnp.int32),
    )
    return cfg, model, obs


def test_only_current_arm_inputs_reach_prefix(tiny):
    _, model, obs = tiny
    own = model._arm_observation(obs, jnp.asarray([0, 1]))
    np.testing.assert_array_equal(own.state[0], obs.state[0, :32])
    np.testing.assert_array_equal(own.state[1], obs.state[1, 32:])
    np.testing.assert_array_equal(own.tokenized_prompt, [[1, 2, 3, 4], [5, 6, 7, 8]])
    assert tuple(own.images) == ("base_0_rgb", "left_wrist_0_rgb")
    np.testing.assert_allclose(own.images["left_wrist_0_rgb"][:, 0, 0, 0], [0.25, 0.5])


def test_first_message_cannot_see_later_private_inputs(tiny):
    _, model, obs = tiny
    indices = jnp.zeros((2,), jnp.int32)
    bos = jnp.broadcast_to(model.intent_bos.value, (2, *model.intent_bos.value.shape))
    before = model._condition(model._arm_observation(obs, indices), bos)[0]
    changed = dataclasses.replace(
        obs,
        state=obs.state.at[:, 32:].set(999),
        tokenized_prompt=obs.tokenized_prompt.at[:, 4:].set(99),
        images={**obs.images, "right_wrist_0_rgb": jnp.ones_like(obs.images["right_wrist_0_rgb"])},
    )
    after = model._condition(model._arm_observation(changed, indices), bos)[0]
    np.testing.assert_array_equal(before, after)


def test_shared_visual_tokens_preserve_prefix_and_encoder_gradients(tiny):
    _, model, obs = tiny
    obs = dataclasses.replace(obs, image_masks={**obs.image_masks, "base_0_rgb": jnp.asarray([True, False])})

    def prefix_loss(m, shared):
        global_tokens = m.PaliGemma.img(obs.images["base_0_rgb"], train=False)[0] if shared else None
        total = 0.0
        for indices in (jnp.asarray([0, 1]), jnp.asarray([1, 0])):
            own = m._arm_observation(obs, indices)
            actual = m._embed_arm_prefix(own, global_tokens) if shared else m.embed_prefix(own)
            tokens, mask, _ = actual
            total += jnp.mean(tokens ** 2 * mask[..., None])
        return total

    reference, reference_grad = nnx.value_and_grad(lambda m: prefix_loss(m, shared=False))(model)
    actual, actual_grad = nnx.value_and_grad(lambda m: prefix_loss(m, shared=True))(model)
    np.testing.assert_allclose(actual, reference, rtol=1e-5, atol=1e-6)
    for expected, result in zip(
        jax.tree.leaves(reference_grad.PaliGemma.img), jax.tree.leaves(actual_grad.PaliGemma.img), strict=True
    ):
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-6)
    own = model._arm_observation(obs, jnp.asarray([0, 1]))
    global_tokens, _ = model.PaliGemma.img(obs.images["base_0_rgb"], train=False)
    for expected, result in zip(model.embed_prefix(own), model._embed_arm_prefix(own, global_tokens), strict=True):
        np.testing.assert_array_equal(result, expected)


def test_training_and_inference_encode_three_views_and_match_repeated_encoding(tiny, monkeypatch):
    _, model, obs = tiny
    image_encoder = model.PaliGemma.img
    encoder_class = type(image_encoder)
    original_call = encoder_class.__call__
    calls = []

    def counted_call(self, *args, **kwargs):
        if self is image_encoder:
            calls.append(1)
        return original_call(self, *args, **kwargs)

    monkeypatch.setattr(encoder_class, "__call__", counted_call)
    actions = jnp.zeros((2, 2, 64))
    key = jax.random.key(11)
    loss = model.compute_loss(key, obs, actions, train=True)
    assert len(calls) == 3  # One global view and two private wrists.
    calls.clear()
    sampled = model.sample_actions(key, obs, num_steps=1)
    assert len(calls) == 3
    original_condition = type(model)._condition

    def repeated_condition(self, own, incoming, global_tokens=None):
        return original_condition(self, own, incoming)  # Encode global view again for each arm.

    monkeypatch.setattr(type(model), "_condition", repeated_condition)
    np.testing.assert_allclose(model.compute_loss(key, obs, actions, train=True), loss, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(model.sample_actions(key, obs, num_steps=1), sampled, rtol=1e-5, atol=1e-6)


def test_downstream_action_loss_differentiates_through_message(tiny):
    _, model, obs = tiny
    first = model._arm_observation(obs, jnp.zeros((2,), jnp.int32))
    second = model._arm_observation(obs, jnp.ones((2,), jnp.int32))
    bos = jnp.broadcast_to(model.intent_bos.value, (2, *model.intent_bos.value.shape))

    def loss_from_first_image(image):
        first_changed = dataclasses.replace(first, images={**first.images, "left_wrist_0_rgb": image})
        message, _, _ = model._condition(first_changed, bos)
        _, prefix, mask = model._condition(second, message)
        velocity = model._velocity(second, jnp.ones((2, 2, 32)), jnp.full((2,), 0.5), prefix, mask)
        return jnp.mean(velocity[..., :8] ** 2)

    grad = jax.grad(loss_from_first_image)(first.images["left_wrist_0_rgb"])
    assert np.isfinite(grad).all()
    assert float(jnp.linalg.norm(grad)) > 1e-8


def test_loss_and_parameter_gradients_are_finite(tiny):
    _, model, obs = tiny
    actions = jnp.zeros((2, 2, 64))
    value, grads = nnx.value_and_grad(lambda m: jnp.mean(m.compute_loss(jax.random.key(11), obs, actions)))(model)
    assert np.isfinite(value)
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(grads))
    assert float(jnp.linalg.norm(grads.intent_bos.value)) > 0
    assert float(jnp.linalg.norm(grads.intent_projection.kernel.value)) > 0
    # Padding targets cannot affect either arm's valid loss.
    changed = actions.reshape(2, 2, 2, 32).at[:, :, :, 8:].set(10000).reshape(actions.shape)
    np.testing.assert_allclose(
        model.compute_loss(jax.random.key(11), obs, actions), model.compute_loss(jax.random.key(11), obs, changed)
    )


@pytest.mark.parametrize("communicate", [False, True])
def test_cached_velocity_matches_joint_forward(tiny, communicate):
    """Cache reuse preserves the action computation of a joint prefix/suffix pass."""
    _, model, obs = tiny
    model.communicate = communicate
    own = model._arm_observation(obs, jnp.zeros((2,), jnp.int32))
    incoming = jnp.broadcast_to(model.intent_bos.value, (2, *model.intent_bos.value.shape))
    message, cache, mask = model._condition(own, incoming)
    base, base_mask, _ = model.embed_prefix(own)
    prefix = jnp.concatenate([base, incoming], axis=1) if communicate else base
    assert mask.shape[1] == base_mask.shape[1] + (model.message_tokens if communicate else 0)
    assert (message is None) == (not communicate)
    noisy = jnp.ones((2, 2, 32))
    time = jnp.full((2,), 0.5)
    actual = model._velocity(own, noisy, time, cache, mask)
    suffix, suffix_mask, suffix_ar, cond = model.embed_suffix(own, noisy, time)
    full_mask = jnp.concatenate([mask, suffix_mask], axis=1)
    ar = jnp.concatenate([jnp.zeros(prefix.shape[1], dtype=jnp.bool_), suffix_ar])
    (_, output), _ = model.PaliGemma.llm(
        [prefix, suffix], mask=pi0.make_attn_mask(full_mask, ar),
        positions=jnp.cumsum(full_mask, axis=1) - 1, adarms_cond=[None, cond],
    )
    expected = model.action_out_proj(output[:, -model.action_horizon :])
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
    # A one-step inference integration uses exactly the training velocity.
    valid = (jnp.arange(32) < 8).astype(noisy.dtype)
    initial = noisy * valid
    velocity = model._velocity(own, initial, jnp.ones((2,)), cache, mask)
    sampled = model._sample_arm(own, noisy, cache, mask, num_steps=1)
    np.testing.assert_allclose(sampled, (initial - velocity) * valid, rtol=1e-4, atol=1e-5)


def test_output_message_is_not_fed_back_into_own_actions(tiny):
    _, model, obs = tiny
    own = model._arm_observation(obs, jnp.zeros((2,), jnp.int32))
    incoming = jnp.broadcast_to(model.intent_bos.value, (2, *model.intent_bos.value.shape))
    before, cache_before, mask = model._condition(own, incoming)
    model.intent_projection.bias.value += 10
    after, cache_after, _ = model._condition(own, incoming)
    assert not np.allclose(before, after)
    for left, right in zip(jax.tree.leaves(cache_before), jax.tree.leaves(cache_after), strict=True):
        np.testing.assert_array_equal(left, right)
    args = (own, jnp.ones((2, 2, 32)), jnp.full((2,), 0.5))
    np.testing.assert_array_equal(
        model._velocity(*args, cache_before, mask), model._velocity(*args, cache_after, mask)
    )


def test_r1_order_invariance_and_padded_action_output(tiny):
    _, model, obs = tiny
    model.communicate = False
    a = model.sample_actions(jax.random.key(9), obs, num_steps=2)
    model.inference_order = (1, 0)
    b = model.sample_actions(jax.random.key(9), obs, num_steps=2)
    assert a.shape == (2, 2, 64)
    assert np.isfinite(a).all()
    np.testing.assert_allclose(a, b, atol=1e-6)
    np.testing.assert_array_equal(a.reshape(2, 2, 2, 32)[..., 8:], 0)


def test_policy_jit_and_cached_sampling(tiny):
    _, model, obs = tiny
    expected = model.sample_actions(jax.random.key(5), obs, num_steps=2)
    compiled = module_jit(model.sample_actions)
    actual = compiled(jax.random.key(5), obs, num_steps=2)
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)


def test_new_model_parameter_roundtrip(tiny):
    cfg, model, obs = tiny
    params = nnx.state(model).to_pure_dict()
    restored = cfg.load(params)
    expected = model.sample_actions(jax.random.key(6), obs, num_steps=1)
    actual = restored.sample_actions(jax.random.key(6), obs, num_steps=1)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_random_order_is_per_scene_and_inference_is_fixed(tiny):
    _, model, _ = tiny
    orders = np.asarray(model._orders(jax.random.key(0), 256, train=True))
    assert set(map(tuple, orders)) == {(0, 1), (1, 0)}
    np.testing.assert_array_equal(np.sort(orders, axis=1), np.tile([0, 1], (256, 1)))
    np.testing.assert_array_equal(model._orders(jax.random.key(1), 3, train=False), [[0, 1]] * 3)


def test_base_loader_only_initializes_message_params(monkeypatch):
    params = {
        "action_in_proj": {"kernel": np.ones((2, 2))},
        "intent_bos": np.zeros((2, 2)),
        "intent_projection": {"kernel": np.zeros((2, 2))},
    }
    monkeypatch.setattr(weight_loaders.download, "maybe_download", lambda path: path)
    monkeypatch.setattr(
        weight_loaders._model, "restore_params", lambda *a, **k: {"action_in_proj": {"kernel": np.full((2, 2), 3.0)}}
    )
    loaded = weight_loaders.Pi05IntentARWeightLoader("unused").load(params)
    np.testing.assert_array_equal(loaded["action_in_proj"]["kernel"], 3.0)
    monkeypatch.setattr(weight_loaders._model, "restore_params", lambda *a, **k: {})
    with pytest.raises((ValueError, AssertionError)):
        weight_loaders.Pi05IntentARWeightLoader("unused").load(params)


def test_ablations_registered_without_changing_baseline():
    configs = [
        get_config("pi05_intent_ar_r1_no_comm_fixed_order"),
        get_config("pi05_intent_ar_r2_comm_fixed_order"),
        get_config("pi05_intent_ar_r3_comm_random_order"),
    ]
    assert [(c.model.communicate, c.model.random_order) for c in configs] == [
        (False, False),
        (True, False),
        (True, True),
    ]
    assert not isinstance(get_config("pi05_two_arms_multi_task").model, Pi05IntentARConfig)
    assert all(c.model.action_dim == 32 and c.batch_size == 32 and c.num_train_steps == 30000 for c in configs)
    assert all(c.save_interval == c.keep_period == 5000 for c in configs)
