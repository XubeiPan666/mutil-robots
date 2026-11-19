"""V1 shared single-arm Pi0.5 with differentiable agent-wise communication.

Config and architecture intentionally live together. The external training
Observation packs two 32D arms into 64 slots; each backbone call sees ONLY one
32D arm, its private prompt/camera and the shared global camera. Physical arm
indices are used for routing, never embedded in the network.
"""

import dataclasses

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from openpi.models import gemma
from openpi.models import model as _model
from openpi.models.pi0 import Pi0
from openpi.models.pi0 import make_attn_mask
from openpi.models.pi0_config import Pi0Config
from openpi.shared import array_typing as at


@dataclasses.dataclass(frozen=True)
class Pi05IntentARConfig(Pi0Config):
    pi05: bool = True
    discrete_state_input: bool = True
    message_tokens: int = 4
    communicate: bool = True
    random_order: bool = True
    inference_order: tuple[int, int] = (0, 1)

    def __post_init__(self):
        super().__post_init__()
        if not self.pi05 or not self.discrete_state_input or self.action_dim != 32:
            raise ValueError("Intent AR requires Pi0.5 with discrete state and a 32D per-arm interface")
        if self.message_tokens < 1 or self.action_horizon < 1:
            raise ValueError("Positive message count and action horizon required")
        if tuple(sorted(self.inference_order)) != (0, 1):
            raise ValueError("V1 inference_order must be (0,1) or (1,0)")

    def create(self, rng):
        return Pi05IntentAR(self, rngs=nnx.Rngs(rng))

    def inputs_spec(self, *, batch_size=1):
        obs, _ = super().inputs_spec(batch_size=batch_size)
        with at.disable_typechecking():
            obs = dataclasses.replace(
                obs,
                state=jax.ShapeDtypeStruct((batch_size, 64), jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct((batch_size, 2 * self.max_token_len), jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct((batch_size, 2 * self.max_token_len), jnp.bool_),
            )
        return obs, jax.ShapeDtypeStruct((batch_size, self.action_horizon, 64), jnp.float32)

    def load_pytorch(self, train_config, weight_path):
        raise NotImplementedError("Intent AR V1 is a JAX model; the plain PyTorch Pi0 loader is incompatible")


class Pi05IntentAR(Pi0):
    def __init__(self, config: Pi05IntentARConfig, rngs: nnx.Rngs):
        # Keep original Pi0.5 parameter paths and shapes for strict base loading.
        super().__init__(config, rngs)
        self.message_tokens = config.message_tokens
        self.communicate = config.communicate
        self.random_order = config.random_order
        self.inference_order = config.inference_order
        width = gemma.get_config(config.paligemma_variant).width
        self.intent_bos = nnx.Param(jax.random.normal(rngs.params(), (config.message_tokens, width)) * 0.02)
        self.intent_projection = nnx.Linear(width, width, rngs=rngs)

    def _validate_observation(self, obs):
        batch = obs.state.shape[0]
        if obs.state.shape != (batch, 64):
            raise ValueError("Expected grouped state [B,64]; apply IntentAR tokenization/packing")
        expected = (batch, 2 * self.max_token_len)
        if obs.tokenized_prompt is None or obs.tokenized_prompt.shape != expected:
            raise ValueError(f"Expected paired private prompts {expected}")
        if obs.tokenized_prompt_mask is None or obs.tokenized_prompt_mask.shape != expected:
            raise ValueError("Missing paired prompt mask")

    def _arm_observation(self, obs, indices):
        """Gather each scene's physical arm into the same canonical single-arm slots."""
        b = obs.state.shape[0]
        rows = jnp.arange(b)
        wrists = jnp.stack([obs.images["left_wrist_0_rgb"], obs.images["right_wrist_0_rgb"]], axis=1)
        masks = jnp.stack([obs.image_masks["left_wrist_0_rgb"], obs.image_masks["right_wrist_0_rgb"]], axis=1)
        return _model.Observation(
            images={"base_0_rgb": obs.images["base_0_rgb"], "left_wrist_0_rgb": wrists[rows, indices]},
            image_masks={"base_0_rgb": obs.image_masks["base_0_rgb"], "left_wrist_0_rgb": masks[rows, indices]},
            state=obs.state.reshape(b, 2, 32)[rows, indices],
            tokenized_prompt=obs.tokenized_prompt.reshape(b, 2, self.max_token_len)[rows, indices],
            tokenized_prompt_mask=obs.tokenized_prompt_mask.reshape(b, 2, self.max_token_len)[rows, indices],
        )

    def _embed_arm_prefix(self, obs, global_tokens):
        """Reuse scene-level visual tokens; keep arm-specific text/images separate."""
        private = dataclasses.replace(
            obs,
            images={name: value for name, value in obs.images.items() if name != "base_0_rgb"},
            image_masks={name: value for name, value in obs.image_masks.items() if name != "base_0_rgb"},
        )
        private_tokens, private_mask, private_ar = self.embed_prefix(private)
        global_mask = jnp.broadcast_to(obs.image_masks["base_0_rgb"][:, None], global_tokens.shape[:2])
        return (
            jnp.concatenate([global_tokens, private_tokens], axis=1),
            jnp.concatenate([global_mask, private_mask], axis=1),
            jnp.concatenate([jnp.zeros(global_tokens.shape[1], dtype=jnp.bool_), private_ar]),
        )

    def _condition(self, obs, incoming, global_tokens=None):
        """Update incoming slots and return differentiable KV from ONE prefix pass.

        R1 uses base only. R2/R3 use one bidirectional prefix [base, incoming],
        where incoming is BOS for the first arm. No independent queries or
        reinsertion of the final message into this arm's action context.
        """
        if global_tokens is None:
            global_tokens, _ = self.PaliGemma.img(obs.images["base_0_rgb"], train=False)
        base, base_mask, _ = self._embed_arm_prefix(obs, global_tokens)
        tokens, mask = base, base_mask
        if self.communicate:
            msg_mask = jnp.ones((base.shape[0], self.message_tokens), dtype=jnp.bool_)
            tokens = jnp.concatenate([base, incoming], axis=1)
            mask = jnp.concatenate([base_mask, msg_mask], axis=1)
        ar = jnp.zeros(tokens.shape[1], dtype=jnp.bool_)
        (hidden, _), cache = self.PaliGemma.llm(
            [tokens, None], mask=make_attn_mask(mask, ar), positions=jnp.cumsum(mask, axis=1) - 1
        )
        message = self.intent_projection(hidden[:, -self.message_tokens :]) if self.communicate else None
        return message, cache, mask

    def _velocity(self, obs, noisy, time, cache, prefix_mask):
        """Read the same prefix KV during training and every inference flow step."""
        suffix, suffix_mask, suffix_ar, cond = self.embed_suffix(obs, noisy, time)
        prefix_attn = jnp.broadcast_to(
            prefix_mask[:, None, :], (noisy.shape[0], suffix.shape[1], prefix_mask.shape[1])
        )
        mask = jnp.concatenate([prefix_attn, make_attn_mask(suffix_mask, suffix_ar)], axis=-1)
        positions = jnp.sum(prefix_mask, axis=1)[:, None] + jnp.cumsum(suffix_mask, axis=1) - 1
        (_, output), _ = self.PaliGemma.llm(
            [None, suffix],
            mask=mask,
            positions=positions,
            kv_cache=cache,
            adarms_cond=[None, cond],
        )
        return self.action_out_proj(output[:, -self.action_horizon :])

    def _orders(self, key, batch, train):
        if train and self.random_order:
            first = jax.random.bernoulli(key, shape=(batch,)).astype(jnp.int32)
            return jnp.stack([first, 1 - first], axis=1)
        return jnp.broadcast_to(jnp.asarray(self.inference_order), (batch, 2))

    def compute_loss(self, rng, observation, actions, *, train=False):
        self._validate_observation(observation)
        b = observation.state.shape[0]
        if actions.shape != (b, self.action_horizon, 64):
            raise ValueError("Expected grouped actions [B,H,64]")
        image_key, noise_key, time_key, order_key = jax.random.split(rng, 4)
        # Augment once BEFORE arm selection: both agents see the same global image.
        obs = _model.preprocess_observation(image_key, observation, train=train)
        # Local to this forward pass; both arms backpropagate through these tokens.
        global_tokens, _ = self.PaliGemma.img(obs.images["base_0_rgb"], train=False)
        target = actions.reshape(b, self.action_horizon, 2, 32)
        valid = (jnp.arange(32) < 8).astype(target.dtype)
        target = target * valid
        # Random streams indexed by physical arm, independently of communication order.
        noise = jax.random.normal(noise_key, target.shape) * valid
        time = jax.random.beta(time_key, 1.5, 1, (b, 2)) * 0.999 + 0.001
        order = self._orders(order_key, b, train)
        bos = jnp.broadcast_to(self.intent_bos.value, (b, *self.intent_bos.value.shape))
        incoming = bos
        losses = []
        rows = jnp.arange(b)
        for position in range(2):
            arm_id = order[:, position]
            own = self._arm_observation(obs, arm_id)
            message, cache, mask = self._condition(own, incoming, global_tokens)
            own_target, own_noise = target[rows, :, arm_id], noise[rows, :, arm_id]
            own_time = time[rows, arm_id]
            t = own_time[:, None, None]
            velocity = self._velocity(own, t * own_noise + (1 - t) * own_target, own_time, cache, mask)
            losses.append(jnp.mean(jnp.square(velocity[..., :8] - (own_noise - own_target)[..., :8]), axis=-1))
            incoming = message if self.communicate else bos
        return jnp.mean(jnp.stack(losses, axis=1), axis=1)

    def _sample_arm(self, obs, noise, cache, prefix_mask, num_steps):
        valid = (jnp.arange(32) < 8).astype(noise.dtype)

        def step(i, x):
            t = jnp.broadcast_to(1.0 - i / num_steps, (x.shape[0],))
            velocity = self._velocity(obs, x, t, cache, prefix_mask)
            return (x - velocity / num_steps) * valid

        return jax.lax.fori_loop(0, num_steps, step, noise * valid)

    def sample_actions(self, rng, observation, *, num_steps=10, noise=None):
        self._validate_observation(observation)
        # Explicit, static inference order; no hidden mutable cross-request state.
        order = self.inference_order
        if isinstance(num_steps, int) and num_steps < 1:
            raise ValueError("num_steps must be positive")
        obs = _model.preprocess_observation(None, observation, train=False)
        global_tokens, _ = self.PaliGemma.img(obs.images["base_0_rgb"], train=False)
        b = obs.state.shape[0]
        shape = (b, self.action_horizon, 64)
        if noise is None:
            noise = jax.random.normal(rng, shape)
        if noise.shape != shape:
            raise ValueError(f"Expected physical-arm-indexed noise {shape}")
        noise = noise.reshape(b, self.action_horizon, 2, 32)
        result = jnp.zeros_like(noise)
        bos = jnp.broadcast_to(self.intent_bos.value, (b, *self.intent_bos.value.shape))
        incoming = bos
        for arm_id in order:
            own = self._arm_observation(obs, jnp.full((b,), arm_id))
            message, cache, mask = self._condition(own, incoming, global_tokens)
            action = self._sample_arm(own, noise[:, :, arm_id], cache, mask, num_steps)
            result = result.at[:, :, arm_id].set(action)
            incoming = message if self.communicate else bos
        return result.reshape(shape)
