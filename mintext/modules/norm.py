"""RMSNorm layer."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx


class RMSNorm(nnx.Module):
    """Root Mean Square Layer Normalization.

    Always computes in float32 for numerical stability.
    """

    def __init__(self, dim: int, epsilon: float = 1e-5, dtype: jnp.dtype = jnp.bfloat16, *, rngs: nnx.Rngs):
        self.epsilon = epsilon
        self.dtype = dtype
        self.scale = nnx.Param(jnp.ones((dim,), dtype=jnp.float32), sharding=("norm",))

    def __call__(self, x: jax.Array) -> jax.Array:
        x_f32 = jnp.asarray(x, jnp.float32)
        mean2 = jnp.mean(jax.lax.square(x_f32), axis=-1, keepdims=True)
        y = x_f32 * jax.lax.rsqrt(mean2 + self.epsilon)
        return jnp.asarray(y, self.dtype) * jnp.asarray(self.scale[...], self.dtype)
