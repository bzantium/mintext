"""Rotary positional embeddings (RoPE) with extended variants."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from flax import nnx

from mintext.config import MinTextConfig


# --- Helper functions for YaRN ---


def _yarn_find_correction_dim(
    num_rotations: float, dim: int, base: float, max_position_embeddings: int,
) -> float:
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
        2 * math.log(base)
    )


def _yarn_find_correction_range(
    beta_fast: float, beta_slow: float, dim: int, base: float, orig_max_pos: int,
) -> tuple[int, int]:
    low = _yarn_find_correction_dim(beta_fast, dim, base, orig_max_pos)
    high = _yarn_find_correction_dim(beta_slow, dim, base, orig_max_pos)
    return max(math.floor(low), 0), min(math.ceil(high), dim - 1)


def _yarn_linear_ramp(low: int, high: int, dim: int) -> jax.Array:
    if low == high:
        high = high + 0.001
    linear = (jnp.arange(dim, dtype=jnp.float32) - low) / (high - low)
    return jnp.clip(linear, 0.0, 1.0)


def _yarn_get_mscale(factor: float, mscale: float, mscale_all_dim: float) -> float:
    if factor <= 1.0:
        return 1.0
    if mscale_all_dim > 0:
        return (0.1 * mscale * math.log(factor) + 1.0) / (
            0.1 * mscale_all_dim * math.log(factor) + 1.0
        )
    return 0.1 * math.log(factor) + 1.0


# --- Inverse frequency computation per RoPE variant ---


def _compute_default_inv_freq(base: float, dim: int) -> tuple[jax.Array, float]:
    inv_freq = 1.0 / (base ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))
    return inv_freq, 1.0


def _compute_linear_inv_freq(
    base: float, dim: int, factor: float,
) -> tuple[jax.Array, float]:
    inv_freq = 1.0 / (factor * base ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))
    return inv_freq, 1.0


def _compute_yarn_inv_freq(
    base: float,
    dim: int,
    factor: float,
    orig_max_pos: int,
    beta_fast: float,
    beta_slow: float,
    mscale: float,
    mscale_all_dim: float,
) -> tuple[jax.Array, float]:
    pos_freqs = base ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim)
    inv_freq_extra = 1.0 / pos_freqs
    inv_freq_inter = 1.0 / (factor * pos_freqs)

    low, high = _yarn_find_correction_range(beta_fast, beta_slow, dim, base, orig_max_pos)
    mask = 1.0 - _yarn_linear_ramp(low, high, dim // 2)

    inv_freq = inv_freq_inter * (1 - mask) + inv_freq_extra * mask
    attention_factor = _yarn_get_mscale(factor, mscale, mscale_all_dim)

    return inv_freq, attention_factor


def _compute_llama3_inv_freq(
    base: float,
    dim: int,
    factor: float,
    orig_max_pos: int,
    low_freq_factor: float,
    high_freq_factor: float,
) -> tuple[jax.Array, float]:
    inv_freq = 1.0 / (base ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))
    wavelen = 2 * jnp.pi / inv_freq

    low_freq_wavelen = orig_max_pos / low_freq_factor
    high_freq_wavelen = orig_max_pos / high_freq_factor

    # Three regions: high-freq unchanged, low-freq scaled, medium smoothly interpolated
    inv_freq_scaled = jnp.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
    smooth = (orig_max_pos / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    smoothed = (1 - smooth) * inv_freq_scaled / factor + smooth * inv_freq_scaled
    is_medium = ~(wavelen < high_freq_wavelen) & ~(wavelen > low_freq_wavelen)
    inv_freq_final = jnp.where(is_medium, smoothed, inv_freq_scaled)

    return inv_freq_final, 1.0


def compute_inv_freq(config: MinTextConfig, dim: int) -> tuple[jax.Array, float]:
    """Compute inverse frequencies and attention scaling factor for RoPE.

    Args:
        config: MinText config with rope_type and scaling parameters.
        dim: Rotary dimension (head_dim or partial thereof).

    Returns:
        (inv_freq, attention_factor) where inv_freq has shape [dim//2].
    """
    base = config.rope_theta
    orig = config.rope_original_max_position_embeddings or config.seq_length

    rope_type = config.rope_type
    if rope_type == "default":
        return _compute_default_inv_freq(base, dim)
    elif rope_type == "linear":
        return _compute_linear_inv_freq(base, dim, config.rope_scaling_factor)
    elif rope_type == "yarn":
        return _compute_yarn_inv_freq(
            base, dim, config.rope_scaling_factor, orig,
            config.rope_yarn_beta_fast, config.rope_yarn_beta_slow,
            config.rope_yarn_mscale, config.rope_yarn_mscale_all_dim,
        )
    elif rope_type == "llama3":
        return _compute_llama3_inv_freq(
            base, dim, config.rope_scaling_factor, orig,
            config.rope_llama3_low_freq_factor, config.rope_llama3_high_freq_factor,
        )
    else:
        raise ValueError(f"Unknown rope_type: {rope_type}")


# --- Rotation helpers ---


def _rotate_half(x: jax.Array) -> jax.Array:
    """Rotate the last dimension by splitting in half and swapping."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return jnp.concatenate([-x2, x1], axis=-1)


def _rotate_interleave(x: jax.Array) -> jax.Array:
    """Interleaved rotation: negate and swap pairs (x0,x1,x2,x3) -> (-x1,x0,-x3,x2)."""
    x = x.reshape(*x.shape[:-1], -1, 2)
    x1, x2 = x[..., 0], x[..., 1]
    rotated = jnp.stack([-x2, x1], axis=-1)
    return rotated.reshape(*rotated.shape[:-2], -1)


# --- RotaryEmbedding module ---


class RotaryEmbedding(nnx.Module):
    """RoPE with half-rotation or interleaved pattern, optional partial application.

    Supports default, linear, YaRN, and Llama3 RoPE variants via config.

    Input shape: [batch, seq_len, num_attention_heads, head_dim]
    Position shape: [batch, seq_len]
    """

    def __init__(self, config: MinTextConfig, rope_dim: int | None = None):
        head_dim = rope_dim if rope_dim is not None else config.head_dim
        self.head_dim = head_dim
        self.interleave = config.rope_interleave
        self.partial_rotary_factor = config.partial_rotary_factor

        # Precompute inverse frequencies and attention scaling factor
        rot_dim = head_dim
        if self.partial_rotary_factor < 1.0:
            rot_dim = int(head_dim * self.partial_rotary_factor)
            rot_dim = rot_dim - (rot_dim % 2)
        self._rot_dim = rot_dim
        self._inv_freq, self._attention_factor = compute_inv_freq(config, rot_dim)

    def __call__(self, inputs: jax.Array, positions: jax.Array) -> jax.Array:
        """Apply RoPE to inputs.

        Args:
            inputs: [batch, seq_len, num_attention_heads, head_dim]
            positions: [batch, seq_len]

        Returns:
            Rotated inputs with same shape.
        """
        full_dim = inputs.shape[-1]
        rot_dim = self._rot_dim

        if rot_dim < full_dim:
            x_rot = inputs[..., :rot_dim]
            x_pass = inputs[..., rot_dim:]
        else:
            x_rot = inputs
            x_pass = None

        inv_freq = self._inv_freq
        attention_factor = self._attention_factor

        # positions: [B, S] -> [B, S, 1, 1]; inv_freq: [half_dim]
        positions_expanded = positions[:, :, jnp.newaxis, jnp.newaxis]
        sinusoid_inp = positions_expanded * inv_freq  # [B, S, 1, half_dim]

        sin_val = jnp.sin(sinusoid_inp) * attention_factor
        cos_val = jnp.cos(sinusoid_inp) * attention_factor
        sin = jnp.concatenate([sin_val, sin_val], axis=-1)
        cos = jnp.concatenate([cos_val, cos_val], axis=-1)

        if self.interleave:
            rotated = (x_rot * cos) + (_rotate_interleave(x_rot) * sin)
        else:
            rotated = (x_rot * cos) + (_rotate_half(x_rot) * sin)

        if x_pass is not None:
            result = jnp.concatenate([rotated, x_pass], axis=-1)
        else:
            result = rotated
        # Cast back to input dtype (RoPE computes in float32 for accuracy)
        return jnp.asarray(result, inputs.dtype)
