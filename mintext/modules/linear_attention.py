"""Gated Delta Rule linear attention (Qwen3-Next style).

Pure JAX implementation matching HF's torch_chunk_gated_delta_rule.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from mintext.config import MinTextConfig
from mintext.modules.linear import Linear
from mintext.modules.norm import RMSNorm


def _causal_conv1d(x: jax.Array, weight: jax.Array) -> jax.Array:
    """Causal depthwise conv1d using jax.lax.conv_general_dilated.

    Args:
        x: [batch, seq_len, channels]
        weight: [channels, 1, kernel_size]

    Returns:
        Convolved output [batch, seq_len, channels]
    """
    B, S, C = x.shape
    kernel_size = weight.shape[-1]
    # Causal padding: pad (kernel_size-1) on left
    x_padded = jnp.pad(x, ((0, 0), (kernel_size - 1, 0), (0, 0)))
    # Transpose to [batch, channels, seq_len+K-1]
    x_t = jnp.transpose(x_padded, (0, 2, 1))
    # Depthwise conv: [B, C, S+K-1] * [C, 1, K] -> [B, C, S]
    result = jax.lax.conv_general_dilated(
        x_t, weight,
        window_strides=(1,),
        padding="VALID",
        feature_group_count=C,
    )
    return jnp.transpose(result, (0, 2, 1))  # [B, S, C]


def _l2_normalize(x: jax.Array, axis: int = -1, eps: float = 1e-6) -> jax.Array:
    """L2 normalization matching HF's l2norm."""
    inv_norm = jax.lax.rsqrt(jnp.sum(x * x, axis=axis, keepdims=True) + eps)
    return x * inv_norm


def chunk_gated_delta_rule(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    beta: jax.Array,
    g: jax.Array,
    chunk_size: int = 64,
) -> jax.Array:
    """Chunked Gated Delta Rule matching HF's torch_chunk_gated_delta_rule.

    Args:
        q: [batch, num_attention_heads, seq_len, key_dim] (already transposed)
        k: [batch, num_attention_heads, seq_len, key_dim]
        v: [batch, num_attention_heads, seq_len, value_dim]
        beta: [batch, num_attention_heads, seq_len]
        g: [batch, num_attention_heads, seq_len]

    Returns:
        output: [batch, num_attention_heads, seq_len, value_dim]
    """
    B, H, S, Dk = q.shape
    Dv = v.shape[-1]

    # L2 normalize Q and K
    q = _l2_normalize(q, axis=-1)
    k = _l2_normalize(k, axis=-1)

    # Scale Q
    scale = Dk ** -0.5
    q = q * scale

    # Pad sequence to multiple of chunk_size
    pad_len = (chunk_size - S % chunk_size) % chunk_size
    if pad_len > 0:
        q = jnp.pad(q, ((0, 0), (0, 0), (0, pad_len), (0, 0)))
        k = jnp.pad(k, ((0, 0), (0, 0), (0, pad_len), (0, 0)))
        v = jnp.pad(v, ((0, 0), (0, 0), (0, pad_len), (0, 0)))
        beta = jnp.pad(beta, ((0, 0), (0, 0), (0, pad_len)))
        g = jnp.pad(g, ((0, 0), (0, 0), (0, pad_len)))

    S_padded = q.shape[2]
    num_chunks = S_padded // chunk_size

    # Pre-compute beta-weighted tensors
    v_beta = v * beta[..., None]  # [B, H, S, Dv]
    k_beta = k * beta[..., None]  # [B, H, S, Dk]

    # Reshape to chunks: [B, H, num_chunks, chunk_size, D]
    q = q.reshape(B, H, num_chunks, chunk_size, Dk)
    k = k.reshape(B, H, num_chunks, chunk_size, Dk)
    v = v.reshape(B, H, num_chunks, chunk_size, Dv)
    k_beta = k_beta.reshape(B, H, num_chunks, chunk_size, Dk)
    v_beta = v_beta.reshape(B, H, num_chunks, chunk_size, Dv)
    g = g.reshape(B, H, num_chunks, chunk_size)

    # Cumulative sum of g within chunks
    g = jnp.cumsum(g, axis=-1)  # [B, H, C, L]

    # Decay mask: exp(g[i] - g[j]) lower triangular
    decay_mask = jnp.exp(
        jnp.clip(g[..., :, None] - g[..., None, :], -30.0, 30.0)
    )  # [B, H, C, L, L]
    causal = jnp.tril(jnp.ones((chunk_size, chunk_size)))
    decay_mask = decay_mask * causal

    # Build correction matrix A (iterative process matching HF)
    # attn = -(k_beta @ key.T * decay_mask), then iterative correction
    kbk = jnp.einsum("bhcid,bhcjd->bhcij", k_beta, k)  # [B, H, C, L, L]
    attn = -(kbk * decay_mask)
    upper_mask = jnp.triu(jnp.ones((chunk_size, chunk_size), dtype=jnp.bool_))
    attn = jnp.where(upper_mask, 0.0, attn)

    # Iterative correction (matching HF's for loop)
    def _correct_row(attn, i):
        row = jax.lax.dynamic_slice_in_dim(attn, i, 1, axis=-2)  # [..., 1, L]
        sub = attn  # [..., L, L]
        # correction = sum_j row[j] * sub[j, :i] for j < i
        correction = jnp.einsum("...j,...jk->...k", row[..., 0, :], sub)
        # Only update positions [:i] in row i
        mask = jnp.arange(chunk_size) < i
        correction = correction * mask
        new_row = row[..., 0, :] + correction
        return attn.at[..., i, :].set(new_row), None

    attn, _ = jax.lax.scan(
        lambda attn, i: _correct_row(attn, i),
        attn,
        jnp.arange(1, chunk_size),
    )

    # Add identity
    eye = jnp.eye(chunk_size)
    attn = attn + eye

    # Corrected value: attn @ v_beta
    value_corrected = jnp.einsum("bhcij,bhcjd->bhcid", attn, v_beta)

    # k_cumdecay: correction for state contribution
    g_exp = jnp.exp(jnp.clip(g, -30.0, 30.0))  # [B, H, C, L]
    k_cumdecay = jnp.einsum(
        "bhcij,bhcjd->bhcid", attn, k_beta * g_exp[..., None]
    )

    # Precompute per-chunk decay values for scan (avoid redundant exp inside loop)
    g_exp_last = g_exp[:, :, :, -1:]  # [B, H, C, 1]
    g_last = g[:, :, :, -1:]  # [B, H, C, 1]
    k_decay_all = jnp.exp(jnp.clip(g_last[..., None] - g[..., None], -30.0, 30.0))  # [B, H, C, L, 1]

    # Scan over chunks with recurrent state
    init_state = jnp.zeros((B, H, Dk, Dv))

    def chunk_step(state, chunk_idx):
        q_c = q[:, :, chunk_idx]  # [B, H, L, Dk]
        k_c = k[:, :, chunk_idx]  # [B, H, L, Dk]
        v_c = value_corrected[:, :, chunk_idx]  # [B, H, L, Dv]
        dm_c = decay_mask[:, :, chunk_idx]  # [B, H, L, L]
        ge_c = g_exp[:, :, chunk_idx]  # [B, H, L]
        kcd_c = k_cumdecay[:, :, chunk_idx]  # [B, H, L, Dk]
        gel_c = g_exp_last[:, :, chunk_idx]  # [B, H, 1]
        kd_c = k_decay_all[:, :, chunk_idx]  # [B, H, L, 1]

        # Intra-chunk attention
        qk = jnp.einsum("bhid,bhjd->bhij", q_c, k_c)
        mask_upper = jnp.triu(jnp.ones((chunk_size, chunk_size), dtype=jnp.bool_), k=1)
        intra_attn = jnp.where(mask_upper, 0.0, qk * dm_c)

        # State correction: v' = v_corrected - k_cumdecay @ state
        v_prime = v_c - jnp.einsum("bhld,bhde->bhle", kcd_c, state)

        # Inter-chunk: q * exp(g) @ state
        inter_out = jnp.einsum("bhld,bhde->bhle", q_c, state)
        inter_out = inter_out * ge_c[..., None]

        # Combine
        output = inter_out + jnp.einsum("bhij,bhjd->bhid", intra_attn, v_prime)

        # State update (using precomputed decay values)
        new_state = (
            state * gel_c[..., None]
            + jnp.einsum("bhld,bhle->bhde", k_c * kd_c, v_prime)
        )

        return new_state, output

    _, outputs = jax.lax.scan(
        chunk_step, init_state, jnp.arange(num_chunks)
    )

    # outputs: [C, B, H, L, Dv] -> [B, H, S_padded, Dv]
    outputs = jnp.transpose(outputs, (1, 2, 0, 3, 4))  # [B, H, C, L, Dv]
    outputs = outputs.reshape(B, H, S_padded, Dv)

    # Remove padding
    if pad_len > 0:
        outputs = outputs[:, :, :S, :]

    return outputs


class GatedDeltaRuleAttention(nnx.Module):
    """Gated Delta Rule linear attention (Qwen3-Next).

    Pure JAX implementation matching HF's Qwen3NextGatedDeltaNet.
    """

    def __init__(self, config: MinTextConfig, *, rngs: nnx.Rngs):
        self.dtype = getattr(jnp, config.dtype)
        weight_dtype = getattr(jnp, config.weight_dtype)

        hidden_size = config.hidden_size
        num_k_heads = config.linear_num_key_heads or config.num_attention_heads
        num_v_heads = config.linear_num_value_heads or config.num_attention_heads
        key_dim = num_k_heads * config.linear_key_head_dim
        value_dim = num_v_heads * config.linear_value_head_dim

        self.num_k_heads = num_k_heads
        self.num_v_heads = num_v_heads
        self.key_head_dim = config.linear_key_head_dim
        self.value_head_dim = config.linear_value_head_dim
        self.v_heads_per_k_group = num_v_heads // num_k_heads
        self.chunk_size = 64

        conv_dim = key_dim * 2 + value_dim

        # Input projections: Q, K, V, Z (gate) — grouped layout matching HF
        self.in_proj_qkvz = Linear(
            hidden_size, key_dim * 2 + value_dim * 2,
            dtype=self.dtype, weight_dtype=weight_dtype,
            kernel_axes=("embed", "mlp"), rngs=rngs,
        )
        # Beta and decay projections
        self.in_proj_ba = Linear(
            hidden_size, num_v_heads * 2,
            dtype=self.dtype, weight_dtype=weight_dtype,
            kernel_axes=("embed", "mlp"), rngs=rngs,
        )

        # Causal depthwise conv1d weights for Q, K, V
        conv_kernel_size = config.linear_conv_kernel_dim
        self.conv_weight = nnx.Param(
            jax.random.normal(
                rngs.params(),
                (conv_dim, 1, conv_kernel_size),
                dtype=weight_dtype,
            ) * 0.02,
        )

        # Decay parameters
        self.A_log = nnx.Param(
            jnp.log(0.5 + jax.random.uniform(rngs.params(), (num_v_heads,)) * 15.5),
        )
        self.dt_bias = nnx.Param(jnp.ones(num_v_heads))

        # Output norm and projection
        self.norm = RMSNorm(config.linear_value_head_dim, config.rms_norm_eps, self.dtype, rngs=rngs)
        self.out_proj = Linear(
            value_dim, hidden_size,
            dtype=self.dtype, weight_dtype=weight_dtype,
            kernel_axes=("mlp", "embed"), rngs=rngs,
        )

    def _split_qkvz(self, qkvz: jax.Array) -> tuple:
        """Split QKVZ using HF's grouped layout.

        HF reshapes to (B, S, num_k_heads, group_size) then splits per group.
        """
        B, S, _ = qkvz.shape
        v_per_group = self.v_heads_per_k_group * self.value_head_dim
        group_size = 2 * self.key_head_dim + 2 * v_per_group

        # Reshape to groups: (B, S, num_k_heads, group_size)
        grouped = qkvz.reshape(B, S, self.num_k_heads, group_size)

        # Split within each group
        splits = [self.key_head_dim, self.key_head_dim, v_per_group, v_per_group]
        q, k, v, z = jnp.split(
            grouped,
            [splits[0], splits[0] + splits[1], splits[0] + splits[1] + splits[2]],
            axis=-1,
        )

        # Reshape v and z from (B, S, num_k_heads, v_per_group) to (B, S, num_v_heads, head_v_dim)
        v = v.reshape(B, S, self.num_v_heads, self.value_head_dim)
        z = z.reshape(B, S, self.num_v_heads, self.value_head_dim)

        return q, k, v, z

    def _split_ba(self, ba: jax.Array) -> tuple:
        """Split beta/alpha using HF's grouped layout."""
        B, S, _ = ba.shape
        ba_per_group = 2 * self.v_heads_per_k_group

        # Reshape to groups: (B, S, num_k_heads, ba_per_group)
        grouped = ba.reshape(B, S, self.num_k_heads, ba_per_group)

        # Split
        b, a = jnp.split(grouped, [self.v_heads_per_k_group], axis=-1)

        # Reshape to (B, S, num_v_heads)
        b = b.reshape(B, S, self.num_v_heads)
        a = a.reshape(B, S, self.num_v_heads)

        return b, a

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass matching HF's Qwen3NextGatedDeltaNet.forward."""
        B, S, D = x.shape

        # 1. Project and split using grouped layout (matching HF)
        qkvz = self.in_proj_qkvz(x)
        q, k, v, z = self._split_qkvz(qkvz)
        # q, k: (B, S, num_k_heads, key_head_dim)
        # v, z: (B, S, num_v_heads, value_head_dim)

        # 2. Split beta and alpha using grouped layout
        ba = self.in_proj_ba(x)
        beta_raw, a = self._split_ba(ba)
        beta = jax.nn.sigmoid(jnp.asarray(beta_raw, jnp.float32))

        # 3. Flatten q, k, v for conv1d, then re-split
        key_total = self.num_k_heads * self.key_head_dim
        value_total = self.num_v_heads * self.value_head_dim
        q_flat = q.reshape(B, S, key_total)
        k_flat = k.reshape(B, S, key_total)
        v_flat = v.reshape(B, S, value_total)
        qkv = jnp.concatenate([q_flat, k_flat, v_flat], axis=-1)

        # Causal conv1d + SiLU activation (matching HF)
        qkv = _causal_conv1d(qkv, jnp.asarray(self.conv_weight[...], self.dtype))
        qkv = jax.nn.silu(qkv)

        q_conv, k_conv, v_conv = jnp.split(qkv, [key_total, key_total * 2], axis=-1)

        # Reshape to heads
        q_h = q_conv.reshape(B, S, self.num_k_heads, self.key_head_dim)
        k_h = k_conv.reshape(B, S, self.num_k_heads, self.key_head_dim)
        v_h = v_conv.reshape(B, S, self.num_v_heads, self.value_head_dim)

        # 4. Compute decay: g = -exp(A_log) * softplus(a + dt_bias)
        A = jnp.exp(jnp.asarray(self.A_log[...], jnp.float32))
        g = -A[None, None, :] * jax.nn.softplus(
            jnp.asarray(a, jnp.float32) + self.dt_bias[...][None, None, :]
        )

        # 5. GQA: repeat K heads to match V heads if needed
        if self.v_heads_per_k_group > 1:
            q_h = jnp.repeat(q_h, self.v_heads_per_k_group, axis=2)
            k_h = jnp.repeat(k_h, self.v_heads_per_k_group, axis=2)

        # 6. Transpose to [B, H, S, D] for chunk_gated_delta_rule
        q_t = jnp.transpose(jnp.asarray(q_h, jnp.float32), (0, 2, 1, 3))
        k_t = jnp.transpose(jnp.asarray(k_h, jnp.float32), (0, 2, 1, 3))
        v_t = jnp.transpose(jnp.asarray(v_h, jnp.float32), (0, 2, 1, 3))
        beta_t = jnp.transpose(beta, (0, 2, 1))
        g_t = jnp.transpose(g, (0, 2, 1))

        # 7. Chunk-based Gated Delta Rule (L2 norm + scaling inside)
        output = chunk_gated_delta_rule(
            q_t, k_t, v_t, beta_t, g_t,
            chunk_size=self.chunk_size,
        )
        # output: [B, H, S, Dv]
        output = jnp.transpose(output, (0, 2, 1, 3))  # [B, S, H, Dv]
        output = jnp.asarray(output, self.dtype)

        # 8. Gated RMSNorm + output projection
        output = self.norm(output)  # norm on last dim (value_head_dim)
        output = output * jax.nn.silu(jnp.asarray(z, self.dtype))

        # Flatten heads and project
        output = output.reshape(B, S, -1)  # [B, S, value_dim]
        return self.out_proj(output)
