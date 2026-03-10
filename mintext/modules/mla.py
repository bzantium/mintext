"""Multi-head Latent Attention (MLA) for DeepSeek-V3."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx
from jax.ad_checkpoint import checkpoint_name

from mintext.config import MinTextConfig
from mintext.modules.linear import Linear
from mintext.modules.norm import RMSNorm
from mintext.modules.rope import RotaryEmbedding


class MLAttention(nnx.Module):
    """Multi-head Latent Attention (DeepSeek-V3).

    Q path: hidden -> q_a_proj -> q_a_norm -> q_b_proj -> split(q_nope, q_rope)
    KV path: hidden -> kv_a_proj -> split(kv_compressed, k_rope)
             kv_compressed -> kv_a_norm -> kv_b_proj -> split(k_nope, values)
    RoPE applied only to q_rope and k_rope portions.
    """

    def __init__(self, config: MinTextConfig, layer_idx: int, *, rngs: nnx.Rngs):
        self.num_attention_heads = config.num_attention_heads
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.dtype = config.jnp_dtype

        embed_dim = config.hidden_size
        weight_dtype = config.jnp_weight_dtype
        qk_head_dim = config.qk_head_dim  # nope + rope

        # Scaling factor
        self.scale = qk_head_dim ** -0.5

        # Q path: compress -> norm -> expand
        if config.q_lora_rank > 0:
            self.q_a_proj = Linear(
                embed_dim, config.q_lora_rank,
                dtype=self.dtype, weight_dtype=weight_dtype,
                kernel_axes=("embed", "mlp"), rngs=rngs,
            )
            self.q_a_norm = RMSNorm(config.q_lora_rank, config.rms_norm_eps, self.dtype, rngs=rngs)
            self.q_b_proj = Linear(
                config.q_lora_rank, (config.num_attention_heads, qk_head_dim),
                dtype=self.dtype, weight_dtype=weight_dtype,
                kernel_axes=("embed", "heads", "kv"), rngs=rngs,
            )
        else:
            # Direct projection (no LoRA compression)
            self.q_proj = Linear(
                embed_dim, (config.num_attention_heads, qk_head_dim),
                dtype=self.dtype, weight_dtype=weight_dtype,
                kernel_axes=("embed", "heads", "kv"), rngs=rngs,
            )

        # KV path: compress to (kv_lora_rank + qk_rope_head_dim)
        self.kv_a_proj = Linear(
            embed_dim, config.kv_lora_rank + config.qk_rope_head_dim,
            dtype=self.dtype, weight_dtype=weight_dtype,
            kernel_axes=("embed", "mlp"), rngs=rngs,
        )
        self.kv_a_norm = RMSNorm(config.kv_lora_rank, config.rms_norm_eps, self.dtype, rngs=rngs)
        self.kv_b_proj = Linear(
            config.kv_lora_rank,
            (config.num_attention_heads, config.qk_nope_head_dim + config.v_head_dim),
            dtype=self.dtype, weight_dtype=weight_dtype,
            kernel_axes=("embed", "heads", "kv"), rngs=rngs,
        )

        # Output projection
        self.out = Linear(
            (config.num_attention_heads, config.v_head_dim),
            embed_dim,
            dtype=self.dtype, weight_dtype=weight_dtype,
            kernel_axes=("heads", "kv", "embed"), rngs=rngs,
        )

        # RoPE for the rope portion only (head_dim = qk_rope_head_dim)
        self.rope = RotaryEmbedding(config, rope_dim=config.qk_rope_head_dim)

    def __call__(
        self,
        x: jax.Array,
        positions: jax.Array,
        mask: jax.Array | None = None,
    ) -> jax.Array:
        B, S, _ = x.shape

        # Q path
        if self.q_lora_rank > 0:
            q = self.q_b_proj(self.q_a_norm(self.q_a_proj(x)))  # [B, S, H, qk_head_dim]
        else:
            q = self.q_proj(x)

        q = checkpoint_name(q, "query_proj")
        q_nope, q_rope = jnp.split(q, [self.qk_nope_head_dim], axis=-1)

        # KV path: compress
        kv_compressed = self.kv_a_proj(x)  # [B, S, kv_lora_rank + qk_rope_head_dim]
        kv_compressed = checkpoint_name(kv_compressed, "key_proj")
        kv_latent, k_rope_raw = jnp.split(kv_compressed, [self.kv_lora_rank], axis=-1)

        # Expand KV
        kv_expanded = self.kv_b_proj(self.kv_a_norm(kv_latent))  # [B, S, H, nope+v]
        kv_expanded = checkpoint_name(kv_expanded, "value_proj")
        k_nope, v = jnp.split(kv_expanded, [self.qk_nope_head_dim], axis=-1)

        # Broadcast k_rope_raw [B, S, rope_dim] -> [B, S, H, rope_dim]
        k_rope = jnp.broadcast_to(
            k_rope_raw[:, :, jnp.newaxis, :],
            (B, S, self.num_attention_heads, self.qk_rope_head_dim),
        )

        # Apply RoPE to rope portions only
        q_rope = self.rope(q_rope, positions)
        k_rope = self.rope(k_rope, positions)

        # Concat nope + rope
        q = jnp.concatenate([q_nope, q_rope], axis=-1)  # [B, S, H, qk_head_dim]
        k = jnp.concatenate([k_nope, k_rope], axis=-1)  # [B, S, H, qk_head_dim]

        # Standard attention from here
        q = jnp.transpose(q, (0, 2, 1, 3))  # [B, H, S, D]
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))

        attn_weights = jnp.matmul(q, jnp.swapaxes(k, -2, -1)) * self.scale
        attn_weights = jnp.asarray(attn_weights, jnp.float32)

        if mask is not None:
            attn_weights = attn_weights + mask

        attn_weights = jax.nn.softmax(attn_weights, axis=-1)
        attn_weights = jnp.asarray(attn_weights, self.dtype)

        attn_output = jnp.matmul(attn_weights, v)  # [B, H, S, v_head_dim]
        attn_output = jnp.transpose(attn_output, (0, 2, 1, 3))  # [B, S, H, v_head_dim]

        output = self.out(attn_output)
        output = checkpoint_name(output, "out_proj")
        return output
