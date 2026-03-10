"""Attention layers: standard MHA/GQA with Splash Attention on TPU."""

from __future__ import annotations

import logging

import jax
import jax.numpy as jnp
from flax import nnx
from jax.ad_checkpoint import checkpoint_name

from mintext.config import MinTextConfig
from mintext.modules.linear import Linear
from mintext.modules.norm import RMSNorm
from mintext.modules.rope import RotaryEmbedding

logger = logging.getLogger(__name__)

# Module-level cache for Splash Attention functions.
# Stored here (not on the nnx.Module) to avoid Splash's internal int8/int32
# mask arrays being captured in the NNX state tree, which breaks jax.grad.
_SPLASH_FN_CACHE: dict[tuple, object] = {}

# --- Optional tokamax import ---
_TOKAMAX_AVAILABLE = False
try:
    import tokamax
    # Pre-parse absl flags to avoid conflicts with pytest/other CLI tools.
    # Tokamax uses absl.flags internally via config_lib.
    try:
        from absl import flags as _absl_flags
        if not _absl_flags.FLAGS.is_parsed():
            _absl_flags.FLAGS.mark_as_parsed()
    except Exception:
        pass
    _TOKAMAX_AVAILABLE = True
except ImportError:
    pass

# --- Optional Splash Attention import (TPU only) ---
_SPLASH_AVAILABLE = False
try:
    from jax.experimental.pallas.ops.tpu.splash_attention import (
        make_splash_mha_single_device as _make_splash_mha,
        BlockSizes as _SplashBlockSizes,
        CausalMask as _SplashCausalMask,
        LocalMask as _SplashLocalMask,
        MultiHeadMask as _SplashMultiHeadMask,
    )
    _SPLASH_AVAILABLE = True
except ImportError:
    pass


def _create_splash_fn(
    seq_len: int,
    num_heads: int,
    sliding_window: int | None = None,
    block_q: int = 512,
    block_kv: int = 512,
):
    """Create a Splash Attention kernel function for the given config.

    Returns a callable that takes (q, k, v) in [H, S, D] layout and returns [H, S, D].
    Splash does NOT scale internally — caller must pre-scale Q.
    """
    if sliding_window is not None:
        mask = _SplashLocalMask(
            shape=(seq_len, seq_len),
            window_size=(sliding_window, 0),  # causal + sliding
        )
    else:
        mask = _SplashCausalMask(shape=(seq_len, seq_len))
    multi_mask = _SplashMultiHeadMask(masks=[mask] * num_heads)
    block_sizes = _SplashBlockSizes(
        block_q=min(block_q, seq_len),
        block_kv=min(block_kv, seq_len),
        block_kv_compute=min(block_kv, seq_len),
        block_q_dkv=min(block_q, seq_len),
        block_kv_dkv=min(block_kv, seq_len),
        block_kv_dkv_compute=min(block_kv, seq_len),
        block_q_dq=min(block_q, seq_len),
        block_kv_dq=min(block_kv, seq_len),
    )
    return _make_splash_mha(
        multi_mask,
        block_sizes=block_sizes,
        head_shards=1,
        q_seq_shards=1,
        residual_checkpoint_name="context",
    )


def make_sliding_window_mask(seq_len: int, window_size: int, dtype: jnp.dtype = jnp.float32) -> jax.Array:
    """Create a sliding window mask overlay (additive, -1e10 for blocked positions)."""
    row_idx = jnp.arange(seq_len)[:, None]
    col_idx = jnp.arange(seq_len)[None, :]
    distance = row_idx - col_idx
    window_mask = jnp.where((distance >= 0) & (distance < window_size), 0.0, -1e10)
    return window_mask[jnp.newaxis, jnp.newaxis, :, :].astype(dtype)


class Attention(nnx.Module):
    """Multi-head / Grouped-query attention with RoPE.

    Supports GQA when num_key_value_heads < num_attention_heads, optional Q/K norm and sliding window.
    Uses Splash Attention on TPU for O(N) memory, falls back to jax.nn.dot_product_attention on GPU.
    """

    def __init__(
        self,
        config: MinTextConfig,
        *,
        rngs: nnx.Rngs,
        mesh: jax.sharding.Mesh | None = None,
        use_qk_norm: bool = False,
        sliding_window: int | None = None,
        use_gate: bool = False,
        rope_local_theta: float | None = None,
    ):
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.dtype = config.jnp_dtype
        self.use_qk_norm = use_qk_norm
        self.sliding_window = sliding_window
        self.use_custom_kernels = config.use_custom_kernels
        self.use_gate = use_gate
        self.attn_logit_softcapping = config.attn_logit_softcapping
        self._mesh = mesh

        embed_dim = config.hidden_size
        weight_dtype = config.jnp_weight_dtype

        # Attention scale: query_pre_attn_scalar overrides default 1/sqrt(head_dim)
        query_pre_attn_scalar = config.query_pre_attn_scalar
        self.scale = query_pre_attn_scalar ** -0.5 if query_pre_attn_scalar is not None else config.head_dim ** -0.5

        # Gated attention: Q proj outputs 2x (Q + gate), fused
        q_head_dim = config.head_dim * 2 if use_gate else config.head_dim
        self.query = Linear(
            embed_dim,
            (config.num_attention_heads, q_head_dim),
            dtype=self.dtype,
            weight_dtype=weight_dtype,
            kernel_axes=("embed", "heads", "kv"),
            rngs=rngs,
        )
        self.key = Linear(
            embed_dim,
            (config.num_key_value_heads, config.head_dim),
            dtype=self.dtype,
            weight_dtype=weight_dtype,
            kernel_axes=("embed", "kv_heads", "kv"),
            rngs=rngs,
        )
        self.value = Linear(
            embed_dim,
            (config.num_key_value_heads, config.head_dim),
            dtype=self.dtype,
            weight_dtype=weight_dtype,
            kernel_axes=("embed", "kv_heads", "kv"),
            rngs=rngs,
        )
        self.out = Linear(
            (config.num_attention_heads, config.head_dim),
            embed_dim,
            dtype=self.dtype,
            weight_dtype=weight_dtype,
            kernel_axes=("heads", "kv", "embed"),
            rngs=rngs,
        )

        # Dual RoPE: use a separate rope_theta for local/sliding layers
        self.use_local_rope = rope_local_theta is not None
        if self.use_local_rope:
            # Create a config copy with overridden rope_theta for local layers
            local_cfg = config.model_copy(update={"rope_theta": rope_local_theta})
            self.rope = RotaryEmbedding(local_cfg)
        else:
            self.rope = RotaryEmbedding(config)

        if use_qk_norm:
            self.q_norm = RMSNorm(config.head_dim, config.rms_norm_eps, self.dtype, rngs=rngs)
            self.k_norm = RMSNorm(config.head_dim, config.rms_norm_eps, self.dtype, rngs=rngs)

        # Setup Splash Attention for TPU (O(N) memory Flash Attention via Pallas).
        # The splash function is stored in a module-level cache (not on self)
        # to prevent NNX from including Splash's int8/int32 mask arrays in the
        # module state tree, which would break jax.grad.
        self._splash_key: tuple | None = None
        if (
            _SPLASH_AVAILABLE
            and config.attn_logit_softcapping is None
            and not use_gate  # Gated attention needs custom handling
            # JAX 0.9.1+ Splash uses ceiling division for head_dim (supports 64, 128, etc.)
        ):
            try:
                if jax.devices()[0].platform == "tpu":
                    cache_key = (config.max_position_embeddings, config.num_attention_heads, sliding_window)
                    if cache_key not in _SPLASH_FN_CACHE:
                        _SPLASH_FN_CACHE[cache_key] = _create_splash_fn(
                            seq_len=config.max_position_embeddings,
                            num_heads=config.num_attention_heads,
                            sliding_window=sliding_window,
                        )
                    self._splash_key = cache_key
                    logger.info(
                        "Splash Attention enabled (seq=%d, heads=%d)",
                        config.max_position_embeddings, config.num_attention_heads,
                    )
            except Exception as e:
                logger.warning("Failed to setup Splash Attention: %s", e)

    def __call__(
        self,
        x: jax.Array,
        positions: jax.Array,
        mask: jax.Array | None = None,
    ) -> jax.Array:
        """Forward pass.

        Args:
            x: [batch, seq_len, embed_dim]
            positions: [batch, seq_len]
            mask: [batch, 1, seq_len, seq_len] causal mask (0 = attend, -inf = block)

        Returns:
            Output tensor [batch, seq_len, embed_dim]
        """
        q_raw = self.query(x)  # [B, S, num_attention_heads, head_dim] or [B, S, num_attention_heads, head_dim*2]
        k = self.key(x)  # [B, S, num_key_value_heads, head_dim]
        v = self.value(x)  # [B, S, num_key_value_heads, head_dim]

        # Split fused Q+gate for gated attention
        if self.use_gate:
            q, gate = jnp.split(q_raw, 2, axis=-1)  # each [B, S, num_attention_heads, head_dim]
        else:
            q = q_raw

        # Checkpoint names for remat policies
        q = checkpoint_name(q, "query_proj")
        k = checkpoint_name(k, "key_proj")
        v = checkpoint_name(v, "value_proj")

        # Q/K norm - applied before RoPE
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Apply RoPE
        q = self.rope(q, positions)
        k = self.rope(k, positions)

        _used_tokamax = False
        if self.use_custom_kernels and _TOKAMAX_AVAILABLE:
            try:
                if self.num_key_value_heads < self.num_attention_heads:
                    repeats = self.num_attention_heads // self.num_key_value_heads
                    k_for_tokamax = jnp.repeat(k, repeats, axis=2)
                    v_for_tokamax = jnp.repeat(v, repeats, axis=2)
                else:
                    k_for_tokamax = k
                    v_for_tokamax = v
                # tokamax expects [B, T, N, H] layout
                attn_output = tokamax.dot_product_attention(
                    q, k_for_tokamax, v_for_tokamax, is_causal=True, scale=self.scale,
                    local_window_size=self.sliding_window,
                )
                _used_tokamax = True
            except (ValueError, NotImplementedError, TypeError):
                pass

        if not _used_tokamax:
            _splash_fn = _SPLASH_FN_CACHE.get(self._splash_key) if self._splash_key is not None else None
            if _splash_fn is not None and self._mesh is not None:
                # Splash Attention (TPU): O(N) memory, Pallas kernel via shard_map
                # Splash does NOT scale internally — pre-scale Q
                # Splash supports GQA natively (Q[H,S,D], K/V[KV_H,S,D])
                from jax.experimental.shard_map import shard_map
                q_scaled = q * self.scale
                # Transpose [B, S, H, D] -> [B, H, S, D] for Splash
                q_t = jnp.transpose(q_scaled, (0, 2, 1, 3))
                k_t = jnp.transpose(k, (0, 2, 1, 3))
                v_t = jnp.transpose(v, (0, 2, 1, 3))
                # Use shard_map to wrap Pallas kernel for multi-device
                # Batch is on 'data' axis; heads/seq/dim are replicated
                mesh = self._mesh
                # Determine batch sharding: use 'data' if size > 1, else replicate
                batch_axis = "data" if mesh.shape.get("data", 1) > 1 else None
                qkv_spec = jax.sharding.PartitionSpec(batch_axis, None, None, None)

                def _splash_attn_fn(q_s, k_s, v_s):
                    return jax.vmap(_splash_fn)(q_s, k_s, v_s)

                _splash_attn = shard_map(
                    _splash_attn_fn, mesh=mesh,
                    in_specs=(qkv_spec, qkv_spec, qkv_spec),
                    out_specs=qkv_spec, check_rep=False,
                )
                attn_output = _splash_attn(q_t, k_t, v_t)
                # Back to [B, S, H, D]
                attn_output = jnp.transpose(attn_output, (0, 2, 1, 3))
            elif self.attn_logit_softcapping is not None:
                # Manual attention with soft-capping (Gemma-style)
                q_t = jnp.transpose(q, (0, 2, 1, 3))
                k_t = jnp.transpose(k, (0, 2, 1, 3))
                v_t = jnp.transpose(v, (0, 2, 1, 3))

                if self.num_key_value_heads < self.num_attention_heads:
                    groups = self.num_attention_heads // self.num_key_value_heads
                    q_t = q_t.reshape(
                        q_t.shape[0], self.num_key_value_heads, groups,
                        q_t.shape[2], q_t.shape[3],
                    )
                    attn_weights = jnp.einsum("bngsd,bntd->bngst", q_t, k_t) * self.scale
                else:
                    attn_weights = jnp.matmul(q_t, jnp.swapaxes(k_t, -2, -1)) * self.scale
                attn_weights = jnp.asarray(attn_weights, jnp.float32)

                cap = self.attn_logit_softcapping
                attn_weights = jnp.tanh(attn_weights / cap) * cap

                if mask is not None:
                    if self.num_key_value_heads < self.num_attention_heads:
                        attn_weights = attn_weights + mask[:, None]
                    else:
                        attn_weights = attn_weights + mask

                if self.sliding_window is not None:
                    seq_len_sw = attn_weights.shape[-1]
                    sw_mask = make_sliding_window_mask(seq_len_sw, self.sliding_window)
                    if self.num_key_value_heads < self.num_attention_heads:
                        attn_weights = attn_weights + sw_mask[:, None]
                    else:
                        attn_weights = attn_weights + sw_mask

                attn_weights = jax.nn.softmax(attn_weights, axis=-1)
                attn_weights = jnp.asarray(attn_weights, self.dtype)

                if self.num_key_value_heads < self.num_attention_heads:
                    attn_output = jnp.einsum("bngst,bntd->bngsd", attn_weights, v_t)
                    attn_output = attn_output.reshape(
                        attn_output.shape[0], self.num_attention_heads,
                        attn_output.shape[3], attn_output.shape[4],
                    )
                else:
                    attn_output = jnp.matmul(attn_weights, v_t)
                attn_output = jnp.transpose(attn_output, (0, 2, 1, 3))
            else:
                # Efficient path: jax.nn.dot_product_attention (GPU/CPU)
                # Handles GQA, causal masking, and scaling natively.
                attn_output = jax.nn.dot_product_attention(
                    q, k, v,
                    scale=self.scale,
                    is_causal=True,
                    local_window_size=self.sliding_window,
                )

        # Apply gated attention: sigmoid gate on attention output
        if self.use_gate:
            attn_output = attn_output * jax.nn.sigmoid(gate)

        output = self.out(attn_output)
        if self._mesh is not None:
            output = jax.lax.with_sharding_constraint(
                output,
                jax.sharding.NamedSharding(
                    self._mesh,
                    jax.sharding.PartitionSpec("data", None, "fsdp"),
                ),
            )
        output = checkpoint_name(output, "out_proj")
        return output
