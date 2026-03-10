"""Core Transformer model supporting Llama3, Qwen3, DeepSeek-V3, Qwen3-Next, and Gemma3."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from mintext.config import MinTextConfig
from mintext.modules.attention import Attention
from mintext.modules.linear import Linear, MLP
from mintext.modules.mla import MLAttention
from mintext.modules.linear_attention import GatedDeltaRuleAttention
from mintext.modules.moe import MoEBlock
from mintext.modules.norm import RMSNorm


def _all_layers_scannable(config: MinTextConfig) -> bool:
    """Check if all layers are structurally identical (same attention + MLP type).

    Scan requires homogeneous layers since parameters are stacked along a
    leading axis. Mixed layer types (e.g., full + linear attention, dense + MoE)
    are not compatible.
    """
    if not config.scan_layers:
        return False
    if config.num_hidden_layers <= 1:
        return False
    # All layers must have the same attention type
    layer_types = config.layer_types
    if layer_types and len(set(layer_types)) > 1:
        return False
    # MoE layers are not scan-compatible (aux output varies)
    if config.num_experts > 0:
        return False
    return True


def _get_remat_kwargs(policy_name: str) -> dict | None:
    """Return kwargs for jax.checkpoint or None for the given policy name."""
    if policy_name == "full":
        return {}
    elif policy_name == "minimal":
        return {"policy": jax.checkpoint_policies.dots_with_no_batch_dims_saveable}
    elif policy_name == "save_qkv_proj":
        return {"policy": jax.checkpoint_policies.save_only_these_names(
            "query_proj", "key_proj", "value_proj",
        )}
    elif policy_name == "save_dot_except_mlp":
        return {"policy": jax.checkpoint_policies.save_only_these_names(
            "query_proj", "key_proj", "value_proj", "out_proj",
        )}
    elif policy_name == "qkv_proj_offloaded":
        return {"policy": jax.checkpoint_policies.save_and_offload_only_these_names(
            names_which_can_be_saved=[],
            names_which_can_be_offloaded=["query_proj", "key_proj", "value_proj"],
            offload_src="device", offload_dst="pinned_host",
        )}
    elif policy_name == "minimal_offloaded":
        return {"policy": jax.checkpoint_policies.save_and_offload_only_these_names(
            names_which_can_be_saved=[],
            names_which_can_be_offloaded=[
                "query_proj", "key_proj", "value_proj",
                "out_proj", "mlp_gate", "mlp_up", "mlp_down",
            ],
            offload_src="device", offload_dst="pinned_host",
        )}
    return None


def _apply_layer_with_remat(layer, x, positions, mask, remat_policy):
    """Apply a decoder layer with the appropriate remat (gradient checkpointing) policy."""
    remat_kwargs = _get_remat_kwargs(remat_policy)
    if remat_kwargs is not None:
        def _checkpointed_layer(x, positions, mask):
            return layer(x, positions, mask)

        _checkpointed_layer = jax.checkpoint(_checkpointed_layer, **remat_kwargs)
        return _checkpointed_layer(x, positions, mask)
    return layer(x, positions, mask)


class DecoderLayer(nnx.Module):
    """Unified decoder layer supporting all architectures.

    Selects attention type (standard, MLA, linear) and MLP type (dense, MoE)
    based on config and layer index.
    """

    def __init__(self, config: MinTextConfig, layer_idx: int, *, rngs: nnx.Rngs, mesh: jax.sharding.Mesh | None = None):
        self._mesh = mesh
        dtype = config.jnp_dtype
        layer_type = config.layer_types[layer_idx] if config.layer_types else "full_attention"

        self.use_post_ffw_norm = config.use_post_ffw_norm

        self.pre_attn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps, dtype, rngs=rngs)
        if self.use_post_ffw_norm:
            # 4-norm pattern: post_attn_norm applied after attention (before residual)
            self.post_attn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps, dtype, rngs=rngs)
            self.pre_ffw_norm = RMSNorm(config.hidden_size, config.rms_norm_eps, dtype, rngs=rngs)
            self.post_ffw_norm = RMSNorm(config.hidden_size, config.rms_norm_eps, dtype, rngs=rngs)
        else:
            # Standard 2-norm: post_attn_norm is pre-FFW norm
            self.post_attn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps, dtype, rngs=rngs)

        # --- Attention selection ---
        self.is_linear = False
        if layer_type == "linear_attention":
            self.attention = GatedDeltaRuleAttention(config, rngs=rngs)
            self.is_linear = True
        elif config.attention_type == "mla":
            self.attention = MLAttention(config, layer_idx, rngs=rngs)
        else:
            # Standard MHA/GQA, optionally with Q/K norm and sliding window
            sliding_window = None
            rope_local_theta = None
            if layer_type == "sliding_attention":
                sliding_window = config.sliding_window
                # Sliding layers may use a different rope_theta
                if config.model_type == "gemma3":
                    rope_local_theta = config.rope_local_theta
            self.attention = Attention(
                config, rngs=rngs, mesh=mesh,
                use_qk_norm=config.use_qk_norm,
                sliding_window=sliding_window,
                use_gate=(config.model_type == "qwen3_next" and layer_type == "full_attention"),
                rope_local_theta=rope_local_theta,
            )

        # --- MLP selection ---
        self.has_moe = False
        if config.num_experts > 0 and layer_idx >= config.first_k_dense_replace:
            self.mlp = MoEBlock(config, rngs=rngs)
            self.has_moe = True
        else:
            self.mlp = MLP(config, rngs=rngs, mesh=mesh)

    def __call__(
        self,
        x: jax.Array,
        positions: jax.Array,
        mask: jax.Array | None = None,
    ) -> tuple[jax.Array, dict | None]:
        if self.use_post_ffw_norm:
            # 4-norm pattern
            h = self.pre_attn_norm(x)
            if self.is_linear:
                h = self.attention(h)
            else:
                h = self.attention(h, positions, mask)
            h = self.post_attn_norm(h)
            x = x + h

            h = self.pre_ffw_norm(x)
            if self.has_moe:
                h, aux = self.mlp(h)
            else:
                h = self.mlp(h)
                aux = None
            h = self.post_ffw_norm(h)
            x = x + h
        else:
            # Standard 2-norm
            h = self.pre_attn_norm(x)
            if self.is_linear:
                h = self.attention(h)
            else:
                h = self.attention(h, positions, mask)
            x = x + h

            h = self.post_attn_norm(x)
            if self.has_moe:
                h, aux = self.mlp(h)
            else:
                h = self.mlp(h)
                aux = None
            x = x + h

        return x, aux


class Transformer(nnx.Module):
    """Multi-architecture decoder-only Transformer.

    Supports Llama3, Qwen3, DeepSeek-V3, Qwen3-Next, and Gemma3 via config.model_type.
    """

    def __init__(self, config: MinTextConfig, *, rngs: nnx.Rngs, mesh: jax.sharding.Mesh | None = None):
        self.config = config
        self.final_logit_softcapping = config.final_logit_softcapping
        dtype = config.jnp_dtype
        weight_dtype = config.jnp_weight_dtype

        # Compute in model dtype to match HF bfloat16 rounding (e.g. sqrt(2560) → 50.5)
        self.embedding_scale = jnp.array(config.hidden_size, dtype=dtype) ** 0.5 if config.scale_embeddings else None

        # Token embedding
        self.token_embedder = nnx.Param(
            jax.random.normal(rngs.params(), (config.vocab_size, config.hidden_size), dtype=weight_dtype)
            * 0.02,
            sharding=("vocab", "embed"),
        )

        # Decoder layers
        self._use_scan = _all_layers_scannable(config)

        if self._use_scan:
            # Create layers with stacked params via nnx.vmap for use with nnx.scan.
            # All layers are structurally identical (verified by _all_layers_scannable).
            @nnx.split_rngs(splits=config.num_hidden_layers)
            @nnx.vmap(in_axes=(0,), out_axes=0)
            def _create_layer(rngs: nnx.Rngs):
                return DecoderLayer(config, layer_idx=0, rngs=rngs, mesh=mesh)

            self.layers = _create_layer(rngs)
        else:
            self.layers = nnx.List([
                DecoderLayer(config, layer_idx=i, rngs=rngs, mesh=mesh)
                for i in range(config.num_hidden_layers)
            ])

        # Remat policy for gradient checkpointing
        self.remat_policy = config.remat_policy

        # Final norm
        self.final_norm = RMSNorm(config.hidden_size, config.rms_norm_eps, dtype, rngs=rngs)

        # Output head
        self.tie_word_embeddings = config.tie_word_embeddings
        if not config.tie_word_embeddings:
            self.output_proj = Linear(
                config.hidden_size,
                config.vocab_size,
                dtype=dtype,
                weight_dtype=weight_dtype,
                kernel_axes=("embed", "vocab"),
                rngs=rngs,
            )

    def __call__(
        self,
        tokens: jax.Array,
        positions: jax.Array,
        mask: jax.Array | None = None,
        return_hidden: bool = False,
    ) -> tuple[jax.Array, list[dict | None]]:
        """Forward pass.

        Args:
            tokens: [batch, seq_len] integer token ids
            positions: [batch, seq_len] position indices
            mask: [batch, 1, seq_len, seq_len] causal mask (additive, 0 or -inf)
            return_hidden: If True, return hidden states before output projection
                instead of logits. Used for vocab tiling.

        Returns:
            (logits_or_hidden, aux_data_list) where the first element is
            [batch, seq_len, vocab_size] logits (default) or
            [batch, seq_len, hidden_size] hidden states (if return_hidden=True).
        """
        dtype = getattr(jnp, self.config.dtype)

        # Embed tokens
        x = jnp.take(self.token_embedder[...], tokens, axis=0)
        x = jnp.asarray(x, dtype)

        # Embedding scaling
        if self.embedding_scale is not None:
            x = x * self.embedding_scale

        # Decoder layers (with optional gradient checkpointing)
        if self._use_scan:
            x, all_aux = self._scan_forward(x, positions, mask)
        else:
            all_aux = []
            for layer in self.layers:
                x, aux = _apply_layer_with_remat(layer, x, positions, mask, self.remat_policy)
                all_aux.append(aux)

        # Final norm
        x = self.final_norm(x)

        if return_hidden:
            return x, all_aux

        # Output logits
        if self.tie_word_embeddings:
            logits = jnp.dot(x, jnp.asarray(self.token_embedder[...], dtype).T)
        else:
            logits = self.output_proj(x)

        # Final logit soft-capping
        if self.final_logit_softcapping is not None:
            cap = self.final_logit_softcapping
            logits = jnp.tanh(logits / cap) * cap

        return logits, all_aux

    def _scan_forward(
        self, x: jax.Array, positions: jax.Array, mask: jax.Array | None,
    ) -> tuple[jax.Array, list[None]]:
        """Forward pass through decoder layers using nnx.scan.

        Only used when all layers are structurally identical (no MoE).
        Reduces compilation time by compiling a single layer body.
        """
        remat_policy = self.remat_policy
        remat_kwargs = _get_remat_kwargs(remat_policy)

        if remat_kwargs is not None:

            @nnx.scan(in_axes=(nnx.Carry, 0, None, None), out_axes=nnx.Carry)
            def scan_fn(x, layer, positions, mask):
                def _fn(x, positions, mask):
                    return layer(x, positions, mask)
                _fn = jax.checkpoint(_fn, **remat_kwargs)
                x, _ = _fn(x, positions, mask)
                return x
        else:
            @nnx.scan(in_axes=(nnx.Carry, 0, None, None), out_axes=nnx.Carry)
            def scan_fn(x, layer, positions, mask):
                x, _ = layer(x, positions, mask)
                return x

        x = scan_fn(x, self.layers, positions, mask)
        all_aux = [None] * self.config.num_hidden_layers
        return x, all_aux


def make_causal_mask(seq_len: int, dtype: jnp.dtype = jnp.float32) -> jax.Array:
    """Create a causal attention mask.

    Returns:
        Mask of shape [1, 1, seq_len, seq_len] with 0 for attend and -1e10 for block.
    """
    mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=dtype))
    mask = (1.0 - mask) * -1e10
    return mask[jnp.newaxis, jnp.newaxis, :, :]
