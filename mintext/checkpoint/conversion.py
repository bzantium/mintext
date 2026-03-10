"""HuggingFace SafeTensors checkpoint import/export for multiple model types.

Supports Llama, Qwen3, DeepSeek-V3, Qwen3-Next, and Gemma3 architectures.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable

import jax.numpy as jnp
import numpy as np
from flax import nnx
from safetensors.numpy import load_file, save_file

from mintext.config import MinTextConfig
from mintext.models import Transformer

logger = logging.getLogger(__name__)


# --- Transform functions ---


def _transpose(x: np.ndarray) -> np.ndarray:
    """Transpose a 2D weight matrix (HF [out, in] <-> Flax [in, out])."""
    return x.T


def _rmsnorm_hf_to_mt(x: np.ndarray) -> np.ndarray:
    """Qwen3-Next RMSNorm: HF uses (1 + weight), MinText uses scale directly."""
    return x + 1.0


def _rmsnorm_mt_to_hf(x: np.ndarray) -> np.ndarray:
    """Qwen3-Next RMSNorm: MinText scale -> HF weight = scale - 1."""
    return x - 1.0


def _make_qkv_hf_to_mt(num_attention_heads: int, head_dim: int) -> Callable:
    """HF q/k/v: (num_attention_heads*head_dim, hidden) -> MinText: (hidden, num_attention_heads, head_dim)."""
    def transform(x: np.ndarray) -> np.ndarray:
        return x.reshape(num_attention_heads, head_dim, -1).transpose(2, 0, 1)
    return transform


def _make_qkv_mt_to_hf(num_attention_heads: int, head_dim: int) -> Callable:
    """MinText q/k/v: (hidden, num_attention_heads, head_dim) -> HF: (num_attention_heads*head_dim, hidden)."""
    def transform(x: np.ndarray) -> np.ndarray:
        return x.transpose(1, 2, 0).reshape(num_attention_heads * head_dim, -1)
    return transform


def _make_out_hf_to_mt(num_attention_heads: int, head_dim: int) -> Callable:
    """HF o_proj: (hidden, num_attention_heads*head_dim) -> MinText: (num_attention_heads, head_dim, hidden)."""
    def transform(x: np.ndarray) -> np.ndarray:
        return x.T.reshape(num_attention_heads, head_dim, -1)
    return transform


def _make_out_mt_to_hf(num_attention_heads: int, head_dim: int) -> Callable:
    """MinText out: (num_attention_heads, head_dim, hidden) -> HF: (hidden, num_attention_heads*head_dim)."""
    def transform(x: np.ndarray) -> np.ndarray:
        return x.reshape(num_attention_heads * head_dim, -1).T
    return transform


# --- Expert weight conversion (per-expert HF <-> fused MinText) ---


def _get_moe_layer_indices(config: MinTextConfig) -> list[int]:
    """Return list of layer indices that use MoE."""
    if config.num_experts <= 0:
        return []
    return [i for i in range(config.num_hidden_layers) if i >= config.first_k_dense_replace]


def _import_expert_weights(
    hf_params: dict[str, np.ndarray],
    config: MinTextConfig,
    flat_params: dict[str, np.ndarray],
) -> set[str]:
    """Import per-expert HF weights into fused MinText format.

    HF stores experts as individual 2D weights:
        model.layers.N.mlp.experts.E.gate_proj.weight  (moe_intermediate, hidden)
        model.layers.N.mlp.experts.E.up_proj.weight    (moe_intermediate, hidden)
        model.layers.N.mlp.experts.E.down_proj.weight  (hidden, moe_intermediate)

    MinText stores fused 3D:
        layers.N.mlp.experts.gate_up_proj  (num_experts, hidden, 2*moe_intermediate)
        layers.N.mlp.experts.down_proj     (num_experts, moe_intermediate, hidden)

    Returns set of HF keys consumed.
    """
    consumed: set[str] = set()
    for i in _get_moe_layer_indices(config):
        hf_prefix = f"model.layers.{i}.mlp.experts"
        mt_prefix = f"layers.{i}.mlp.experts"

        gate_up_list = []
        down_list = []
        for e in range(config.num_experts):
            gate_key = f"{hf_prefix}.{e}.gate_proj.weight"
            up_key = f"{hf_prefix}.{e}.up_proj.weight"
            down_key = f"{hf_prefix}.{e}.down_proj.weight"

            if gate_key not in hf_params:
                # Try fused format as fallback
                fused_gu = f"{hf_prefix}.gate_up_proj"
                fused_dn = f"{hf_prefix}.down_proj"
                if fused_gu in hf_params:
                    # Fused format: (num_experts, 2*intermediate, hidden) -> (num_experts, hidden, 2*intermediate)
                    flat_params[f"{mt_prefix}.gate_up_proj"] = hf_params[fused_gu].transpose(0, 2, 1)
                    flat_params[f"{mt_prefix}.down_proj"] = hf_params[fused_dn].transpose(0, 2, 1)
                    consumed.add(fused_gu)
                    consumed.add(fused_dn)
                break

            gate = hf_params[gate_key].T  # (hidden, moe_intermediate)
            up = hf_params[up_key].T      # (hidden, moe_intermediate)
            down = hf_params[down_key].T  # (moe_intermediate, hidden)

            gate_up_list.append(np.concatenate([gate, up], axis=-1))
            down_list.append(down)
            consumed.update([gate_key, up_key, down_key])

        if gate_up_list:
            flat_params[f"{mt_prefix}.gate_up_proj"] = np.stack(gate_up_list)
            flat_params[f"{mt_prefix}.down_proj"] = np.stack(down_list)

    return consumed


def _export_expert_weights(
    flat_params: dict[str, np.ndarray],
    config: MinTextConfig,
) -> dict[str, np.ndarray]:
    """Export fused MinText expert weights to per-expert HF format.

    Returns dict of HF key -> weight array.
    """
    hf_expert_params: dict[str, np.ndarray] = {}
    for i in _get_moe_layer_indices(config):
        mt_prefix = f"layers.{i}.mlp.experts"
        hf_prefix = f"model.layers.{i}.mlp.experts"

        gate_up = flat_params.get(f"{mt_prefix}.gate_up_proj")
        down = flat_params.get(f"{mt_prefix}.down_proj")
        if gate_up is None or down is None:
            continue

        moe_dim = gate_up.shape[2] // 2
        for e in range(config.num_experts):
            # gate_up[e] shape: (hidden, 2*moe_intermediate)
            gate = gate_up[e, :, :moe_dim].T  # (moe_intermediate, hidden)
            up = gate_up[e, :, moe_dim:].T     # (moe_intermediate, hidden)
            d = down[e].T                       # (hidden, moe_intermediate)

            hf_expert_params[f"{hf_prefix}.{e}.gate_proj.weight"] = gate
            hf_expert_params[f"{hf_prefix}.{e}.up_proj.weight"] = up
            hf_expert_params[f"{hf_prefix}.{e}.down_proj.weight"] = d

    return hf_expert_params


# --- Per-model key mapping ---


def _llama_hf_to_mt_map(config: MinTextConfig) -> dict[str, tuple[str, Callable | None]]:
    """Llama-family key mapping (HF -> MinText)."""
    mapping: dict[str, tuple[str, Callable | None]] = {}
    num_attention_heads = config.num_attention_heads
    num_key_value_heads = config.num_key_value_heads or config.num_attention_heads
    head_dim = config.head_dim or (config.hidden_size // config.num_attention_heads)

    mapping["model.embed_tokens.weight"] = ("token_embedder", None)

    for i in range(config.num_hidden_layers):
        hf = f"model.layers.{i}"
        mt = f"layers.{i}"

        mapping[f"{hf}.self_attn.q_proj.weight"] = (
            f"{mt}.attention.query.kernel", _make_qkv_hf_to_mt(num_attention_heads, head_dim)
        )
        mapping[f"{hf}.self_attn.k_proj.weight"] = (
            f"{mt}.attention.key.kernel", _make_qkv_hf_to_mt(num_key_value_heads, head_dim)
        )
        mapping[f"{hf}.self_attn.v_proj.weight"] = (
            f"{mt}.attention.value.kernel", _make_qkv_hf_to_mt(num_key_value_heads, head_dim)
        )
        mapping[f"{hf}.self_attn.o_proj.weight"] = (
            f"{mt}.attention.out.kernel", _make_out_hf_to_mt(num_attention_heads, head_dim)
        )

        mapping[f"{hf}.mlp.gate_proj.weight"] = (f"{mt}.mlp.gate.kernel", _transpose)
        mapping[f"{hf}.mlp.up_proj.weight"] = (f"{mt}.mlp.up.kernel", _transpose)
        mapping[f"{hf}.mlp.down_proj.weight"] = (f"{mt}.mlp.down.kernel", _transpose)

        mapping[f"{hf}.input_layernorm.weight"] = (f"{mt}.pre_attn_norm.scale", None)
        mapping[f"{hf}.post_attention_layernorm.weight"] = (f"{mt}.post_attn_norm.scale", None)

    mapping["model.norm.weight"] = ("final_norm.scale", None)
    mapping["lm_head.weight"] = ("output_proj.kernel", _transpose)

    return mapping


def _qwen3_hf_to_mt_map(config: MinTextConfig) -> dict[str, tuple[str, Callable | None]]:
    """Qwen3 key mapping: extends Llama with Q/K norm."""
    mapping = _llama_hf_to_mt_map(config)

    for i in range(config.num_hidden_layers):
        hf = f"model.layers.{i}"
        mt = f"layers.{i}"
        mapping[f"{hf}.self_attn.q_norm.weight"] = (f"{mt}.attention.q_norm.scale", None)
        mapping[f"{hf}.self_attn.k_norm.weight"] = (f"{mt}.attention.k_norm.scale", None)

    return mapping


def _deepseek_v3_hf_to_mt_map(config: MinTextConfig) -> dict[str, tuple[str, Callable | None]]:
    """DeepSeek-V3 key mapping: MLA attention + MoE."""
    mapping: dict[str, tuple[str, Callable | None]] = {}
    num_attention_heads = config.num_attention_heads
    qk_head_dim = config.qk_head_dim
    v_head_dim = config.v_head_dim
    qk_nope_head_dim = config.qk_nope_head_dim

    mapping["model.embed_tokens.weight"] = ("token_embedder", None)

    for i in range(config.num_hidden_layers):
        hf = f"model.layers.{i}"
        mt = f"layers.{i}"

        # MLA attention
        if config.q_lora_rank > 0:
            mapping[f"{hf}.self_attn.q_a_proj.weight"] = (
                f"{mt}.attention.q_a_proj.kernel", _transpose
            )
            mapping[f"{hf}.self_attn.q_a_layernorm.weight"] = (
                f"{mt}.attention.q_a_norm.scale", None
            )
            mapping[f"{hf}.self_attn.q_b_proj.weight"] = (
                f"{mt}.attention.q_b_proj.kernel",
                _make_qkv_hf_to_mt(num_attention_heads, qk_head_dim),
            )
        else:
            mapping[f"{hf}.self_attn.q_proj.weight"] = (
                f"{mt}.attention.q_proj.kernel",
                _make_qkv_hf_to_mt(num_attention_heads, qk_head_dim),
            )

        mapping[f"{hf}.self_attn.kv_a_proj_with_mqa.weight"] = (
            f"{mt}.attention.kv_a_proj.kernel", _transpose
        )
        mapping[f"{hf}.self_attn.kv_a_layernorm.weight"] = (
            f"{mt}.attention.kv_a_norm.scale", None
        )
        mapping[f"{hf}.self_attn.kv_b_proj.weight"] = (
            f"{mt}.attention.kv_b_proj.kernel",
            _make_qkv_hf_to_mt(num_attention_heads, qk_nope_head_dim + v_head_dim),
        )
        mapping[f"{hf}.self_attn.o_proj.weight"] = (
            f"{mt}.attention.out.kernel",
            _make_out_hf_to_mt(num_attention_heads, v_head_dim),
        )

        # MLP: dense layers (first_k_dense_replace) use standard MLP
        if i < config.first_k_dense_replace:
            mapping[f"{hf}.mlp.gate_proj.weight"] = (f"{mt}.mlp.gate.kernel", _transpose)
            mapping[f"{hf}.mlp.up_proj.weight"] = (f"{mt}.mlp.up.kernel", _transpose)
            mapping[f"{hf}.mlp.down_proj.weight"] = (f"{mt}.mlp.down.kernel", _transpose)
        else:
            # MoE layers: router gate + shared expert (expert weights handled separately)
            mapping[f"{hf}.mlp.gate.weight"] = (
                f"{mt}.mlp.router.gate.kernel", _transpose
            )
            mapping[f"{hf}.mlp.gate.e_score_correction_bias"] = (
                f"{mt}.mlp.router.e_score_correction_bias", None
            )
            # Per-expert weights handled by _import_expert_weights / _export_expert_weights
            if config.n_shared_experts > 0:
                mapping[f"{hf}.mlp.shared_experts.gate_proj.weight"] = (
                    f"{mt}.mlp.shared_expert.gate.kernel", _transpose
                )
                mapping[f"{hf}.mlp.shared_experts.up_proj.weight"] = (
                    f"{mt}.mlp.shared_expert.up.kernel", _transpose
                )
                mapping[f"{hf}.mlp.shared_experts.down_proj.weight"] = (
                    f"{mt}.mlp.shared_expert.down.kernel", _transpose
                )

        mapping[f"{hf}.input_layernorm.weight"] = (f"{mt}.pre_attn_norm.scale", None)
        mapping[f"{hf}.post_attention_layernorm.weight"] = (f"{mt}.post_attn_norm.scale", None)

    mapping["model.norm.weight"] = ("final_norm.scale", None)
    mapping["lm_head.weight"] = ("output_proj.kernel", _transpose)

    return mapping


def _qwen3_next_hf_to_mt_map(config: MinTextConfig) -> dict[str, tuple[str, Callable | None]]:
    """Qwen3-Next key mapping: hybrid full + linear attention + MoE."""
    mapping: dict[str, tuple[str, Callable | None]] = {}
    num_attention_heads = config.num_attention_heads
    num_key_value_heads = config.num_key_value_heads or config.num_attention_heads
    head_dim = config.head_dim or (config.hidden_size // config.num_attention_heads)
    layer_types = config.layer_types

    mapping["model.embed_tokens.weight"] = ("token_embedder", None)

    for i in range(config.num_hidden_layers):
        hf = f"model.layers.{i}"
        mt = f"layers.{i}"
        layer_type = layer_types[i] if i < len(layer_types) else "full_attention"

        if layer_type == "full_attention":
            # Gated attention: Q proj is doubled (Q + gate fused)
            mapping[f"{hf}.self_attn.q_proj.weight"] = (
                f"{mt}.attention.query.kernel", _make_qkv_hf_to_mt(num_attention_heads, head_dim * 2)
            )
            mapping[f"{hf}.self_attn.k_proj.weight"] = (
                f"{mt}.attention.key.kernel", _make_qkv_hf_to_mt(num_key_value_heads, head_dim)
            )
            mapping[f"{hf}.self_attn.v_proj.weight"] = (
                f"{mt}.attention.value.kernel", _make_qkv_hf_to_mt(num_key_value_heads, head_dim)
            )
            mapping[f"{hf}.self_attn.o_proj.weight"] = (
                f"{mt}.attention.out.kernel", _make_out_hf_to_mt(num_attention_heads, head_dim)
            )
            if config.use_qk_norm:
                # RMSNorm with (1 + weight) offset
                mapping[f"{hf}.self_attn.q_norm.weight"] = (
                    f"{mt}.attention.q_norm.scale", _rmsnorm_hf_to_mt
                )
                mapping[f"{hf}.self_attn.k_norm.weight"] = (
                    f"{mt}.attention.k_norm.scale", _rmsnorm_hf_to_mt
                )
        else:
            # Linear attention
            mapping[f"{hf}.linear_attn.in_proj_qkvz.weight"] = (
                f"{mt}.attention.in_proj_qkvz.kernel", _transpose
            )
            mapping[f"{hf}.linear_attn.in_proj_ba.weight"] = (
                f"{mt}.attention.in_proj_ba.kernel", _transpose
            )
            mapping[f"{hf}.linear_attn.conv1d.weight"] = (
                f"{mt}.attention.conv_weight", None
            )
            mapping[f"{hf}.linear_attn.A_log"] = (f"{mt}.attention.A_log", None)
            mapping[f"{hf}.linear_attn.dt_bias"] = (f"{mt}.attention.dt_bias", None)
            # Gated norm uses standard weight (torch.ones init), no +1 transform
            mapping[f"{hf}.linear_attn.norm.weight"] = (f"{mt}.attention.norm.scale", None)
            mapping[f"{hf}.linear_attn.out_proj.weight"] = (
                f"{mt}.attention.out_proj.kernel", _transpose
            )

        # MLP: dense or MoE
        if config.num_experts > 0 and i >= config.first_k_dense_replace:
            mapping[f"{hf}.mlp.gate.weight"] = (f"{mt}.mlp.router.gate.kernel", _transpose)
            mapping[f"{hf}.mlp.gate.e_score_correction_bias"] = (
                f"{mt}.mlp.router.e_score_correction_bias", None
            )
            # Per-expert weights handled by _import_expert_weights / _export_expert_weights
            if config.n_shared_experts > 0:
                mapping[f"{hf}.mlp.shared_experts.gate_proj.weight"] = (
                    f"{mt}.mlp.shared_expert.gate.kernel", _transpose
                )
                mapping[f"{hf}.mlp.shared_experts.up_proj.weight"] = (
                    f"{mt}.mlp.shared_expert.up.kernel", _transpose
                )
                mapping[f"{hf}.mlp.shared_experts.down_proj.weight"] = (
                    f"{mt}.mlp.shared_expert.down.kernel", _transpose
                )
        else:
            mapping[f"{hf}.mlp.gate_proj.weight"] = (f"{mt}.mlp.gate.kernel", _transpose)
            mapping[f"{hf}.mlp.up_proj.weight"] = (f"{mt}.mlp.up.kernel", _transpose)
            mapping[f"{hf}.mlp.down_proj.weight"] = (f"{mt}.mlp.down.kernel", _transpose)

        # RMSNorm with (1 + weight) offset for layer norms
        mapping[f"{hf}.input_layernorm.weight"] = (
            f"{mt}.pre_attn_norm.scale", _rmsnorm_hf_to_mt
        )
        mapping[f"{hf}.post_attention_layernorm.weight"] = (
            f"{mt}.post_attn_norm.scale", _rmsnorm_hf_to_mt
        )

    mapping["model.norm.weight"] = ("final_norm.scale", _rmsnorm_hf_to_mt)
    mapping["lm_head.weight"] = ("output_proj.kernel", _transpose)

    return mapping


def _gemma3_hf_to_mt_map(config: MinTextConfig) -> dict[str, tuple[str, Callable | None]]:
    """Gemma3 key mapping: 4-norm, Q/K norm, RMSNorm offset (1+weight)."""
    mapping: dict[str, tuple[str, Callable | None]] = {}
    num_attention_heads = config.num_attention_heads
    num_key_value_heads = config.num_key_value_heads or config.num_attention_heads
    head_dim = config.head_dim or (config.hidden_size // config.num_attention_heads)

    mapping["model.embed_tokens.weight"] = ("token_embedder", None)

    for i in range(config.num_hidden_layers):
        hf = f"model.layers.{i}"
        mt = f"layers.{i}"

        mapping[f"{hf}.self_attn.q_proj.weight"] = (
            f"{mt}.attention.query.kernel", _make_qkv_hf_to_mt(num_attention_heads, head_dim)
        )
        mapping[f"{hf}.self_attn.k_proj.weight"] = (
            f"{mt}.attention.key.kernel", _make_qkv_hf_to_mt(num_key_value_heads, head_dim)
        )
        mapping[f"{hf}.self_attn.v_proj.weight"] = (
            f"{mt}.attention.value.kernel", _make_qkv_hf_to_mt(num_key_value_heads, head_dim)
        )
        mapping[f"{hf}.self_attn.o_proj.weight"] = (
            f"{mt}.attention.out.kernel", _make_out_hf_to_mt(num_attention_heads, head_dim)
        )

        # Q/K norm
        if config.use_qk_norm:
            mapping[f"{hf}.self_attn.q_norm.weight"] = (f"{mt}.attention.q_norm.scale", None)
            mapping[f"{hf}.self_attn.k_norm.weight"] = (f"{mt}.attention.k_norm.scale", None)

        # MLP
        mapping[f"{hf}.mlp.gate_proj.weight"] = (f"{mt}.mlp.gate.kernel", _transpose)
        mapping[f"{hf}.mlp.up_proj.weight"] = (f"{mt}.mlp.up.kernel", _transpose)
        mapping[f"{hf}.mlp.down_proj.weight"] = (f"{mt}.mlp.down.kernel", _transpose)

        # 4 norms with RMSNorm offset (1 + weight)
        mapping[f"{hf}.input_layernorm.weight"] = (
            f"{mt}.pre_attn_norm.scale", _rmsnorm_hf_to_mt
        )
        mapping[f"{hf}.post_attention_layernorm.weight"] = (
            f"{mt}.post_attn_norm.scale", _rmsnorm_hf_to_mt
        )
        mapping[f"{hf}.pre_feedforward_layernorm.weight"] = (
            f"{mt}.pre_ffw_norm.scale", _rmsnorm_hf_to_mt
        )
        mapping[f"{hf}.post_feedforward_layernorm.weight"] = (
            f"{mt}.post_ffw_norm.scale", _rmsnorm_hf_to_mt
        )

    mapping["model.norm.weight"] = ("final_norm.scale", _rmsnorm_hf_to_mt)
    # tie_word_embeddings is typical, so no lm_head
    mapping["lm_head.weight"] = ("output_proj.kernel", _transpose)

    return mapping


# --- Dispatch ---


def _hf_to_mintext_key_map(config: MinTextConfig) -> dict[str, tuple[str, Callable | None]]:
    """Build mapping from HF param keys to (MinText param key, transform_fn)."""
    model_type = config.model_type
    if model_type == "qwen3":
        return _qwen3_hf_to_mt_map(config)
    elif model_type == "deepseek_v3":
        return _deepseek_v3_hf_to_mt_map(config)
    elif model_type == "qwen3_next":
        return _qwen3_next_hf_to_mt_map(config)
    elif model_type == "gemma3":
        return _gemma3_hf_to_mt_map(config)
    else:
        return _llama_hf_to_mt_map(config)


def _mintext_to_hf_key_map(config: MinTextConfig) -> dict[str, tuple[str, Callable | None]]:
    """Reverse mapping: MinText key -> (HF key, transform)."""
    forward = _hf_to_mintext_key_map(config)
    reverse: dict[str, tuple[str, Callable | None]] = {}

    # Build inverse transform registry
    inverse_registry = _build_inverse_transforms(config)

    for hf_key, (mt_key, fwd_transform) in forward.items():
        if mt_key in inverse_registry:
            rev_transform = inverse_registry[mt_key]
        elif fwd_transform is _transpose:
            rev_transform = _transpose
        elif fwd_transform is _rmsnorm_hf_to_mt:
            rev_transform = _rmsnorm_mt_to_hf
        else:
            rev_transform = None
        reverse[mt_key] = (hf_key, rev_transform)

    return reverse


def _build_inverse_transforms(config: MinTextConfig) -> dict[str, Callable | None]:
    """Build inverse transform lookup for MinText -> HF direction."""
    inverse: dict[str, Callable | None] = {}
    num_attention_heads = config.num_attention_heads
    num_key_value_heads = config.num_key_value_heads or config.num_attention_heads
    head_dim = config.head_dim or (config.hidden_size // config.num_attention_heads)
    model_type = config.model_type

    for i in range(config.num_hidden_layers):
        mt = f"layers.{i}"
        layer_type = config.layer_types[i] if i < len(config.layer_types) else "full_attention"

        if model_type == "gemma3":
            inverse[f"{mt}.attention.query.kernel"] = _make_qkv_mt_to_hf(num_attention_heads, head_dim)
            inverse[f"{mt}.attention.key.kernel"] = _make_qkv_mt_to_hf(num_key_value_heads, head_dim)
            inverse[f"{mt}.attention.value.kernel"] = _make_qkv_mt_to_hf(num_key_value_heads, head_dim)
            inverse[f"{mt}.attention.out.kernel"] = _make_out_mt_to_hf(num_attention_heads, head_dim)
        elif model_type == "deepseek_v3":
            qk_head_dim = config.qk_head_dim
            v_head_dim = config.v_head_dim
            qk_nope_head_dim = config.qk_nope_head_dim

            if config.q_lora_rank > 0:
                inverse[f"{mt}.attention.q_b_proj.kernel"] = (
                    _make_qkv_mt_to_hf(num_attention_heads, qk_head_dim)
                )
            else:
                inverse[f"{mt}.attention.q_proj.kernel"] = (
                    _make_qkv_mt_to_hf(num_attention_heads, qk_head_dim)
                )
            inverse[f"{mt}.attention.kv_b_proj.kernel"] = (
                _make_qkv_mt_to_hf(num_attention_heads, qk_nope_head_dim + v_head_dim)
            )
            inverse[f"{mt}.attention.out.kernel"] = _make_out_mt_to_hf(num_attention_heads, v_head_dim)

            # Expert weights handled by _export_expert_weights (not in key map)

        elif model_type == "qwen3_next" and layer_type == "linear_attention":
            # Linear attention layers: transforms are all _transpose (handled above)
            pass
        elif model_type == "qwen3_next" and layer_type == "full_attention":
            # Gated attention: Q proj is doubled (head_dim * 2)
            inverse[f"{mt}.attention.query.kernel"] = _make_qkv_mt_to_hf(num_attention_heads, head_dim * 2)
            inverse[f"{mt}.attention.key.kernel"] = _make_qkv_mt_to_hf(num_key_value_heads, head_dim)
            inverse[f"{mt}.attention.value.kernel"] = _make_qkv_mt_to_hf(num_key_value_heads, head_dim)
            inverse[f"{mt}.attention.out.kernel"] = _make_out_mt_to_hf(num_attention_heads, head_dim)
        else:
            # Standard / Qwen3 attention
            inverse[f"{mt}.attention.query.kernel"] = _make_qkv_mt_to_hf(num_attention_heads, head_dim)
            inverse[f"{mt}.attention.key.kernel"] = _make_qkv_mt_to_hf(num_key_value_heads, head_dim)
            inverse[f"{mt}.attention.value.kernel"] = _make_qkv_mt_to_hf(num_key_value_heads, head_dim)
            inverse[f"{mt}.attention.out.kernel"] = _make_out_mt_to_hf(num_attention_heads, head_dim)

            # Expert weights handled by _export_expert_weights (not in key map)

    return inverse


# --- Flatten/unflatten NNX state ---


def _flatten_state(state: nnx.State) -> dict[str, np.ndarray]:
    """Flatten an NNX State into a dot-separated key -> numpy array dict."""
    flat = {}
    for path_tuple, val in nnx.to_flat_state(state):
        key = ".".join(str(k) for k in path_tuple)
        arr = val[...] if hasattr(val, '__getitem__') and not isinstance(val, np.ndarray) else val
        if hasattr(arr, 'shape'):
            flat[key] = np.asarray(arr)
    return flat


def _unflatten_state(
    flat_params: dict[str, np.ndarray],
    template_state: nnx.State,
) -> nnx.State:
    """Reconstruct an NNX State from flat params using template for structure."""
    new_flat: list[tuple[tuple, Any]] = []
    for path_tuple, val in nnx.to_flat_state(template_state):
        key = ".".join(str(k) for k in path_tuple)
        if key in flat_params:
            new_val = jnp.array(flat_params[key])
            if hasattr(val, 'replace'):
                new_flat.append((path_tuple, val.replace(value=new_val)))
            else:
                new_flat.append((path_tuple, new_val))
        else:
            new_flat.append((path_tuple, val))

    return nnx.from_flat_state(new_flat)


# --- Import from HuggingFace ---


def load_hf_checkpoint(
    hf_path: str,
    config: MinTextConfig,
    model: Transformer,
) -> Transformer:
    """Load HuggingFace SafeTensors checkpoint into a MinText Transformer.

    Args:
        hf_path: Path to HF model directory (containing *.safetensors files).
        config: MinText config matching the HF model architecture.
        model: Initialized MinText model to load weights into.

    Returns:
        Model with loaded weights.
    """
    hf_dir = Path(hf_path)
    if not hf_dir.exists():
        raise FileNotFoundError(f"HF checkpoint not found: {hf_dir}")

    # Read HF config.json and extract rope_scaling if present
    hf_config_path = hf_dir / "config.json"
    if hf_config_path.exists():
        with open(hf_config_path) as f:
            hf_cfg = json.load(f)
        rs = hf_cfg.get("rope_scaling")
        if rs:
            logger.info("Detected HF rope_scaling config: %s", rs)
            # These are informational — actual config should already be set
            # but we log them for debugging
            logger.info(
                "rope_type=%s, factor=%s, orig_max_pos=%s",
                rs.get("rope_type", "default"),
                rs.get("factor", 1.0),
                rs.get("original_max_position_embeddings", 0),
            )

    hf_params = _load_safetensors_dir(hf_dir)
    logger.info("Loaded %d HF parameters from %s", len(hf_params), hf_dir)

    key_map = _hf_to_mintext_key_map(config)

    _, state = nnx.split(model)
    flat_params = _flatten_state(state)

    loaded_keys = set()
    missing_keys = []

    for hf_key, (mt_key, transform) in key_map.items():
        if hf_key not in hf_params:
            missing_keys.append(hf_key)
            continue

        if mt_key not in flat_params:
            logger.warning("MinText key %s not found in model (skipping HF key %s)", mt_key, hf_key)
            continue

        weight = hf_params[hf_key]
        if transform is not None:
            weight = transform(weight)

        target_shape = flat_params[mt_key].shape
        if weight.shape != target_shape:
            raise ValueError(
                f"Shape mismatch for {mt_key}: HF={weight.shape}, MinText={target_shape}"
            )

        flat_params[mt_key] = weight
        loaded_keys.add(hf_key)

    # Import per-expert weights (MoE models)
    expert_keys = _import_expert_weights(hf_params, config, flat_params)
    loaded_keys.update(expert_keys)

    if missing_keys:
        logger.warning("Missing HF keys (not loaded): %s", missing_keys)

    if config.tie_word_embeddings and "lm_head.weight" not in loaded_keys:
        if "model.embed_tokens.weight" in loaded_keys:
            logger.info("Weight tying: output shares embedding weights")

    extra_hf = set(hf_params.keys()) - loaded_keys
    if extra_hf:
        logger.info("Extra HF keys not mapped (%d): %s", len(extra_hf), list(extra_hf)[:5])

    state = _unflatten_state(flat_params, state)
    graph, _ = nnx.split(model)
    model = nnx.merge(graph, state)

    logger.info("Loaded %d/%d HF parameters into model", len(loaded_keys), len(key_map))
    return model


def _load_safetensors_dir(path: Path) -> dict[str, np.ndarray]:
    """Load all .safetensors files in a directory into a single dict."""
    params = {}
    st_files = sorted(path.glob("*.safetensors"))
    if not st_files:
        raise FileNotFoundError(f"No .safetensors files in {path}")
    for f in st_files:
        shard = load_file(str(f))
        params.update(shard)
    return params


# --- Export to HuggingFace ---


def save_hf_checkpoint(
    model: Transformer,
    config: MinTextConfig,
    output_path: str,
    hf_config: dict[str, Any] | None = None,
) -> None:
    """Export MinText model to HuggingFace SafeTensors format.

    Args:
        model: Trained MinText model.
        config: MinText config.
        output_path: Output directory for SafeTensors files.
        hf_config: Optional HF config dict to write as config.json.
    """
    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    _, state = nnx.split(model)
    flat_params = _flatten_state(state)

    key_map = _mintext_to_hf_key_map(config)

    hf_params: dict[str, np.ndarray] = {}
    for mt_key, (hf_key, transform) in key_map.items():
        if mt_key not in flat_params:
            logger.warning("MinText key %s not found, skipping", mt_key)
            continue

        weight = flat_params[mt_key]
        if transform is not None:
            weight = transform(weight)
        hf_params[hf_key] = weight

    # Export per-expert weights (MoE models)
    hf_params.update(_export_expert_weights(flat_params, config))

    st_path = out_dir / "model.safetensors"
    save_file(hf_params, str(st_path))
    logger.info("Saved %d parameters to %s", len(hf_params), st_path)

    cfg = hf_config if hf_config is not None else _mintext_to_hf_config(config)
    config_path = out_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)
    logger.info("Wrote %s", config_path)


def _build_rope_scaling_config(config: MinTextConfig) -> dict[str, Any] | None:
    """Build HF rope_scaling dict from MinText config, if non-default."""
    if config.rope_type == "default":
        return None
    rs: dict[str, Any] = {
        "rope_type": config.rope_type,
        "factor": config.rope_scaling_factor,
    }
    if config.rope_original_max_position_embeddings > 0:
        rs["original_max_position_embeddings"] = config.rope_original_max_position_embeddings
    if config.rope_type == "yarn":
        rs["beta_fast"] = config.rope_yarn_beta_fast
        rs["beta_slow"] = config.rope_yarn_beta_slow
        rs["mscale"] = config.rope_yarn_mscale
        rs["mscale_all_dim"] = config.rope_yarn_mscale_all_dim
    elif config.rope_type == "llama3":
        rs["low_freq_factor"] = config.rope_llama3_low_freq_factor
        rs["high_freq_factor"] = config.rope_llama3_high_freq_factor
    return rs


def _mintext_to_hf_config(config: MinTextConfig) -> dict[str, Any]:
    """Convert MinText config to a basic HF-compatible config.json."""
    model_type = config.model_type
    if model_type == "qwen3":
        hf_cfg = {
            "architectures": ["Qwen3ForCausalLM"],
            "model_type": "qwen3",
            "hidden_size": config.hidden_size,
            "intermediate_size": config.intermediate_size,
            "num_hidden_layers": config.num_hidden_layers,
            "num_attention_heads": config.num_attention_heads,
            "num_key_value_heads": config.num_key_value_heads,
            "max_position_embeddings": config.max_position_embeddings,
            "vocab_size": config.vocab_size,
            "rms_norm_eps": config.rms_norm_eps,
            "rope_theta": config.rope_theta,
            "head_dim": config.head_dim,
            "tie_word_embeddings": config.tie_word_embeddings,
            "dtype": "bfloat16" if config.dtype == "bfloat16" else "float32",
        }
    elif model_type == "deepseek_v3":
        hf_cfg = {
            "architectures": ["DeepseekV3ForCausalLM"],
            "model_type": "deepseek_v3",
            "hidden_size": config.hidden_size,
            "intermediate_size": config.intermediate_size,
            "num_hidden_layers": config.num_hidden_layers,
            "num_attention_heads": config.num_attention_heads,
            "max_position_embeddings": config.max_position_embeddings,
            "vocab_size": config.vocab_size,
            "rms_norm_eps": config.rms_norm_eps,
            "rope_theta": config.rope_theta,
            "q_lora_rank": config.q_lora_rank,
            "kv_lora_rank": config.kv_lora_rank,
            "qk_nope_head_dim": config.qk_nope_head_dim,
            "qk_rope_head_dim": config.qk_rope_head_dim,
            "v_head_dim": config.v_head_dim,
            "n_routed_experts": config.num_experts,
            "num_experts_per_tok": config.num_experts_per_tok,
            "moe_intermediate_size": config.moe_intermediate_size,
            "n_shared_experts": config.n_shared_experts,
            "first_k_dense_replace": config.first_k_dense_replace,
            "n_group": config.n_group,
            "topk_group": config.topk_group,
            "routed_scaling_factor": config.routed_scaling_factor,
            "norm_topk_prob": config.norm_topk_prob,
            "rope_interleave": config.rope_interleave,
            "tie_word_embeddings": config.tie_word_embeddings,
            "dtype": "bfloat16" if config.dtype == "bfloat16" else "float32",
        }
    elif model_type == "qwen3_next":
        hf_cfg = {
            "architectures": ["Qwen3NextForCausalLM"],
            "model_type": "qwen3_next",
            "hidden_size": config.hidden_size,
            "intermediate_size": config.intermediate_size,
            "num_hidden_layers": config.num_hidden_layers,
            "num_attention_heads": config.num_attention_heads,
            "num_key_value_heads": config.num_key_value_heads,
            "max_position_embeddings": config.max_position_embeddings,
            "vocab_size": config.vocab_size,
            "rms_norm_eps": config.rms_norm_eps,
            "rope_theta": config.rope_theta,
            "head_dim": config.head_dim,
            "partial_rotary_factor": config.partial_rotary_factor,
            "layer_types": config.layer_types,
            "linear_key_head_dim": config.linear_key_head_dim,
            "linear_value_head_dim": config.linear_value_head_dim,
            "linear_num_key_heads": config.linear_num_key_heads,
            "linear_num_value_heads": config.linear_num_value_heads,
            "linear_conv_kernel_dim": config.linear_conv_kernel_dim,
            "num_experts": config.num_experts,
            "num_experts_per_tok": config.num_experts_per_tok,
            "moe_intermediate_size": config.moe_intermediate_size,
            "shared_expert_intermediate_size": config.moe_intermediate_size * config.n_shared_experts if config.n_shared_experts > 0 else 0,
            "decoder_sparse_step": config.first_k_dense_replace,
            "tie_word_embeddings": config.tie_word_embeddings,
            "dtype": "bfloat16" if config.dtype == "bfloat16" else "float32",
        }
    elif model_type == "gemma3":
        hf_cfg = {
            "architectures": ["Gemma3ForCausalLM"],
            "model_type": "gemma3_text",
            "hidden_size": config.hidden_size,
            "intermediate_size": config.intermediate_size,
            "num_hidden_layers": config.num_hidden_layers,
            "num_attention_heads": config.num_attention_heads,
            "num_key_value_heads": config.num_key_value_heads,
            "max_position_embeddings": config.max_position_embeddings,
            "vocab_size": config.vocab_size,
            "rms_norm_eps": config.rms_norm_eps,
            "rope_theta": config.rope_theta,
            "head_dim": config.head_dim,
            "hidden_activation": config.hidden_activation,
            "tie_word_embeddings": config.tie_word_embeddings,
            "sliding_window": config.sliding_window,
            "_sliding_window_pattern": config.sliding_window_pattern,
            "dtype": "bfloat16" if config.dtype == "bfloat16" else "float32",
        }
        if config.query_pre_attn_scalar is not None:
            hf_cfg["query_pre_attn_scalar"] = config.query_pre_attn_scalar
        if config.attn_logit_softcapping is not None:
            hf_cfg["attn_logit_softcapping"] = config.attn_logit_softcapping
        if config.final_logit_softcapping is not None:
            hf_cfg["final_logit_softcapping"] = config.final_logit_softcapping
    else:
        hf_cfg = {
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "hidden_size": config.hidden_size,
            "intermediate_size": config.intermediate_size,
            "num_hidden_layers": config.num_hidden_layers,
            "num_attention_heads": config.num_attention_heads,
            "num_key_value_heads": config.num_key_value_heads,
            "max_position_embeddings": config.max_position_embeddings,
            "vocab_size": config.vocab_size,
            "rms_norm_eps": config.rms_norm_eps,
            "rope_theta": config.rope_theta,
            "tie_word_embeddings": config.tie_word_embeddings,
            "dtype": "bfloat16" if config.dtype == "bfloat16" else "float32",
        }
    rope_scaling = _build_rope_scaling_config(config)
    if rope_scaling is not None:
        hf_cfg["rope_scaling"] = rope_scaling
    return hf_cfg
