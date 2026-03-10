"""Pytree utilities: parameter counting, NaN detection, TFLOP estimation."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from mintext.config import MinTextConfig


def count_params(params) -> int:
    """Count total number of parameters in a pytree."""
    return sum(x.size for x in jax.tree.leaves(params))


def check_nan(tree, label: str = "params") -> bool:
    """Check if any leaf in a pytree contains NaN values.

    Returns True if NaN found (bad).
    """
    leaves = jax.tree.leaves(tree)
    for leaf in leaves:
        if jnp.any(jnp.isnan(leaf)):
            return True
    return False


def calculate_tflops_per_device(config: MinTextConfig) -> float:
    """Estimate training TFLOP per device per step.

    Uses the standard FLOP estimation formula (same as MaxText):
      Total = (FFN + Attention_weights + Attention_QK_scores + Embedding) * num_hidden_layers * 3
    The ×3 accounts for forward + backward (2× weight, 1× activation grads).

    Per-device TFLOP = model_flops(per_device_batch_size, seq_len) * 3.
    This assumes each device independently processes per_device_batch_size
    samples (standard data-parallel convention).

    Returns:
        TFLOP per device per step (float).
    """
    B = config.per_device_batch_size
    S = config.seq_length
    D = config.hidden_size
    V = config.vocab_size
    L = config.num_hidden_layers
    H = config.num_attention_heads
    Hkv = config.num_key_value_heads or H
    Dh = config.head_dim or (D // H)
    M = config.intermediate_size
    grad_accum = config.gradient_accumulation_steps

    # FFN: SwiGLU has gate, up, down = 3 matmuls of (B*S, D) x (D, M) each
    # gate and up are fused as 2*M, so: 2*B*S*D*(2*M) + 2*B*S*M*D = 2*B*S*D*M*3
    ffn_flops = 2 * B * S * D * M * 3  # SwiGLU: gate+up+down

    # MoE: scale FFN flops
    if config.num_experts > 0:
        # Gate routing: 2 * B * S * D * num_experts
        gate_flops = 2 * B * S * D * config.num_experts
        # Each token uses num_experts_per_tok experts
        num_moe_layers = max(0, L - config.first_k_dense_replace)
        num_dense_layers = L - num_moe_layers
        moe_intermediate = config.moe_intermediate_size
        moe_ffn_flops = 2 * B * S * D * moe_intermediate * 3  # per expert
        total_ffn_flops = (
            ffn_flops * num_dense_layers
            + (gate_flops + moe_ffn_flops * config.num_experts_per_tok) * num_moe_layers
        )
    else:
        total_ffn_flops = ffn_flops * L

    # Attention weights: Q, K, V projections + O projection
    # QKV: 2 * B * S * D * (H + 2*Hkv) * Dh
    # O: 2 * B * S * D * H * Dh
    if config.attention_type == "mla":
        # MLA: q_a(D→q_lora_rank) + q_b(q_lora_rank→H*qk_head_dim) + kv_a(D→kv_lora_rank+rope) + kv_b(kv_lora_rank→H*(nope+v)) + o(H*v→D)
        q_lora_rank = config.q_lora_rank
        kv_lora_rank = config.kv_lora_rank
        qk_head_dim = config.qk_head_dim
        v_head_dim = config.v_head_dim
        qk_rope_dim = config.qk_rope_head_dim
        qk_nope_dim = config.qk_nope_head_dim
        attn_weight_flops = 2 * B * S * (
            D * q_lora_rank + q_lora_rank * H * qk_head_dim  # Q path
            + D * (kv_lora_rank + qk_rope_dim)  # KV compress
            + kv_lora_rank * H * (qk_nope_dim + v_head_dim)  # KV expand
            + H * v_head_dim * D  # O proj
        )
    else:
        attn_weight_flops = 2 * B * S * D * ((H + 2 * Hkv) * Dh + H * Dh)

    # Attention scores: Q*K^T + softmax*V
    # Non-causal: 4 * B * S^2 * H * Dh, divide by 2 for causal mask
    attn_score_flops = 4 * B * S * S * H * Dh // 2

    # Embedding: 2 * B * S * D * V (forward + backward)
    embedding_flops = 2 * B * S * D * V

    # Total with ×3 for fwd+bwd, scale by gradient accumulation
    total_flops = (total_ffn_flops + (attn_weight_flops + attn_score_flops) * L + embedding_flops) * 3 * grad_accum

    # Convert to TFLOP per device
    return total_flops / 1e12


def calculate_tokens_per_device(config: MinTextConfig) -> int:
    """Calculate tokens processed per device per step."""
    return (
        config.per_device_batch_size
        * config.seq_length
        * config.gradient_accumulation_steps
    )
