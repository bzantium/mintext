"""Mixture of Experts: Router, Experts, and MoE Block (DeepSeek-V3 style)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from mintext.config import MinTextConfig
from mintext.modules.linear import Linear, MLP, _default_kernel_init


class MoERouter(nnx.Module):
    """Top-k group router with loss-free bias update (DeepSeek-V3 style).

    Uses sigmoid scoring + group-based top-k selection.
    """

    def __init__(self, config: MinTextConfig, *, rngs: nnx.Rngs):
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor

        dtype = config.jnp_dtype
        weight_dtype = config.jnp_weight_dtype

        self.gate = Linear(
            config.hidden_size, config.num_experts,
            dtype=dtype, weight_dtype=weight_dtype,
            kernel_axes=("embed", "mlp"),
            use_bias=False, rngs=rngs,
        )
        # Non-gradient bias for loss-free load balancing
        self.e_score_correction_bias = nnx.Variable(
            jnp.zeros(config.num_experts, dtype=jnp.float32)
        )

    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Route tokens to experts.

        Args:
            x: [batch * seq_len, hidden_size]

        Returns:
            topk_indices: [batch * seq_len, num_experts_per_tok]
            topk_weights: [batch * seq_len, num_experts_per_tok]
            scores: [batch * seq_len, num_experts] (raw sigmoid scores)
        """
        logits = self.gate(x)  # [N, num_experts]
        scores = jax.nn.sigmoid(jnp.asarray(logits, jnp.float32))

        # Add correction bias for routing only (not for weights)
        scores_for_routing = scores + self.e_score_correction_bias[...]

        # Group-based top-k routing
        topk_indices, topk_weights = self._group_topk_routing(
            scores, scores_for_routing
        )

        if self.norm_topk_prob:
            topk_weights = topk_weights / (topk_weights.sum(-1, keepdims=True) + 1e-20)

        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_indices, topk_weights, scores

    def _group_topk_routing(
        self, scores: jax.Array, scores_for_routing: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        N = scores.shape[0]
        experts_per_group = self.num_experts // self.n_group

        # Reshape to groups: [N, n_group, experts_per_group]
        grouped = scores_for_routing.reshape(N, self.n_group, experts_per_group)

        # Get top-2 scores per group for group scoring
        group_scores = jax.lax.top_k(grouped, min(2, experts_per_group))[0]
        group_scores = group_scores.sum(axis=-1)  # [N, n_group]

        # Select top-k groups
        topk_group_indices = jax.lax.top_k(group_scores, self.topk_group)[1]  # [N, topk_group]

        # Create group mask: [N, n_group]
        group_mask = jnp.zeros((N, self.n_group), dtype=jnp.bool_)
        group_mask = group_mask.at[
            jnp.arange(N)[:, None], topk_group_indices
        ].set(True)

        # Expand to expert mask: [N, num_experts]
        expert_mask = jnp.repeat(group_mask, experts_per_group, axis=1)

        # Apply mask to routing scores (set non-selected groups to -inf)
        masked_scores = jnp.where(expert_mask, scores_for_routing, -jnp.inf)

        # Global top-k from selected groups
        topk_indices = jax.lax.top_k(masked_scores, self.num_experts_per_tok)[1]

        # Get actual scores (not routing scores) for the selected experts
        topk_weights = jnp.take_along_axis(scores, topk_indices, axis=1)

        return topk_indices, topk_weights


class MoEExperts(nnx.Module):
    """Batched expert computation via scatter-gather.

    Each expert is a SwiGLU MLP with gate_up fused and down separate.
    Uses ragged_dot grouped GEMM with optional custom VJP for tiling control.
    """

    def __init__(self, config: MinTextConfig, *, rngs: nnx.Rngs):
        self.num_experts = config.num_experts
        self.dtype = config.jnp_dtype
        self.use_custom_vjp = config.moe_use_custom_vjp
        self.mosaic_fusion = config.moe_mosaic_fusion_group
        self.gate_up_tiling = tuple(config.moe_gate_up_tiling) if config.moe_gate_up_tiling else None
        self.down_tiling = tuple(config.moe_down_tiling) if config.moe_down_tiling else None
        weight_dtype = config.jnp_weight_dtype

        init = _default_kernel_init()

        # gate_up fused: [num_experts, hidden_size, 2 * moe_intermediate_size]
        self.gate_up_proj = nnx.Param(
            init(
                rngs.params(),
                (config.num_experts, config.hidden_size, 2 * config.moe_intermediate_size),
                weight_dtype, in_axis=1, out_axis=2,
            ),
            sharding=("exp", "embed", "mlp"),
        )
        # down: [num_experts, moe_intermediate_size, hidden_size]
        self.down_proj = nnx.Param(
            init(
                rngs.params(),
                (config.num_experts, config.moe_intermediate_size, config.hidden_size),
                weight_dtype, in_axis=1, out_axis=2,
            ),
            sharding=("exp", "mlp", "embed"),
        )

    def __call__(
        self,
        x: jax.Array,
        topk_indices: jax.Array,
        topk_weights: jax.Array,
    ) -> jax.Array:
        """Compute weighted sum of expert outputs for each token.

        Args:
            x: [N, hidden_size]
            topk_indices: [N, K] expert indices
            topk_weights: [N, K] expert weights

        Returns:
            Output [N, hidden_size]
        """
        return self._forward_kernel(x, topk_indices, topk_weights)

    def _forward_kernel(
        self,
        x: jax.Array,
        topk_indices: jax.Array,
        topk_weights: jax.Array,
    ) -> jax.Array:
        """Forward using grouped matmul (ragged_dot)."""
        from mintext.kernels import route, unroute, grouped_matmul, grouped_matmul_vjp

        gate_up = jnp.asarray(self.gate_up_proj[...], self.dtype)
        down = jnp.asarray(self.down_proj[...], self.dtype)

        # 1. Route tokens by expert
        sorted_tokens, sort_indices, group_sizes = route(
            x, topk_indices, self.num_experts
        )

        if self.use_custom_vjp:
            # 9-param tiling: gate_up uses first 9, down uses its own 9
            gu_tiling = self.gate_up_tiling or (128, 128, 128)
            dn_tiling = self.down_tiling or (128, 128, 128)
            # Pad to 9 params if only 3 provided
            if len(gu_tiling) < 9:
                gu_tiling = (gu_tiling[:3] * 3)[:9]
            if len(dn_tiling) < 9:
                dn_tiling = (dn_tiling[:3] * 3)[:9]

            # 2. Gate+Up projection
            h = grouped_matmul_vjp(
                sorted_tokens, gate_up, group_sizes,
                tiling=gu_tiling, mosaic_fusion=self.mosaic_fusion,
            )

            # 3. SwiGLU activation
            gate_h, up_h = jnp.split(h, 2, axis=-1)
            intermediate = jax.nn.silu(gate_h) * up_h

            # 4. Down projection
            out = grouped_matmul_vjp(
                intermediate, down, group_sizes,
                tiling=dn_tiling, mosaic_fusion=self.mosaic_fusion,
            )
        else:
            # 2. Gate+Up projection: [N*K, D] @ [E, D, 2*I] -> [N*K, 2*I]
            h = grouped_matmul(
                sorted_tokens, gate_up, group_sizes,
                tiling=self.gate_up_tiling,
            )

            # 3. SwiGLU activation
            gate_h, up_h = jnp.split(h, 2, axis=-1)
            intermediate = jax.nn.silu(gate_h) * up_h

            # 4. Down projection: [N*K, I] @ [E, I, D] -> [N*K, D]
            out = grouped_matmul(
                intermediate, down, group_sizes,
                tiling=self.down_tiling,
            )

        # 5. Unroute: unsort + weighted sum -> [N, D]
        return unroute(out, sort_indices, topk_weights, topk_indices.shape[1])


class MoEBlock(nnx.Module):
    """Full MoE block: router + routed experts + optional shared expert."""

    def __init__(self, config: MinTextConfig, *, rngs: nnx.Rngs):
        self.router = MoERouter(config, rngs=rngs)
        self.experts = MoEExperts(config, rngs=rngs)
        self.n_shared_experts = config.n_shared_experts
        if config.n_shared_experts > 0:
            # Shared expert uses moe_intermediate_size * n_shared_experts, not intermediate_size
            shared_cfg = config.model_copy(update={
                "intermediate_size": config.moe_intermediate_size * config.n_shared_experts,
            })
            self.shared_expert = MLP(shared_cfg, rngs=rngs)
        else:
            self.shared_expert = None

    def __call__(self, x: jax.Array) -> tuple[jax.Array, dict]:
        """Forward pass through MoE block.

        Args:
            x: [batch, seq_len, hidden_size]

        Returns:
            (output, aux_data) where aux_data contains routing info for bias updates.
        """
        B, S, D = x.shape
        x_flat = x.reshape(B * S, D)

        topk_indices, topk_weights, scores = self.router(x_flat)
        routed_out = self.experts(x_flat, topk_indices, topk_weights)
        routed_out = routed_out.reshape(B, S, D)

        if self.shared_expert is not None:
            shared_out = self.shared_expert(x)
            output = routed_out + shared_out
        else:
            output = routed_out

        aux_data = {
            "topk_indices": topk_indices,
            "scores": scores,
        }
        return output, aux_data
