"""Token routing for MoE grouped matmul: sort by expert, unsort + combine.

The core idea: sort tokens so that all tokens for expert 0 are contiguous,
then expert 1, etc. This enables grouped matmul (ragged_dot). After expert
computation, unsort back to original order and apply weighted sum.

Uses jax.custom_vjp so that the backward of sort = unsort (via argsort),
avoiding materializing a full permutation matrix.

"""

from __future__ import annotations

import jax
import jax.numpy as jnp


# --- Custom VJP for efficient sort/unsort gradients ---


@jax.custom_vjp
def _sort_with_vjp(x: jax.Array, indices: jax.Array) -> jax.Array:
    """Gather x by indices: out[i] = x[indices[i]]."""
    return x[indices]


def _sort_fwd(
    x: jax.Array, indices: jax.Array
) -> tuple[jax.Array, jax.Array]:
    return _sort_with_vjp(x, indices), indices


def _sort_bwd(
    indices: jax.Array, grads: jax.Array
) -> tuple[jax.Array, None]:
    # Backward of gather-by-indices is scatter-by-inverse-indices,
    # which is equivalent to gather-by-argsort(indices).
    return _sort_with_vjp(grads, jnp.argsort(indices)), None


_sort_with_vjp.defvjp(_sort_fwd, _sort_bwd)


# --- Public API ---


def route(
    tokens: jax.Array,
    topk_indices: jax.Array,
    num_experts: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Sort tokens by expert assignment for grouped matmul.

    Args:
        tokens: [N, D] flattened activations.
        topk_indices: [N, K] expert indices per token.
        num_experts: Total number of experts.

    Returns:
        sorted_tokens: [N*K, D] tokens sorted by expert (contiguous groups).
        sort_indices: [N*K] permutation indices (needed by unroute).
        group_sizes: [E] number of tokens assigned to each expert.
    """
    N, K = topk_indices.shape

    # Replicate each token K times (once per selected expert)
    replicated = jnp.repeat(tokens, K, axis=0)  # [N*K, D]

    # Flatten expert assignments
    flat_indices = jnp.ravel(topk_indices)  # [N*K]

    # Sort permutation: groups tokens by expert
    sort_indices = jnp.argsort(flat_indices, stable=True)

    # Apply permutation with custom VJP for efficient backward
    sorted_tokens = _sort_with_vjp(replicated, sort_indices)

    # Count tokens per expert
    group_sizes = jnp.bincount(flat_indices, length=num_experts)

    return sorted_tokens, sort_indices, group_sizes


def unroute(
    sorted_output: jax.Array,
    sort_indices: jax.Array,
    topk_weights: jax.Array,
    num_experts_per_tok: int,
) -> jax.Array:
    """Unsort expert outputs and apply weighted combination.

    Args:
        sorted_output: [N*K, D] expert outputs in sorted order.
        sort_indices: [N*K] permutation from route().
        topk_weights: [N, K] routing weights per token per expert.
        num_experts_per_tok: K, number of experts per token.

    Returns:
        output: [N, D] weighted combination of expert outputs.
    """
    # Inverse permutation: unsort back to original token order
    unsort_indices = jnp.argsort(sort_indices)
    unsorted = _sort_with_vjp(sorted_output, unsort_indices)

    # Reshape to [N, K, D] and apply weighted sum
    N = topk_weights.shape[0]
    D = sorted_output.shape[-1]
    unsorted = unsorted.reshape(N, num_experts_per_tok, D)

    # Weighted sum over experts: [N, K, D] * [N, K, 1] -> [N, D]
    return jnp.einsum("nkd,nk->nd", unsorted, topk_weights)
