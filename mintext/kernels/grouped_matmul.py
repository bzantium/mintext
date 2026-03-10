"""Grouped matrix multiplication: x[group_i] @ weights[i] per expert.

Two modes:
- ragged_dot: XLA-native grouped GEMM via jax.lax.ragged_dot
- ragged_dot with custom VJP for backward tiling control (via grouped_matmul_vjp)
"""

from __future__ import annotations

import contextlib
import functools
import random

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# XLA metadata context manager for tiling/fusion hints
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _xla_metadata(**kwargs: str):
    """Apply XLA metadata hints (ragged_dot_tiling, mosaic_fusion_group, etc.)."""
    try:
        from jax._src.lib.xla_extension import set_xla_metadata
    except ImportError:
        set_xla_metadata = None

    if set_xla_metadata is not None:
        with set_xla_metadata(**kwargs):
            yield
    else:
        yield


def _make_tiling_str(tiling: tuple[int, ...] | list[int]) -> str:
    return ",".join(str(t) for t in tiling)


def _fusion_group_id(mosaic_fusion: bool) -> str:
    """Generate mosaic fusion group ID.

    TPU needs random IDs for Mosaic fusion; GPU needs deterministic "0".
    """
    if not mosaic_fusion:
        return "0"
    platform = jax.devices()[0].platform
    if platform == "tpu":
        return str(random.randint(0, 1_000_000_000))
    return "0"


# ---------------------------------------------------------------------------
# Original grouped_matmul (no custom VJP, backward compat)
# ---------------------------------------------------------------------------

def grouped_matmul(
    x: jax.Array,
    weights: jax.Array,
    group_sizes: jax.Array,
    *,
    tiling: tuple[int, ...] | list[int] | None = None,
) -> jax.Array:
    """Grouped matrix multiplication: x[group_i] @ weights[i] for each group.

    Args:
        x: [M, K_in] sorted tokens (contiguous per expert).
        weights: [E, K_in, K_out] per-expert weight matrices.
        group_sizes: [E] int32, tokens per expert (sum = M).
        tiling: Optional (tm, tk, tn) for ragged_dot XLA hints.

    Returns:
        output: [M, K_out]
    """
    return _gmm_ragged_dot(x, weights, group_sizes, tiling)


# ---------------------------------------------------------------------------
# Custom VJP grouped matmul with 9-param tiling
# ---------------------------------------------------------------------------

def grouped_matmul_vjp(
    x: jax.Array,
    weights: jax.Array,
    group_sizes: jax.Array,
    *,
    tiling: tuple[int, ...] | list[int] = (128, 128, 128, 128, 128, 128, 128, 128, 128),
    mosaic_fusion: bool = True,
) -> jax.Array:
    """Grouped matmul with custom VJP for explicit backward tiling control.

    Uses 9-param tiling: fwd(3) + dlhs(3) + drhs(3).
    Forward: ragged_dot(x, W, gs) with tiling[:3]
    Backward dlhs: ragged_dot(grad, W^T, gs) with tiling[3:6]
    Backward drhs: tgmm(x, grad, gs) with tiling[6:9]

    Args:
        x: [M, K_in] sorted tokens.
        weights: [E, K_in, K_out] per-expert weight matrices.
        group_sizes: [E] int32.
        tiling: 9-tuple of ints (tm_f, tk_f, tn_f, tm_b, tk_b, tn_b, tm_w, tk_w, tn_w).
        mosaic_fusion: Enable mosaic fusion group hints.

    Returns:
        output: [M, K_out]
    """
    tiling_tuple = tuple(tiling)
    if len(tiling_tuple) < 9:
        # Pad with first 3 repeated
        base = tiling_tuple[:3] if len(tiling_tuple) >= 3 else (128, 128, 128)
        tiling_tuple = (base * 3)[:9]

    return _gmm_vjp_impl(x, weights, group_sizes, tiling_tuple, mosaic_fusion)


@functools.partial(jax.custom_vjp, nondiff_argnums=(3, 4))
def _gmm_vjp_impl(
    x: jax.Array,
    weights: jax.Array,
    group_sizes: jax.Array,
    tiling: tuple[int, ...],
    mosaic_fusion: bool,
) -> jax.Array:
    return _gmm_vjp_fwd(x, weights, group_sizes, tiling, mosaic_fusion)[0]


def _gmm_vjp_fwd(
    x: jax.Array,
    weights: jax.Array,
    group_sizes: jax.Array,
    tiling: tuple[int, ...],
    mosaic_fusion: bool,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array]]:
    """Forward pass: ragged_dot with forward tiling."""
    fwd_tiling = tiling[:3]
    metadata: dict[str, str] = {}
    metadata["ragged_dot_tiling"] = _make_tiling_str(fwd_tiling)
    if mosaic_fusion:
        metadata["mosaic_fusion_group"] = _fusion_group_id(mosaic_fusion)

    with _xla_metadata(**metadata):
        out = jax.lax.ragged_dot(
            lhs=x,
            rhs=weights,
            group_sizes=group_sizes,
        )
    return out, (x, weights, group_sizes)


def _gmm_vjp_bwd(
    tiling: tuple[int, ...],
    mosaic_fusion: bool,
    residuals: tuple[jax.Array, jax.Array, jax.Array],
    grad: jax.Array,
) -> tuple[jax.Array, jax.Array, None]:
    """Backward pass with separate tiling for dlhs and drhs."""
    x, weights, group_sizes = residuals
    dlhs_tiling = tiling[3:6]
    drhs_tiling = tiling[6:9]

    # dlhs: grad @ W^T  =>  ragged_dot(grad, weights_transposed, group_sizes)
    # weights: [E, K_in, K_out] -> transpose last 2: [E, K_out, K_in]
    weights_t = jnp.swapaxes(weights, 1, 2)

    dlhs_metadata: dict[str, str] = {}
    dlhs_metadata["ragged_dot_tiling"] = _make_tiling_str(dlhs_tiling)
    if mosaic_fusion:
        dlhs_metadata["mosaic_fusion_group"] = _fusion_group_id(mosaic_fusion)

    with _xla_metadata(**dlhs_metadata):
        dlhs = jax.lax.ragged_dot(
            lhs=grad,
            rhs=weights_t,
            group_sizes=group_sizes,
        )

    # drhs: x^T @ grad per group => tgmm
    drhs_metadata: dict[str, str] = {}
    drhs_metadata["ragged_dot_tiling"] = _make_tiling_str(drhs_tiling)
    if mosaic_fusion:
        drhs_metadata["mosaic_fusion_group"] = _fusion_group_id(mosaic_fusion)

    with _xla_metadata(**drhs_metadata):
        drhs = tgmm(x, grad, group_sizes)

    return dlhs, drhs, None


_gmm_vjp_impl.defvjp(_gmm_vjp_fwd, _gmm_vjp_bwd)


# ---------------------------------------------------------------------------
# tgmm: transposed grouped matmul (for weight gradients)
# ---------------------------------------------------------------------------

def tgmm(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
) -> jax.Array:
    """Transposed grouped matmul: lhs[:, group_i]^T @ rhs[group_i, :] per group.

    Computes weight gradients: for each expert i,
        drhs[i] = x[group_i]^T @ grad[group_i]

    Args:
        lhs: [M, K] (input activations from forward, i.e. x)
        rhs: [M, N] (grad from backward)
        group_sizes: [E] int32, tokens per expert.

    Returns:
        output: [E, K, N] per-expert weight gradients.
    """
    E = group_sizes.shape[0]
    K = lhs.shape[1]
    N = rhs.shape[1]
    M = lhs.shape[0]

    # Build expert assignment from group_sizes
    expert_ids = jnp.repeat(
        jnp.arange(E, dtype=jnp.int32), group_sizes, total_repeat_length=M
    )

    # One-hot mask: [M, E]
    mask = jax.nn.one_hot(expert_ids, E, dtype=lhs.dtype)  # [M, E]

    # Masked inputs per expert: lhs_per_expert[e] = mask[:, e] * lhs
    # [E, M, K] = [E, M, 1] * [1, M, K]
    lhs_masked = mask.T[:, :, None] * lhs[None, :, :]  # [E, M, K]
    rhs_expanded = rhs[None, :, :]  # [1, M, N]

    # Batched matmul: [E, M, K]^T @ [E, M, N] -> [E, K, N]
    # = [E, K, M] @ [E, M, N] -> [E, K, N]
    return jnp.einsum("emk,emn->ekn", lhs_masked, rhs_expanded)


# ---------------------------------------------------------------------------
# Backend implementations
# ---------------------------------------------------------------------------

def _gmm_ragged_dot(
    x: jax.Array,
    weights: jax.Array,
    group_sizes: jax.Array,
    tiling: tuple[int, ...] | list[int] | None,
) -> jax.Array:
    """Primary backend using jax.lax.ragged_dot (XLA native grouped GEMM)."""
    M = x.shape[0]
    original_M = M

    # Pad M to multiple of tile size if tiling is specified
    pad_amount = 0
    if tiling and len(tiling) >= 1:
        tm = tiling[0]
        pad_amount = (-M % tm) % tm
        if pad_amount > 0:
            x = jnp.pad(x, ((0, pad_amount), (0, 0)))

    # Apply tiling hint via XLA metadata if available
    if tiling:
        with _xla_metadata(ragged_dot_tiling=_make_tiling_str(tiling)):
            out = jax.lax.ragged_dot(
                lhs=x,
                rhs=weights,
                group_sizes=group_sizes,
            )
    else:
        out = jax.lax.ragged_dot(
            lhs=x, rhs=weights, group_sizes=group_sizes,
        )

    # Remove padding
    if pad_amount > 0:
        out = out[:original_M]

    return out


