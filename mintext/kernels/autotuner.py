"""Auto-tune MoE grouped matmul tiling parameters with disk caching."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import jax
import jax.numpy as jnp

logger = logging.getLogger(__name__)


@dataclass
class MoETuningConfig:
    """Problem dimensions for MoE autotuning."""
    num_experts: int
    hidden_size: int
    moe_intermediate_size: int
    num_experts_per_tok: int
    batch_seq_tokens: int  # B * S
    dtype: str = "bfloat16"


@dataclass
class MoETuningResult:
    """Best tiling parameters found by autotuner.

    Tiling can be 3-tuple (fwd only) or 9-tuple (fwd + dlhs + drhs).
    """
    gate_up_tiling: tuple[int, ...] | None
    down_tiling: tuple[int, ...] | None
    backend: str
    throughput_gflops: float


def _cache_key(config: MoETuningConfig) -> str:
    """SHA256 hash of config + platform + device for cache lookup."""
    platform = jax.devices()[0].platform
    device_kind = jax.devices()[0].device_kind
    key_data = {
        "num_experts": config.num_experts,
        "hidden_size": config.hidden_size,
        "moe_intermediate_size": config.moe_intermediate_size,
        "num_experts_per_tok": config.num_experts_per_tok,
        "batch_seq_tokens": config.batch_seq_tokens,
        "dtype": config.dtype,
        "platform": platform,
        "device_kind": device_kind,
    }
    key_str = json.dumps(key_data, sort_keys=True)
    return hashlib.sha256(key_str.encode()).hexdigest()


def _load_cache(cache_dir: str, key: str) -> MoETuningResult | None:
    """Load cached result if it exists."""
    cache_path = Path(cache_dir) / f"moe_tune_{key}.json"
    if not cache_path.exists():
        return None
    try:
        data = json.loads(cache_path.read_text())
        return MoETuningResult(
            gate_up_tiling=tuple(data["gate_up_tiling"]) if data["gate_up_tiling"] else None,
            down_tiling=tuple(data["down_tiling"]) if data["down_tiling"] else None,
            backend=data["backend"],
            throughput_gflops=data["throughput_gflops"],
        )
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def _save_cache(cache_dir: str, key: str, result: MoETuningResult) -> None:
    """Save result to disk cache."""
    cache_path = Path(cache_dir) / f"moe_tune_{key}.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "gate_up_tiling": list(result.gate_up_tiling) if result.gate_up_tiling else None,
        "down_tiling": list(result.down_tiling) if result.down_tiling else None,
        "backend": result.backend,
        "throughput_gflops": result.throughput_gflops,
    }
    cache_path.write_text(json.dumps(data, indent=2))


def _generate_candidates(
    M: int, K: int, N: int
) -> list[tuple[int, int, int]]:
    """Generate tiling candidates: powers of 2 in [32, 512], filtered by dims."""
    powers = [32, 64, 128, 256, 512]
    candidates = []
    for tm in powers:
        if tm > M * 2:
            continue
        for tk in powers:
            if tk > K * 2:
                continue
            for tn in powers:
                if tn > N * 2:
                    continue
                candidates.append((tm, tk, tn))
    return candidates


def _benchmark_tiling(
    x: jax.Array,
    weights: jax.Array,
    group_sizes: jax.Array,
    tiling: tuple[int, int, int] | None,
    warmup_iters: int,
    bench_iters: int,
) -> float:
    """Benchmark a single tiling configuration, return throughput in GFLOPS."""
    from mintext.kernels.grouped_matmul import grouped_matmul

    fn = jax.jit(
        lambda x, w, gs: grouped_matmul(
            x, w, gs, tiling=tiling
        )
    )

    # Warmup
    for _ in range(warmup_iters):
        out = fn(x, weights, group_sizes)
        out.block_until_ready()

    # Benchmark
    start = time.perf_counter()
    for _ in range(bench_iters):
        out = fn(x, weights, group_sizes)
        out.block_until_ready()
    elapsed = time.perf_counter() - start

    M, K_in = x.shape
    K_out = weights.shape[2]
    flops = 2 * M * K_in * K_out  # matmul FLOPS
    gflops = (flops * bench_iters) / elapsed / 1e9
    return gflops


def _benchmark_tiling_vjp(
    x: jax.Array,
    weights: jax.Array,
    group_sizes: jax.Array,
    tiling_9: tuple[int, ...],
    warmup_iters: int,
    bench_iters: int,
) -> float:
    """Benchmark a 9-param tiling with custom VJP (fwd+bwd), return throughput."""
    from mintext.kernels.grouped_matmul import grouped_matmul_vjp

    def loss(x_, w_):
        return grouped_matmul_vjp(x_, w_, group_sizes, tiling=tiling_9).sum()

    grad_fn = jax.jit(jax.grad(loss, argnums=(0, 1)))

    for _ in range(warmup_iters):
        dx, dw = grad_fn(x, weights)
        dx.block_until_ready()

    start = time.perf_counter()
    for _ in range(bench_iters):
        dx, dw = grad_fn(x, weights)
        dx.block_until_ready()
    elapsed = time.perf_counter() - start

    M, K_in = x.shape
    K_out = weights.shape[2]
    flops = 2 * M * K_in * K_out * 4  # fwd + bwd (~4x)
    gflops = (flops * bench_iters) / elapsed / 1e9
    return gflops


def autotune_moe(
    config: MoETuningConfig,
    cache_dir: str | None = None,
    max_trials: int = 20,
    warmup_iters: int = 3,
    bench_iters: int = 10,
) -> MoETuningResult:
    """Auto-tune MoE grouped matmul tiling parameters.

    Args:
        config: Problem dimensions.
        cache_dir: Directory for caching results. None = no caching.
        max_trials: Maximum number of tiling candidates to try.
        warmup_iters: JIT warmup iterations per candidate.
        bench_iters: Benchmark iterations per candidate.

    Returns:
        MoETuningResult with best tiling for gate_up and down projections.
    """
    # Check cache
    if cache_dir:
        key = _cache_key(config)
        cached = _load_cache(cache_dir, key)
        if cached is not None:
            logger.info("MoE autotuner: loaded cached result (%.1f GFLOPS)", cached.throughput_gflops)
            return cached

    dtype = getattr(jnp, config.dtype)
    E = config.num_experts
    D = config.hidden_size
    I = config.moe_intermediate_size
    M = config.batch_seq_tokens * config.num_experts_per_tok

    rng = jax.random.key(42)
    k1, k2, k3 = jax.random.split(rng, 3)

    # Check if ragged_dot is available
    try:
        test_x = jnp.ones((4, 4), dtype=dtype)
        test_w = jnp.ones((2, 4, 4), dtype=dtype)
        test_gs = jnp.array([2, 2], dtype=jnp.int32)
        jax.lax.ragged_dot(lhs=test_x, rhs=test_w, group_sizes=test_gs)
        has_ragged_dot = True
    except Exception:
        has_ragged_dot = False

    if not has_ragged_dot:
        logger.warning("MoE autotuner: ragged_dot not available, returning default tiling")
        result = MoETuningResult(
            gate_up_tiling=None, down_tiling=None,
            backend="ragged_dot", throughput_gflops=0.0,
        )
        if cache_dir:
            _save_cache(cache_dir, key, result)
        return result

    # Tune gate_up projection: [M, D] @ [E, D, 2*I] -> [M, 2*I]
    logger.info("MoE autotuner: tuning gate_up (%d, %d) @ (%d, %d, %d)", M, D, E, D, 2 * I)

    # Create synthetic data
    x_gu = jax.random.normal(k1, (M, D), dtype=dtype)
    w_gu = jax.random.normal(k2, (E, D, 2 * I), dtype=dtype)
    # Uniform group sizes for tuning
    base_size = M // E
    remainder = M % E
    gs = jnp.array(
        [base_size + (1 if i < remainder else 0) for i in range(E)],
        dtype=jnp.int32,
    )

    candidates_gu = _generate_candidates(M, D, 2 * I)
    if len(candidates_gu) > max_trials:
        # Sample evenly
        step = len(candidates_gu) // max_trials
        candidates_gu = candidates_gu[::step][:max_trials]

    best_gu_tiling = None
    best_gu_gflops = 0.0

    # Benchmark no-tiling baseline
    baseline_gflops = _benchmark_tiling(x_gu, w_gu, gs, None, warmup_iters, bench_iters)
    best_gu_gflops = baseline_gflops
    logger.info("  baseline (no tiling): %.1f GFLOPS", baseline_gflops)

    for tiling in candidates_gu:
        try:
            gflops = _benchmark_tiling(x_gu, w_gu, gs, tiling, warmup_iters, bench_iters)
            if gflops > best_gu_gflops:
                best_gu_gflops = gflops
                best_gu_tiling = tiling
                logger.info("  new best gate_up: %s -> %.1f GFLOPS", tiling, gflops)
        except Exception as e:
            logger.debug("  tiling %s failed: %s", tiling, e)

    # Tune down projection: [M, I] @ [E, I, D] -> [M, D]
    logger.info("MoE autotuner: tuning down (%d, %d) @ (%d, %d, %d)", M, I, E, I, D)
    x_dn = jax.random.normal(k3, (M, I), dtype=dtype)
    w_dn = jax.random.normal(k1, (E, I, D), dtype=dtype)

    candidates_dn = _generate_candidates(M, I, D)
    if len(candidates_dn) > max_trials:
        step = len(candidates_dn) // max_trials
        candidates_dn = candidates_dn[::step][:max_trials]

    best_dn_tiling = None
    best_dn_gflops = 0.0

    baseline_dn = _benchmark_tiling(x_dn, w_dn, gs, None, warmup_iters, bench_iters)
    best_dn_gflops = baseline_dn
    logger.info("  baseline (no tiling): %.1f GFLOPS", baseline_dn)

    for tiling in candidates_dn:
        try:
            gflops = _benchmark_tiling(x_dn, w_dn, gs, tiling, warmup_iters, bench_iters)
            if gflops > best_dn_gflops:
                best_dn_gflops = gflops
                best_dn_tiling = tiling
                logger.info("  new best down: %s -> %.1f GFLOPS", tiling, gflops)
        except Exception as e:
            logger.debug("  tiling %s failed: %s", tiling, e)

    total_gflops = best_gu_gflops + best_dn_gflops
    result = MoETuningResult(
        gate_up_tiling=best_gu_tiling,
        down_tiling=best_dn_tiling,
        backend="ragged_dot",
        throughput_gflops=total_gflops,
    )

    if cache_dir:
        _save_cache(cache_dir, key, result)
        logger.info("MoE autotuner: saved to cache (%.1f GFLOPS total)", total_gflops)

    return result
