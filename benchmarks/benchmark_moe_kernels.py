#!/usr/bin/env python3
"""Benchmark MoE kernel: ragged_dot and ragged_dot with custom VJP.

Compares forward, backward, and fwd+bwd at kernel level.

Usage:
    python benchmarks/benchmark_moe_kernels.py
    python benchmarks/benchmark_moe_kernels.py --configs tiny small medium dsv3_small
    python benchmarks/benchmark_moe_kernels.py --modes forward backward fwd_bwd
"""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp


CONFIGS = {
    "tiny":       {"E": 4,   "D": 64,    "I": 128,   "K": 2, "tokens": 128},
    "small":      {"E": 4,   "D": 128,   "I": 256,   "K": 2, "tokens": 256},
    "medium":     {"E": 8,   "D": 256,   "I": 512,   "K": 2, "tokens": 1024},
    "dsv3_small": {"E": 64,  "D": 512,   "I": 1024,  "K": 8, "tokens": 2048},
    "dsv3":       {"E": 256, "D": 7168,  "I": 2048,  "K": 8, "tokens": 4096},
    "llama_moe":  {"E": 8,   "D": 4096,  "I": 14336, "K": 2, "tokens": 4096},
}


def _make_data(M, D, N, E, dtype=jnp.bfloat16):
    rng = jax.random.key(0)
    k1, k2 = jax.random.split(rng)
    x = jax.random.normal(k1, (M, D), dtype=dtype)
    w = jax.random.normal(k2, (E, D, N), dtype=dtype)
    base = M // E
    rem = M % E
    gs = jnp.array([base + (1 if i < rem else 0) for i in range(E)], dtype=jnp.int32)
    return x, w, gs


def benchmark_forward(
    M, D, N, E, backend, warmup=3, iters=10
) -> tuple[float, float]:
    from mintext.kernels.grouped_matmul import grouped_matmul, grouped_matmul_vjp

    x, w, gs = _make_data(M, D, N, E)

    if backend == "ragged_dot_vjp":
        fn = jax.jit(lambda x, w, gs: grouped_matmul_vjp(x, w, gs))
    else:
        fn = jax.jit(lambda x, w, gs: grouped_matmul(x, w, gs))

    for _ in range(warmup):
        fn(x, w, gs).block_until_ready()

    start = time.perf_counter()
    for _ in range(iters):
        fn(x, w, gs).block_until_ready()
    elapsed = time.perf_counter() - start

    latency_ms = (elapsed / iters) * 1000
    flops = 2 * M * D * N
    gflops = (flops * iters) / elapsed / 1e9
    return latency_ms, gflops


def benchmark_backward(
    M, D, N, E, backend, warmup=3, iters=10
) -> tuple[float, float]:
    from mintext.kernels.grouped_matmul import grouped_matmul, grouped_matmul_vjp

    x, w, gs = _make_data(M, D, N, E)

    if backend == "ragged_dot_vjp":
        loss = lambda x, w, gs: grouped_matmul_vjp(x, w, gs).sum()
    else:
        loss = lambda x, w, gs: grouped_matmul(x, w, gs).sum()

    grad_fn = jax.jit(jax.grad(loss, argnums=(0, 1)))

    for _ in range(warmup):
        dx, dw = grad_fn(x, w, gs)
        dx.block_until_ready()

    start = time.perf_counter()
    for _ in range(iters):
        dx, dw = grad_fn(x, w, gs)
        dx.block_until_ready()
    elapsed = time.perf_counter() - start

    latency_ms = (elapsed / iters) * 1000
    # backward has ~3x flops of forward (dlhs + drhs)
    flops = 2 * M * D * N * 3
    gflops = (flops * iters) / elapsed / 1e9
    return latency_ms, gflops


def benchmark_fwd_bwd(
    M, D, N, E, backend, warmup=3, iters=10
) -> tuple[float, float]:
    from mintext.kernels.grouped_matmul import grouped_matmul, grouped_matmul_vjp

    x, w, gs = _make_data(M, D, N, E)

    if backend == "ragged_dot_vjp":
        loss = lambda x, w, gs: grouped_matmul_vjp(x, w, gs).sum()
    else:
        loss = lambda x, w, gs: grouped_matmul(x, w, gs).sum()

    val_grad_fn = jax.jit(jax.value_and_grad(loss, argnums=(0, 1)))

    for _ in range(warmup):
        v, (dx, dw) = val_grad_fn(x, w, gs)
        dx.block_until_ready()

    start = time.perf_counter()
    for _ in range(iters):
        v, (dx, dw) = val_grad_fn(x, w, gs)
        dx.block_until_ready()
    elapsed = time.perf_counter() - start

    latency_ms = (elapsed / iters) * 1000
    flops = 2 * M * D * N * 4  # fwd + bwd
    gflops = (flops * iters) / elapsed / 1e9
    return latency_ms, gflops


BENCH_FNS = {
    "forward": benchmark_forward,
    "backward": benchmark_backward,
    "fwd_bwd": benchmark_fwd_bwd,
}


def main():
    parser = argparse.ArgumentParser(description="MoE kernel benchmark")
    parser.add_argument(
        "--backends", nargs="+",
        default=["ragged_dot", "ragged_dot_vjp"],
    )
    parser.add_argument("--configs", nargs="+", default=["tiny", "small", "medium"])
    parser.add_argument("--modes", nargs="+", default=["forward", "backward", "fwd_bwd"])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    print(f"Platform: {jax.devices()[0].platform}, Device: {jax.devices()[0].device_kind}")
    print(f"JAX version: {jax.__version__}, Devices: {jax.device_count()}")
    print()

    active = args.backends

    for mode in args.modes:
        bench_fn = BENCH_FNS[mode]
        print("=" * 90)
        print(f"  {mode.upper()}")
        print("=" * 90)

        header = f"{'Config':<12} {'M':>6} {'D':>5} {'N':>5} {'E':>4}"
        for b in active:
            header += f"  | {b:>14} ms  {'GFLOPS':>8}"
        print(header)
        print("-" * len(header))

        for cfg_name in args.configs:
            if cfg_name not in CONFIGS:
                print(f"  Unknown config: {cfg_name}")
                continue
            cfg = CONFIGS[cfg_name]
            E, D, I, K, tokens = cfg["E"], cfg["D"], cfg["I"], cfg["K"], cfg["tokens"]
            M = tokens * K
            N = 2 * I  # gate_up fused

            row = f"{cfg_name:<12} {M:>6} {D:>5} {N:>5} {E:>4}"

            for b in active:
                try:
                    lat, gf = bench_fn(M, D, N, E, b, args.warmup, args.iters)
                    row += f"  | {lat:>14.3f}     {gf:>8.1f}"
                except Exception as e:
                    row += f"  | {'ERROR':>14}     {'N/A':>8}"
                    print(f"    [{b}] {cfg_name}: {e}")

            print(row)

        print()

    print("Done.")


if __name__ == "__main__":
    main()
