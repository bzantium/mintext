#!/usr/bin/env python3
"""End-to-end MoE block benchmark: route -> GMM -> SwiGLU -> GMM -> unroute.

Compares forward+backward with autodiff vs custom_vjp.

Usage:
    python benchmarks/benchmark_moe_e2e.py
    python benchmarks/benchmark_moe_e2e.py --configs tiny small medium
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import jax
import jax.numpy as jnp
from flax import nnx
from flax.core import spmd

from mintext.config import MinTextConfig, DEFAULT_LOGICAL_AXIS_RULES
from mintext.modules.moe import MoEBlock


CONFIGS = {
    "tiny":       {"E": 4,   "D": 64,   "I": 128,  "K": 2, "B": 2, "S": 32,  "n_group": 2, "topk_group": 1},
    "small":      {"E": 8,   "D": 128,  "I": 256,  "K": 2, "B": 2, "S": 64,  "n_group": 4, "topk_group": 2},
    "medium":     {"E": 16,  "D": 256,  "I": 512,  "K": 4, "B": 2, "S": 128, "n_group": 4, "topk_group": 2},
    "dsv3_small": {"E": 64,  "D": 512,  "I": 1024, "K": 8, "B": 2, "S": 256, "n_group": 8, "topk_group": 4},
}


def _build_block(cfg_dict, use_custom_vjp):
    E = cfg_dict["E"]
    D = cfg_dict["D"]
    I = cfg_dict["I"]
    K = cfg_dict["K"]
    n_group = cfg_dict["n_group"]
    topk_group = cfg_dict["topk_group"]

    config = MinTextConfig(
        hidden_size=D, num_attention_heads=max(4, D // 64), intermediate_size=D * 4, vocab_size=256,
        num_experts=E, num_experts_per_tok=K, moe_intermediate_size=I,
        n_group=n_group, topk_group=topk_group,
        n_shared_experts=0, routed_scaling_factor=1.0,
        dtype="bfloat16", weight_dtype="bfloat16",
        moe_use_custom_vjp=use_custom_vjp,
    )
    rngs = nnx.Rngs(params=0, dropout=1)
    return MoEBlock(config, rngs=rngs)


def benchmark_moe_block(cfg_dict, use_custom_vjp, warmup=5, iters=20):
    B = cfg_dict["B"]
    S = cfg_dict["S"]
    D = cfg_dict["D"]

    block = _build_block(cfg_dict, use_custom_vjp)
    x = jax.random.normal(jax.random.key(42), (B, S, D), dtype=jnp.bfloat16)

    def loss_fn(x_):
        out, _ = block(x_)
        return out.sum()

    val_grad_fn = jax.jit(jax.value_and_grad(loss_fn))

    for _ in range(warmup):
        v, g = val_grad_fn(x)
        g.block_until_ready()

    start = time.perf_counter()
    for _ in range(iters):
        v, g = val_grad_fn(x)
        g.block_until_ready()
    elapsed = time.perf_counter() - start

    latency_ms = (elapsed / iters) * 1000

    # Estimate FLOPS: gate_up + down, both fwd+bwd
    E, K, I = cfg_dict["E"], cfg_dict["K"], cfg_dict["I"]
    M = B * S * K  # tokens after top-k routing
    # gate_up: M*D*2I, down: M*I*D, each fwd+bwd ~4x
    flops = (2 * M * D * 2 * I + 2 * M * I * D) * 4
    gflops = (flops * iters) / elapsed / 1e9

    return latency_ms, gflops


def main():
    parser = argparse.ArgumentParser(description="E2E MoE block benchmark")
    parser.add_argument("--configs", nargs="+", default=["tiny", "small", "medium"])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    print(f"Platform: {jax.devices()[0].platform}, Device: {jax.devices()[0].device_kind}")
    print(f"JAX version: {jax.__version__}, Devices: {jax.device_count()}")
    print()

    # Set up mesh for NNX modules
    devices = np.array(jax.devices()).reshape(1, -1, 1)
    mesh = jax.sharding.Mesh(devices, ("data", "fsdp", "tensor"))
    jax.sharding.set_mesh(mesh)
    spmd.set_logical_axis_rules(DEFAULT_LOGICAL_AXIS_RULES)

    variants = [
        (False, "ragged_dot (autodiff)"),
        (True, "ragged_dot (custom_vjp)"),
    ]

    print("=" * 100)
    print("  E2E MoE BLOCK BENCHMARK (fwd + bwd)")
    print("=" * 100)

    header = f"{'Config':<12} {'E':>4} {'D':>5} {'I':>5} {'K':>2} {'B*S':>6}"
    for _, label in variants:
        header += f"  | {label:>24} ms"
    print(header)
    print("-" * len(header))

    for cfg_name in args.configs:
        if cfg_name not in CONFIGS:
            print(f"  Unknown config: {cfg_name}")
            continue
        cfg = CONFIGS[cfg_name]
        E, D, I, K = cfg["E"], cfg["D"], cfg["I"], cfg["K"]
        BS = cfg["B"] * cfg["S"]

        row = f"{cfg_name:<12} {E:>4} {D:>5} {I:>5} {K:>2} {BS:>6}"

        for use_vjp, label in variants:
            try:
                lat, gf = benchmark_moe_block(cfg, use_vjp, args.warmup, args.iters)
                row += f"  | {lat:>18.3f} ({gf:>7.0f})"
            except Exception as e:
                row += f"  | {'ERROR':>18} ({str(e)[:7]})"
                print(f"    [{label}] {cfg_name}: {e}")

        print(row)

    print()
    print("Done.")


if __name__ == "__main__":
    main()
