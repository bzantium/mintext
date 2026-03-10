# MinText Performance Optimization Progress

## Target
Match MaxText MFU for Llama-3.2-1B on TPU v6e-8 (within 5%).

## Hardware
- TPU v6e-8 (8 chips, single host)
- Peak BF16: 918 TFLOP/s/chip

## Config
- Model: Llama-3.2-1B (1.24B params)
- Sequence length: 2048
- Per-device batch size: 4
- dtype/weight_dtype: bfloat16
- Remat: full
- Mesh: FSDP=8

## Results

| Step | Optimization | MFU% | Step Time (s) | TFLOP/s/dev | Gap vs MaxText |
|------|-------------|------|---------------|-------------|----------------|
| 0 | MinText Baseline (jax.nn.dot_product_attention) | 14.6% | 0.478 | 134.0 | 28% |
| 1 | + Splash Attention (shard_map, with K/V repeat) | 18.1% | 0.386 | 166.0 | 11% |
| 2 | + Native GQA (no K/V repeat in Splash) | 19.1% | 0.366 | 175.0 | **6.4%** |
| 3 | + save_qkv_proj remat (full→save_qkv_proj) | 19.4% | 0.359 | 178.5 | **5.6%** |
| - | MaxText Reference (full remat) | 20.3% | 0.343 | 186.7 | target |
| - | MaxText Reference (save_qkv_proj remat) | 20.7% | 0.337 | 190.0 | target |

## Key Changes
1. **Splash Attention enabled for head_dim=64** — JAX 0.9.1 Splash uses ceiling division (`pl.cdiv`), NOT `divmod`, so head_dim=64 works fine. Removed incorrect head_dim>=128 guard.
2. **shard_map wrapper** — Pallas/Mosaic kernels require explicit `shard_map`, not auto-partitioning.
3. **Native GQA** — Splash MHA handles Q[32 heads] + K/V[8 heads] natively; no need to repeat K/V.
4. **Backward block sizes** — Splash requires explicit backward block sizes (block_q_dkv, block_kv_dkv, etc.) for gradient computation.

## Remaining Gap Analysis
- With full remat: MinText 175 vs MaxText 187 = **6.4% gap**
- With save_qkv_proj: MinText 178.5 vs MaxText 190 = **5.6% gap**
- Gap is consistent across batch sizes (4 and 8) and remat policies
- XLA compiler flags (VMEM limit, async fusion) hurt small-model performance
- Possible sources of remaining gap:
  - MaxText uses `lax.dot_general(out_sharding=...)` for explicit output sharding
  - MaxText applies `with_sharding_constraint` on MLP intermediates
  - Flax Linen bridge overhead differences

## Notes
- Warmup steps excluded (first 2 for JIT compilation)
- Measured over steps 2-29 (28 timed steps)
- Python 3.12, JAX 0.9.1 for both
- MinText: uv venv at /home/ryan/mintext/.venv
- MaxText: uv venv at /home/ryan/maxtext/.venv
