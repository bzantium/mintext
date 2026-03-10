#!/bin/bash
# Benchmark Llama 3.2 1B on TPU v6e-8 with multiple batch sizes.
# Usage: bash scripts/benchmark_llama_1b.sh [batch_sizes...]
# Example: bash scripts/benchmark_llama_1b.sh 1 4 8 16
#
# Default batch sizes: 1 4 8 16 32

set -euo pipefail
cd "$(dirname "$0")/.."

# XLA TPU optimization flags
export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:-} --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true"

# Default batch sizes if none provided
BATCH_SIZES="${@:-1 4 8 16 32}"

SEQ_LEN=4096
STEPS=20
CONFIG=configs/models/meta-llama-3.2-1b.yml

echo "=== Llama 3.2 1B Benchmark ==="
echo "Config: ${CONFIG}"
echo "Seq length: ${SEQ_LEN}"
echo "Steps: ${STEPS}"
echo "Batch sizes: ${BATCH_SIZES}"
echo "LIBTPU_INIT_ARGS: ${LIBTPU_INIT_ARGS}"
echo ""

for BS in ${BATCH_SIZES}; do
    echo "=============================================="
    echo "=== per_device_batch_size=${BS} seq_len=${SEQ_LEN} ==="
    echo "=============================================="
    python -m mintext.train \
        --config "${CONFIG}" \
        steps="${STEPS}" \
        per_device_batch_size="${BS}" \
        seq_length="${SEQ_LEN}" \
        dataset_type=synthetic \
        enable_checkpointing=false \
        enable_tensorboard=false \
        weight_dtype=bfloat16 \
        remat_policy=full \
        scan_layers=true \
        log_period=1 \
    2>&1 || echo "FAILED with bs=${BS}"
    echo ""
done
