#!/bin/bash
# Quick benchmark on TPU to verify MFU logging.
# Usage: bash scripts/run_tpu_benchmark.sh

set -euo pipefail

cd "$(dirname "$0")/.."

python -m mintext.train \
  --config configs/base.yml \
  steps=20 \
  per_device_batch_size=4 \
  max_position_embeddings=2048 \
  num_hidden_layers=2 \
  hidden_size=512 \
  num_attention_heads=8 \
  intermediate_size=1024 \
  vocab_size=32000 \
  dataset_type=synthetic \
  enable_checkpointing=false
