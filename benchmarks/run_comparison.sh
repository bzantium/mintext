#!/bin/bash
# Run MinText vs MaxText comparison benchmark for Llama-3.2-1B on TPU v6e-8
set -e

MINTEXT_DIR="/home/ryan/mintext"
MAXTEXT_DIR="/home/ryan/maxtext"
RESULTS_DIR="${MINTEXT_DIR}/benchmarks/results"
mkdir -p "$RESULTS_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "============================================================"
echo "MinText vs MaxText Benchmark Comparison"
echo "Model: Llama-3.2-1B | Hardware: TPU v6e-8"
echo "Timestamp: $TIMESTAMP"
echo "============================================================"

# --- MinText Benchmark ---
echo ""
echo ">>> Running MinText benchmark..."
echo "------------------------------------------------------------"
cd "$MINTEXT_DIR"
python -m mintext.train \
    --config configs/benchmarks/llama-3.2-1b-v6e-8.yml \
    2>&1 | tee "${RESULTS_DIR}/mintext_${TIMESTAMP}.log"

echo ""
echo ">>> Running MaxText benchmark..."
echo "------------------------------------------------------------"
cd "$MAXTEXT_DIR"
python3 -m maxtext.trainers.pre_train.train \
    src/maxtext/configs/base.yml \
    run_name=bench-llama-1b-${TIMESTAMP} \
    base_output_directory=/tmp/maxtext-bench \
    model_name=llama3-8b \
    base_emb_dim=2048 \
    base_num_query_heads=32 \
    base_num_kv_heads=8 \
    base_num_decoder_layers=16 \
    base_mlp_dim=8192 \
    head_dim=64 \
    vocab_size=128256 \
    decoder_block=llama2 \
    mlp_activations="['silu','linear']" \
    logits_via_embedding=true \
    normalization_layer_epsilon=1e-5 \
    rope_max_timescale=500000 \
    max_target_length=8192 \
    per_device_batch_size=8 \
    weight_dtype=bfloat16 \
    dtype=bfloat16 \
    remat_policy=full \
    scan_layers=true \
    dataset_type=synthetic \
    steps=30 \
    enable_checkpointing=false \
    2>&1 | tee "${RESULTS_DIR}/maxtext_${TIMESTAMP}.log"

echo ""
echo "============================================================"
echo "Comparison complete. Logs saved to:"
echo "  MinText: ${RESULTS_DIR}/mintext_${TIMESTAMP}.log"
echo "  MaxText: ${RESULTS_DIR}/maxtext_${TIMESTAMP}.log"
echo "============================================================"
