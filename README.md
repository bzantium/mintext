# MinText

Minimal high-performance JAX LLM training framework.

## Requirements

- Python >= 3.11 (flax 0.12+ requirement)
- [uv](https://docs.astral.sh/uv/) package manager
- NVIDIA GPU (CUDA 12) or Google Cloud TPU

## Setup

### GPU (CUDA 12)

```bash
git clone <repo-url> && cd mintext

# Option A: one-liner script
bash scripts/setup_gpu.sh

# Option B: manual
uv pip install --system -e ".[gpu]"
uv pip install --system -e ".[dev]"    # pytest, ruff, transformers, torch
```

Verify:
```bash
python3 -c "import jax; print(jax.devices())"
# [CudaDevice(id=0), CudaDevice(id=1), ...]
```

### TPU

**Step 1: Sync code** from the GPU machine (or wherever you have the repo):

```bash
# Default target: tpu-v6e-8:~/mintext
bash scripts/sync_tpu.sh

# Custom target
TPU_HOST=my-tpu-vm TPU_DIR=~/my-path bash scripts/sync_tpu.sh
```

**Step 2: Run one-time setup** on the TPU VM:

```bash
ssh tpu-v6e-8
cd ~/mintext
bash scripts/setup_tpu.sh
```

This creates a Python 3.11 venv (TPU VMs ship with Python 3.10 which is too old for flax 0.12+),
installs `uv`, and runs `pip install -e ".[tpu]"` with libtpu find-links.

**Step 3: Activate the venv** before running anything:

```bash
source ~/mintext/.venv/bin/activate
python -c "import jax; print(jax.devices())"
# [TpuDevice(id=0, ...), ..., TpuDevice(id=7, ...)]
```

### Optional extras

```bash
# HuggingFace transformers (for checkpoint conversion)
uv pip install -e ".[data]"

# Tokamax MoE kernels
uv pip install -e ".[kernels]"
```

## Dependency structure

`jax` and `jaxlib` are **not** in core dependencies — they're platform-specific:

| Extra | What it installs |
|---|---|
| `.[gpu]` | `jax[cuda12]` (CUDA 12 jaxlib) |
| `.[tpu]` | `jax[tpu]` (libtpu jaxlib) |
| `.[dev]` | pytest, ruff, transformers, torch |
| `.[data]` | transformers |
| `.[kernels]` | tokamax |

Core deps (`flax`, `optax`, etc.) pull in a base `jax` transitively. The extras ensure the correct platform-specific jaxlib backend.

## Quick start

```bash
# 10-step training on synthetic data (works on any platform)
python -m mintext.train --config configs/base.yml

# Train with a model config override
python -m mintext.train --config configs/base.yml --config configs/models/test-tiny.yml
```

## Tests

```bash
python -m pytest tests/ -x
```

## Development workflow (GPU + TPU)

1. Develop and test on GPU
2. Sync to TPU: `bash scripts/sync_tpu.sh`
3. Run on TPU: `ssh tpu-v6e-8 'cd ~/mintext && source .venv/bin/activate && python -m mintext.train ...'`

The sync script uses `rsync --delete` and excludes build artifacts (`.git`, `__pycache__`, `.egg-info`, etc.).

## Supported models

- Llama (Llama 2, Llama 3)
- Qwen3
- DeepSeek-V3 (MLA + MoE)
- Qwen3-Next (hybrid full + linear attention)

## Project structure

```
mintext/
  config.py              # Pydantic config
  train.py               # Training entry point
  trainer.py             # Trainer class
  models/
    transformer.py       # Core transformer (Flax NNX)
    layers.py            # Attention, MLP, RMSNorm
    positions.py         # RoPE variants
    moe.py               # Mixture-of-Experts
    linear_attention.py  # Gated Delta Rule
  data/
    pipeline.py          # Grain data pipeline
  distributed/
    mesh.py              # Device mesh (auto TPU/GPU)
    sharding.py          # Sharding rules
  optim/
    optimizer.py         # AdamW / Muon + LR schedules
  checkpoint/
    manager.py           # Orbax checkpoint manager
    conversion.py        # HuggingFace import/export
  kernels/
    moe_dispatch.py      # Token routing (sort-based)
    grouped_matmul.py    # Grouped matmul backends
  utils/
    logging.py           # TensorBoard logging
    profiling.py         # JAX profiler
configs/
  base.yml               # Base config
  models/                # Model-specific overrides
scripts/
  setup_gpu.sh           # GPU setup
  setup_tpu.sh           # TPU setup
  sync_tpu.sh            # Sync code to TPU VM
tests/                   # pytest test suite
```
