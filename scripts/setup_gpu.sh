#!/bin/bash
set -e
# GPU setup: install MinText with CUDA deps
cd "$(dirname "$0")/.."

uv pip install --system -e ".[gpu]"
uv pip install --system -e ".[dev]"

python3 -c "import jax; print(f'JAX {jax.__version__}, devices: {jax.devices()}')"
python3 -c "from mintext.config import MinTextConfig; print('MinText OK')"
