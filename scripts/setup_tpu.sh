#!/bin/bash
set -e
# One-time TPU VM setup: create venv with Python 3.11, install MinText with TPU deps

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.local/bin/env
fi

cd ~/mintext

# Create venv with Python 3.11 (flax 0.12+ requires Python >= 3.11)
VENV_DIR=~/mintext/.venv
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating venv with Python 3.11..."
    uv venv --python=python3.11 "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

uv pip install -e ".[tpu]" --find-links https://storage.googleapis.com/jax-releases/libtpu_releases.html
uv pip install -e ".[dev]"

# Verify
python -c "import jax; print(f'JAX {jax.__version__}, devices: {jax.devices()}')"
python -c "from mintext.config import MinTextConfig; print('MinText OK')"
echo ""
echo "Setup complete. Activate with: source ~/mintext/.venv/bin/activate"
