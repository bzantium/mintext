"""Shared test fixtures for MinText."""

from pathlib import Path

import numpy as np
import jax
import pytest
from flax.core import spmd

# Force full float32 precision on GPU (A100 defaults to TF32 which has 10-bit mantissa)
jax.config.update("jax_default_matmul_precision", "float32")

from mintext.config import MinTextConfig, load_config

CONFIGS_DIR = Path(__file__).parent.parent / "configs"

# Logical axis rules mapping logical names to mesh axis names (or None for replicated)
_TEST_AXIS_RULES = [
    ("batch", "data"),
    ("embed", "fsdp"),
    ("heads", "tensor"),
    ("kv", None),
    ("kv_heads", None),
    ("kv_head_dim", None),
    ("mlp", "tensor"),
    ("vocab", "tensor"),
    ("norm", None),
    ("exp", None),
]


def _setup_test_mesh():
    """Set up a single-device mesh for testing."""
    devices = np.array(jax.devices()[:1])
    mesh = jax.sharding.Mesh(devices.reshape(1, 1, 1), ("data", "fsdp", "tensor"))
    jax.sharding.set_mesh(mesh)
    spmd.set_logical_axis_rules(_TEST_AXIS_RULES)


# Set up mesh at module import time so all tests have it
_setup_test_mesh()


@pytest.fixture
def tiny_config() -> MinTextConfig:
    """Return a minimal test config using defaults."""
    return MinTextConfig()


@pytest.fixture
def base_config() -> MinTextConfig:
    """Load config from base.yml."""
    return load_config(CONFIGS_DIR / "base.yml")


@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Path:
    """Return a temporary output directory."""
    out = tmp_path / "mintext_test"
    out.mkdir()
    return out
