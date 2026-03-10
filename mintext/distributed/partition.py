"""Partition specs for model parameters.

This module provides utilities for sharded model initialization,
ensuring parameters are created directly on the correct devices.
"""

from __future__ import annotations

from flax import nnx

from mintext.config import MinTextConfig
from mintext.distributed.sharding import get_logical_axis_rules


def create_sharded_model(
    config: MinTextConfig,
    mesh: Mesh,
    model_cls: type[nnx.Module] | None = None,
) -> nnx.Module:
    """Create a model with parameters sharded across the mesh.

    Uses jax.jit with out_shardings to initialize parameters directly
    on the correct shards, avoiding gathering on a single device.

    Args:
        config: MinText config.
        mesh: Device mesh.
        model_cls: Model class. Defaults to Transformer.

    Returns:
        Initialized model with sharded parameters.
    """
    if model_cls is None:
        from mintext.models import Transformer

        model_cls = Transformer

    # Create the model (mesh and rules should already be set globally)
    model = model_cls(config, rngs=nnx.Rngs(params=config.seed), mesh=mesh)
    return model
