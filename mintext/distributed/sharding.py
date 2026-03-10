"""Sharding rules and utilities."""

from __future__ import annotations

import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from flax import nnx

from mintext.config import MinTextConfig


from mintext.config import DEFAULT_LOGICAL_AXIS_RULES


def get_logical_axis_rules(
    config: MinTextConfig | None = None,
) -> list[tuple[str, str | None]]:
    """Get logical-to-mesh axis rules.

    If config provides custom rules, use those. Otherwise use defaults.
    """
    if config is not None and config.logical_axis_rules:
        return [(name, axis) for name, axis in config.logical_axis_rules]
    return DEFAULT_LOGICAL_AXIS_RULES


def _remove_size_one_axes(spec: P, mesh: Mesh) -> P:
    """Replace mesh axis references with None when that axis has size 1."""
    new_axes = []
    for axis in spec:
        if axis is None:
            new_axes.append(None)
        elif isinstance(axis, str):
            if mesh.shape.get(axis, 1) == 1:
                new_axes.append(None)
            else:
                new_axes.append(axis)
        elif isinstance(axis, tuple):
            filtered = tuple(a for a in axis if mesh.shape.get(a, 1) > 1)
            new_axes.append(filtered if filtered else None)
        else:
            new_axes.append(axis)
    return P(*new_axes)


def logical_to_pspec(
    logical_names: tuple[str | None, ...],
    rules: list[tuple[str, str | None]],
    mesh: Mesh,
) -> P:
    """Convert logical axis names to a PartitionSpec via rules.

    Args:
        logical_names: Logical axis names for each dimension.
        rules: Logical-to-mesh axis mapping.
        mesh: Device mesh (used to remove size-1 axes).

    Returns:
        PartitionSpec with mesh axis names.
    """
    rule_map = {}
    for logical, physical in rules:
        if logical not in rule_map:
            rule_map[logical] = physical

    mesh_axes = []
    for name in logical_names:
        if name is None:
            mesh_axes.append(None)
        else:
            mesh_axes.append(rule_map.get(name))

    spec = P(*mesh_axes)
    return _remove_size_one_axes(spec, mesh)


def create_named_sharding(
    mesh: Mesh,
    logical_names: tuple[str | None, ...],
    rules: list[tuple[str, str | None]] | None = None,
) -> NamedSharding:
    """Create a NamedSharding from logical axis names.

    Args:
        mesh: Device mesh.
        logical_names: Logical axis names.
        rules: Optional axis rules. Uses defaults if None.

    Returns:
        NamedSharding for the given logical axes.
    """
    if rules is None:
        rules = DEFAULT_LOGICAL_AXIS_RULES
    pspec = logical_to_pspec(logical_names, rules, mesh)
    return NamedSharding(mesh, pspec)


def add_data_axis_to_sharding(sharding: NamedSharding) -> NamedSharding:
    """Add 'data' axis to the first unsharded dimension of a NamedSharding.

    Used for ZeRO-1: optimizer state gets an extra 'data' axis so each
    data-parallel rank stores only a shard of the optimizer state.
    """
    pspec = sharding.spec
    for idx, partition in enumerate(pspec):
        if partition is None:
            new_parts = list(pspec)
            new_parts[idx] = "data"
            return NamedSharding(sharding.mesh, P(*new_parts))
    # All dims already sharded: compound-shard the first dim
    first = pspec[0]
    if isinstance(first, str):
        first = (first,)
    new_first = ("data",) + tuple(first)
    new_parts = list(pspec)
    new_parts[0] = new_first
    return NamedSharding(sharding.mesh, P(*new_parts))


def get_input_data_sharding(config: MinTextConfig, mesh: Mesh) -> NamedSharding:
    """Get sharding for input data tensors (tokens, positions).

    Data tensors have shape [batch, seq_len] and are sharded over the
    data axis (batch dimension), with sequence replicated.
    """
    rules = get_logical_axis_rules(config)
    return create_named_sharding(mesh, ("batch", "length"), rules)


def get_model_param_shardings(
    model: nnx.Module,
    mesh: Mesh,
    rules: list[tuple[str, str | None]] | None = None,
) -> dict:
    """Extract sharding specs from model parameters.

    Reads the `sharding` metadata from each nnx.Param and converts
    logical axis names to NamedSharding using the axis rules.

    Returns:
        A pytree of NamedSharding matching the model's state structure.
    """
    if rules is None:
        rules = DEFAULT_LOGICAL_AXIS_RULES

    _, state = nnx.split(model)

    def _get_sharding(leaf):
        if not hasattr(leaf, "sharding") or leaf.sharding is None:
            return NamedSharding(mesh, P())
        sharding = leaf.sharding
        # If set_mesh was active, sharding is already a NamedSharding
        if isinstance(sharding, NamedSharding):
            return sharding
        # Otherwise it's a tuple of logical axis names
        if isinstance(sharding, (tuple, list)):
            return create_named_sharding(mesh, tuple(sharding), rules)
        return NamedSharding(mesh, P())

    param_shardings = jax.tree.map(
        _get_sharding,
        state,
        is_leaf=lambda x: isinstance(x, nnx.Variable),
    )
    return param_shardings
