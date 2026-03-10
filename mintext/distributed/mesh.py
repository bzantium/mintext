"""Device mesh creation with auto TPU/GPU detection and multi-node support."""

from __future__ import annotations

import logging
import os

import numpy as np
import jax
from jax.experimental import mesh_utils
from jax.sharding import Mesh
from flax.core import spmd

from mintext.config import MinTextConfig
from mintext.distributed.sharding import get_logical_axis_rules

logger = logging.getLogger(__name__)


def initialize_distributed(timeout: int = 300) -> None:
    """Initialize JAX distributed runtime for multi-node training.

    Reads configuration from environment variables:
      - JAX_COORDINATOR_IP: IP of the coordinator node
      - JAX_COORDINATOR_PORT: Port (default: 29500)
      - NNODES: Total number of nodes
      - NODE_RANK: This node's rank
      - CUDA_VISIBLE_DEVICES: GPU devices to use (optional)

    On TPU, jax.distributed.initialize() is called without arguments
    (auto-discovers from TPU topology).
    """
    if jax.distributed.is_initialized():
        logger.info("JAX distributed already initialized")
        return

    coordinator_ip = os.environ.get("JAX_COORDINATOR_IP")

    if coordinator_ip:
        # GPU multi-node via SLURM or manual env vars
        coordinator_port = os.environ.get("JAX_COORDINATOR_PORT", "29500")
        num_processes = int(os.environ.get("NNODES", "1"))
        process_id = int(os.environ.get("NODE_RANK", "0"))

        local_device_ids = None
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible:
            local_device_ids = [int(x) for x in cuda_visible.split(",")]

        logger.info(
            "Initializing JAX distributed: coordinator=%s:%s, "
            "num_processes=%d, process_id=%d, local_devices=%s",
            coordinator_ip, coordinator_port, num_processes, process_id, local_device_ids,
        )
        jax.distributed.initialize(
            coordinator_address=f"{coordinator_ip}:{coordinator_port}",
            num_processes=num_processes,
            process_id=process_id,
            local_device_ids=local_device_ids,
            initialization_timeout=timeout,
        )
    else:
        # Check if we're on TPU and need multi-host init
        try:
            platform = jax.devices()[0].platform
        except Exception:
            platform = "gpu"

        if platform == "tpu":
            logger.info("Initializing JAX distributed for TPU")
            jax.distributed.initialize(initialization_timeout=timeout)
        else:
            logger.info("Single-node setup, skipping distributed init")


def _detect_num_slices(devices: list) -> int:
    """Detect number of slices from device metadata."""
    try:
        return max(d.slice_index for d in devices) + 1
    except (AttributeError, ValueError):
        return 1


def _resolve_auto_parallelism(
    ici_data: int, ici_fsdp: int, ici_tensor: int, num_devices: int
) -> tuple[int, int, int]:
    """Resolve -1 (auto) values in parallelism config.

    Exactly one axis may be -1, which gets the remaining devices.
    """
    values = [ici_data, ici_fsdp, ici_tensor]
    names = ["data", "fsdp", "tensor"]
    auto_count = values.count(-1)

    if auto_count > 1:
        raise ValueError(
            f"At most one ICI parallelism axis can be -1 (auto), got {auto_count}: "
            f"data={ici_data}, fsdp={ici_fsdp}, tensor={ici_tensor}"
        )

    if auto_count == 0:
        product = ici_data * ici_fsdp * ici_tensor
        if product != num_devices:
            raise ValueError(
                f"ICI parallelism product ({product}) != num_devices ({num_devices}). "
                f"data={ici_data}, fsdp={ici_fsdp}, tensor={ici_tensor}"
            )
        return ici_data, ici_fsdp, ici_tensor

    # Resolve the -1 axis
    known_product = 1
    auto_idx = -1
    for i, v in enumerate(values):
        if v == -1:
            auto_idx = i
        else:
            known_product *= v

    if num_devices % known_product != 0:
        raise ValueError(
            f"Cannot auto-resolve {names[auto_idx]} axis: "
            f"num_devices ({num_devices}) not divisible by product of other axes ({known_product})"
        )

    values[auto_idx] = num_devices // known_product
    logger.info(
        "Auto-resolved mesh: data=%d, fsdp=%d, tensor=%d (num_devices=%d)",
        values[0], values[1], values[2], num_devices,
    )
    return values[0], values[1], values[2]


def create_mesh(config: MinTextConfig, devices: list | None = None) -> Mesh:
    """Create a device mesh from config.

    Supports both single-slice and multi-slice (DCN) configurations.

    Args:
        config: MinText config with ICI/DCN parallelism settings.
        devices: Optional device list. Defaults to jax.devices().

    Returns:
        JAX Mesh with axes (data, fsdp, tensor).
    """
    if devices is None:
        devices = jax.devices()

    num_devices = len(devices)
    platform = devices[0].platform
    num_slices = _detect_num_slices(devices)

    # Override from config if explicitly set
    if config.num_slices > 1:
        num_slices = config.num_slices

    logger.info(
        "Creating mesh on %s with %d devices, %d slices",
        platform, num_devices, num_slices,
    )

    devices_per_slice = num_devices // num_slices

    ici_data, ici_fsdp, ici_tensor = _resolve_auto_parallelism(
        config.ici_data_parallelism,
        config.ici_fsdp_parallelism,
        config.ici_tensor_parallelism,
        devices_per_slice,
    )

    mesh_shape = (ici_data, ici_fsdp, ici_tensor)

    if num_slices > 1:
        # Multi-slice: hybrid mesh with DCN axes
        dcn_shape = (
            config.dcn_data_parallelism,
            config.dcn_fsdp_parallelism,
            config.dcn_tensor_parallelism,
        )
        try:
            devices_array = mesh_utils.create_hybrid_device_mesh(
                mesh_shape, dcn_shape, devices
            )
        except Exception as e:
            logger.warning("Hybrid mesh creation failed: %s, falling back", e)
            devices_array = np.array(devices).reshape(
                dcn_shape[0] * mesh_shape[0],
                dcn_shape[1] * mesh_shape[1],
                dcn_shape[2] * mesh_shape[2],
            )
    else:
        # Single-slice
        try:
            devices_array = mesh_utils.create_device_mesh(mesh_shape, devices)
        except Exception:
            # Fallback: simple reshape if mesh_utils fails (e.g., single CPU)
            devices_array = np.array(devices).reshape(mesh_shape)

    mesh = Mesh(devices_array, ("data", "fsdp", "tensor"))
    return mesh


def setup_mesh(config: MinTextConfig) -> Mesh:
    """Create mesh and set it as the global mesh for NNX sharding.

    This sets both `jax.sharding.set_mesh` and Flax logical axis rules,
    which are required before creating NNX modules with sharding annotations.

    Args:
        config: MinText config.

    Returns:
        The created Mesh.
    """
    mesh = create_mesh(config)
    jax.sharding.set_mesh(mesh)
    spmd.set_logical_axis_rules(get_logical_axis_rules(config))
    logger.info("Global mesh set: %s", mesh.shape)
    return mesh
