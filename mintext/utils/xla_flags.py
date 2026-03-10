"""XLA/LIBTPU compiler flags for TPU performance optimization.

Per-device-type flag presets based on MaxText's benchmarks/xla_flags_library.py.
Must be called before JAX backend initialization.

Key insight: full async fusion + VMEM flags hurt small single-host models (tested
5% regression on Llama-3.2-1B / v6e-8). These flags primarily benefit multi-slice
and larger model configs. For single-host small models, only the RNG flag helps.
"""

from __future__ import annotations

import logging
import os
import re

logger = logging.getLogger(__name__)

# --- Flag building blocks ---

FAST_RNG = "--xla_tpu_spmd_rng_bit_generator_unsafe=true"

DENSE_VMEM_LIMIT = "--xla_tpu_scoped_vmem_limit_kib=98304"

CF_FOR_ALL_GATHER = (
    "--xla_tpu_enable_async_collective_fusion=true "
    "--xla_tpu_enable_async_collective_fusion_fuse_all_gather=true "
    "--xla_tpu_enable_async_collective_fusion_multiple_steps=true "
    "--xla_tpu_overlap_compute_collective_tc=true "
    "--xla_enable_async_all_gather=true"
)

LAYOUT_FOR_ALL_REDUCE_SCATTER = (
    "--xla_tpu_use_minor_sharding_for_major_trivial_input=true "
    "--xla_tpu_relayout_group_size_threshold_for_reduce_scatter=1 "
    "--xla_tpu_assign_all_reduce_scatter_layout=true"
)

# Full flag set for large models / multi-slice
_FULL_FLAGS = " ".join([
    FAST_RNG,
    DENSE_VMEM_LIMIT,
    CF_FOR_ALL_GATHER,
    LAYOUT_FOR_ALL_REDUCE_SCATTER,
])

# Conservative flags for single-host small models
_CONSERVATIVE_FLAGS = FAST_RNG

# --- Per-device presets ---

# v6e (Trillium): conservative for single-host, full for multi-slice
# v6p / Ironwood: full flags (targets larger models)
# v5e / v5p: conservative for single-host, full for multi-slice
# v4: full flags
_DEVICE_PRESETS: dict[str, dict[str, str]] = {
    "v6e": {
        "single_host": _CONSERVATIVE_FLAGS,
        "multi_slice": _FULL_FLAGS,
    },
    "v6p": {
        "single_host": _FULL_FLAGS,
        "multi_slice": _FULL_FLAGS,
    },
    "v5e": {
        "single_host": _CONSERVATIVE_FLAGS,
        "multi_slice": _FULL_FLAGS,
    },
    "v5p": {
        "single_host": _FULL_FLAGS,
        "multi_slice": _FULL_FLAGS,
    },
    "v4": {
        "single_host": _FULL_FLAGS,
        "multi_slice": _FULL_FLAGS,
    },
}


def detect_tpu_type() -> str:
    """Detect TPU type from environment or system files.

    Checks in order:
    1. TPU_TYPE env var (e.g. "v6e-8", "v5p-16")
    2. TPU_ACCELERATOR_TYPE env var (GKE sets this)
    3. /sys/class/hwmon chip identification (future)

    Returns canonical type: "v6e", "v6p", "v5e", "v5p", "v4", or "unknown".
    """
    for env_var in ("TPU_TYPE", "TPU_ACCELERATOR_TYPE", "ACCELERATOR_TYPE"):
        val = os.environ.get(env_var, "")
        if val:
            return _parse_tpu_type(val)
    return "unknown"


def _parse_tpu_type(raw: str) -> str:
    """Parse a TPU type string to canonical form.

    Examples: "v6e-8" -> "v6e", "TPU v6 lite" -> "v6e", "v5p-16" -> "v5p"
    """
    raw_lower = raw.lower().strip()
    # Handle "TPU v6 lite" style (from jax.devices()[0].device_kind)
    if "v6" in raw_lower and "lite" in raw_lower:
        return "v6e"
    # Handle "vN[ep]" prefix patterns
    match = re.match(r"(v\d+[ep]?)", raw_lower)
    if match:
        return match.group(1)
    return "unknown"


def get_flags_for_device(tpu_type: str = "auto", num_slices: int = 1) -> str:
    """Get recommended LIBTPU_INIT_ARGS for the given TPU type.

    Args:
        tpu_type: TPU type ("v6e", "v6p", etc.) or "auto" to detect.
        num_slices: Number of TPU slices (>1 = multi-slice).

    Returns:
        Space-separated LIBTPU flag string.
    """
    if tpu_type == "auto":
        tpu_type = detect_tpu_type()

    preset = _DEVICE_PRESETS.get(tpu_type)
    if preset is None:
        logger.info("Unknown TPU type %r, using conservative flags", tpu_type)
        return _CONSERVATIVE_FLAGS

    if num_slices > 1:
        return preset["multi_slice"]
    return preset["single_host"]


def set_xla_flags(
    tpu_type: str = "auto",
    num_slices: int = 1,
    extra_flags: str = "",
) -> None:
    """Set LIBTPU_INIT_ARGS environment variable for TPU optimization.

    Must be called BEFORE jax.distributed.initialize() or any JAX backend init.

    Args:
        tpu_type: TPU type or "auto" to detect from environment.
        num_slices: Number of TPU slices.
        extra_flags: Additional flags to append (user overrides).
    """
    existing = os.environ.get("LIBTPU_INIT_ARGS", "")
    device_flags = get_flags_for_device(tpu_type, num_slices)

    all_flags = " ".join(filter(None, [existing, device_flags, extra_flags]))
    os.environ["LIBTPU_INIT_ARGS"] = all_flags
    logger.info("LIBTPU_INIT_ARGS set (tpu_type=%s, num_slices=%d): %s",
                tpu_type, num_slices, all_flags)
