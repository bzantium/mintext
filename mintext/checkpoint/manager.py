"""Orbax checkpoint manager for MinText.

Handles save/restore of training state (model params, optimizer state, step)
with support for async checkpointing and configurable retention.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import jax
import numpy as np
import orbax.checkpoint as ocp

from mintext.config import MinTextConfig

logger = logging.getLogger(__name__)


def create_checkpoint_manager(config: MinTextConfig) -> ocp.CheckpointManager:
    """Create an Orbax CheckpointManager from config.

    Args:
        config: MinText config with checkpointing settings.

    Returns:
        Configured CheckpointManager.
    """
    ckpt_dir = Path(config.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    options = ocp.CheckpointManagerOptions(
        save_interval_steps=config.checkpoint_period,
        max_to_keep=config.max_checkpoints,
        create=True,
        enable_async_checkpointing=config.async_checkpointing,
    )

    manager = ocp.CheckpointManager(
        directory=ckpt_dir,
        options=options,
    )

    logger.info(
        "Checkpoint manager: dir=%s, period=%d, max_keep=%d, async=%s",
        ckpt_dir, config.checkpoint_period, config.max_checkpoints,
        config.async_checkpointing,
    )
    return manager


def save_checkpoint(
    manager: ocp.CheckpointManager,
    step: int,
    state: Any,
    force: bool = False,
) -> bool:
    """Save a checkpoint if the step matches the save interval (or force=True).

    Args:
        manager: Orbax checkpoint manager.
        step: Current training step.
        state: TrainState pytree to save.
        force: Force save regardless of interval.

    Returns:
        True if a checkpoint was saved.
    """
    saved = manager.save(step, args=ocp.args.StandardSave(state), force=force)
    if saved:
        logger.info("Checkpoint saved at step %d", step)
    return saved


def restore_checkpoint(
    manager: ocp.CheckpointManager,
    state: Any,
    step: int | None = None,
) -> tuple[Any, int]:
    """Restore training state from the latest (or specified) checkpoint.

    Args:
        manager: Orbax checkpoint manager.
        state: Abstract TrainState (same structure as saved, used for restore shape/dtype).
        step: Specific step to restore. If None, restores latest.

    Returns:
        (restored_state, step) tuple.
    """
    if step is None:
        step = manager.latest_step()

    if step is None:
        raise FileNotFoundError("No checkpoints found")

    restored = manager.restore(step, args=ocp.args.StandardRestore(state))
    logger.info("Checkpoint restored from step %d", step)
    return restored, step


def maybe_restore_checkpoint(
    manager: ocp.CheckpointManager | None,
    state: Any,
    config: MinTextConfig,
    model: Any = None,
) -> tuple[Any, int]:
    """Restore from checkpoint if available, otherwise return initial state at step 0.

    Priority:
        1. Resume from checkpoint_dir (existing training run)
        2. load_checkpoint (external Orbax path — full state or params-only)
        3. load_hf_checkpoint (HF SafeTensors → params, reset optimizer + step)
        4. Start from scratch

    Args:
        manager: Orbax checkpoint manager (may be None if checkpointing disabled).
        state: Initial TrainState.
        config: MinText config.
        model: NNX model instance (required for load_hf_checkpoint).

    Returns:
        (state, start_step) tuple.
    """
    # 1. Try resuming from checkpoint_dir (takes priority — no re-import on resume)
    if manager is not None:
        latest = manager.latest_step()
        if latest is not None:
            restored = manager.restore(latest, args=ocp.args.StandardRestore(state))
            start_step = int(_get_step(restored))
            logger.info("Resumed from checkpoint at step %d", start_step)
            return restored, start_step

    # 2. External Orbax checkpoint
    if config.load_checkpoint:
        ext_dir = Path(config.load_checkpoint)
        if not ext_dir.exists():
            raise FileNotFoundError(f"load_checkpoint path not found: {ext_dir}")
        ext_manager = ocp.CheckpointManager(
            directory=ext_dir, options=ocp.CheckpointManagerOptions(read_only=True)
        )
        ext_step = ext_manager.latest_step()
        if ext_step is None:
            raise FileNotFoundError(f"No checkpoints in: {ext_dir}")
        restored = ext_manager.restore(ext_step, args=ocp.args.StandardRestore(state))

        if config.load_params_only:
            # Fine-tuning: take params, reset optimizer + step
            state = state.replace(params=restored.params, step=0)
            logger.info("Loaded params from %s (optimizer reset, step=0)", ext_dir)
            return state, 0
        else:
            # Full state restore (params + optimizer + step)
            start_step = int(_get_step(restored))
            logger.info("Loaded full state from %s at step %d", ext_dir, start_step)
            return restored, start_step

    # 3. HF checkpoint import (convert HF → MinText params, reset optimizer + step)
    if config.load_hf_checkpoint:
        if model is None:
            raise ValueError("model required for HF checkpoint loading")
        from flax import nnx
        from mintext.checkpoint.conversion import load_hf_checkpoint
        model = load_hf_checkpoint(config.load_hf_checkpoint, config, model)
        _graph, new_state = nnx.split(model)
        state = state.replace(params=new_state, step=0)
        logger.info("Loaded HF params from %s (optimizer reset, step=0)",
                     config.load_hf_checkpoint)
        return state, 0

    # 4. No checkpoint found — start from scratch
    return state, 0


def _get_step(state: Any) -> int:
    """Extract step from a TrainState (works with both pytree dict and TrainState object)."""
    if hasattr(state, "step"):
        step = state.step
    elif isinstance(state, dict) and "step" in state:
        step = state["step"]
    else:
        return 0
    # Handle JAX arrays
    if hasattr(step, "item"):
        return int(step.item())
    return int(step)


def wait_for_checkpoint(manager: ocp.CheckpointManager | None) -> None:
    """Wait for any in-progress async checkpoints to complete."""
    if manager is not None:
        manager.wait_until_finished()
        logger.info("All pending checkpoints complete")
