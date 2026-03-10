"""Training loop entry point."""

from __future__ import annotations

# Set XLA/LIBTPU flags BEFORE any JAX initialization.
# Per-device-type flag presets auto-detect TPU type from environment.
# Config-level overrides via MINTEXT_TPU_TYPE and MINTEXT_XLA_FLAGS env vars.
import os as _os
from mintext.utils.xla_flags import set_xla_flags as _set_xla_flags
_set_xla_flags(
    tpu_type=_os.environ.get("MINTEXT_TPU_TYPE", "auto"),
    num_slices=int(_os.environ.get("MINTEXT_NUM_SLICES", "1")),
    extra_flags=_os.environ.get("MINTEXT_XLA_FLAGS", ""),
)

# On TPU, jax.distributed.initialize() must be called before ANY JAX backend
# initialization (including jax.devices()). Try unconditionally; on GPU single-node
# this will fail harmlessly (no coordinator) and we handle GPU multi-node later.
import jax
try:
    if not jax.distributed.is_initialized():
        jax.distributed.initialize()
except Exception:
    pass

import logging
import sys
import time
from functools import partial
from pathlib import Path

import jax.numpy as jnp

from mintext.config import load_config, print_help_config, MinTextConfig
from mintext.distributed.mesh import setup_mesh
from mintext.distributed.partition import create_sharded_model
from mintext.distributed.sharding import get_input_data_sharding
from mintext.trainer import create_train_state, train_step, eval_step, update_expert_biases

logger = logging.getLogger(__name__)


def _apply_moe_tiling(model, gate_up_tiling, down_tiling):
    """Apply autotuned tiling to all MoEExperts modules in the model."""
    from flax import nnx
    from mintext.modules.moe import MoEExperts
    for _path, module in nnx.iter_modules(model):
        if isinstance(module, MoEExperts):
            if gate_up_tiling is not None:
                module.gate_up_tiling = gate_up_tiling
            if down_tiling is not None:
                module.down_tiling = down_tiling


def _make_synthetic_batch(
    config: MinTextConfig, rng: jax.Array, mesh: jax.sharding.Mesh,
) -> dict[str, jax.Array]:
    """Generate a synthetic training batch for testing."""
    data_parallelism = mesh.shape["data"]
    batch_size = config.per_device_batch_size * data_parallelism
    seq_len = config.seq_length
    tokens = jax.random.randint(rng, (batch_size, seq_len), 0, config.vocab_size)
    return {
        "input_tokens": tokens,
        "target_tokens": tokens,  # Self-supervised: predict same tokens shifted
    }


def train(config: MinTextConfig):
    """Run the training loop.

    Args:
        config: Validated MinText config.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    # Initialize distributed runtime (multi-node).
    # On TPU, this was already done at module import time (before backend init).
    # On GPU, this handles multi-node SLURM setup.
    from mintext.distributed.mesh import initialize_distributed
    if not jax.distributed.is_initialized():
        initialize_distributed(timeout=config.jax_distributed_init_timeout)

    logger.info("Starting training: %s (%d steps)", config.run_name, config.steps)
    logger.info("Platform: %s, devices: %d", jax.devices()[0].platform, jax.device_count())

    # Setup mesh
    mesh = setup_mesh(config)

    # Create model
    model = create_sharded_model(config, mesh)

    # Create train state
    state = create_train_state(model, config)

    # Setup logging and profiling
    from mintext.utils.logging import MetricLogger
    from mintext.utils.profiling import Profiler

    metric_logger = MetricLogger(config, params=state.params)
    profiler = Profiler(config)

    # Setup checkpointing
    checkpoint_manager = None
    start_step = 0
    if config.enable_checkpointing:
        from mintext.checkpoint.manager import (
            create_checkpoint_manager,
            maybe_restore_checkpoint,
            save_checkpoint,
            wait_for_checkpoint,
        )
        checkpoint_manager = create_checkpoint_manager(config)
        state, start_step = maybe_restore_checkpoint(checkpoint_manager, state, config, model=model)
        if start_step > 0:
            logger.info("Resuming from step %d", start_step)

    # MoE autotuning (before JIT compilation)
    if config.moe_autotune and config.num_experts > 0:
        from mintext.kernels.autotuner import autotune_moe, MoETuningConfig
        tune_cfg = MoETuningConfig(
            num_experts=config.num_experts,
            hidden_size=config.hidden_size,
            moe_intermediate_size=config.moe_intermediate_size,
            num_experts_per_tok=config.num_experts_per_tok,
            batch_seq_tokens=config.per_device_batch_size * config.seq_length * jax.device_count(),
            dtype=config.dtype,
        )
        result = autotune_moe(tune_cfg, cache_dir=config.moe_autotune_cache_dir or None)
        # Apply tuned tiling to model experts (mutate NNX module attributes)
        logger.info("MoE autotuner: gate_up_tiling=%s, down_tiling=%s",
                     result.gate_up_tiling, result.down_tiling)
        _apply_moe_tiling(model, result.gate_up_tiling, result.down_tiling)

    # Data sharding for input batches
    data_sharding = get_input_data_sharding(config, mesh)

    # Derive state shardings from the actual state pytree for explicit JIT shardings.
    # This mirrors MaxText's approach: explicit in_shardings/out_shardings tells the
    # XLA compiler exactly how state is sharded, avoiding inference and enabling
    # better compute/communication overlap.
    state_shardings = jax.tree.map(
        lambda x: x.sharding if hasattr(x, 'sharding') else None,
        state,
    )

    # JIT-compile train step with explicit shardings
    jit_train_step = jax.jit(
        partial(train_step, model=model, config=config),
        in_shardings=(state_shardings, data_sharding),
        out_shardings=(state_shardings, None),
        donate_argnums=0,
    )

    # Data pipeline
    train_iter = None
    if config.dataset_type != "synthetic" and config.data_path:
        from mintext.data.pipeline import create_train_iterator
        train_iter = create_train_iterator(config, mesh)
        logger.info("Using %s data pipeline: %s", config.dataset_type, config.data_path)
    else:
        logger.info("Using synthetic data pipeline")

    # Training loop
    rng = jax.random.key(config.seed)

    for step in range(start_step, config.steps):
        profiler.maybe_activate(step)
        step_start = time.time()

        # Get batch and shard across devices
        if train_iter is not None:
            batch = next(train_iter)
        else:
            rng, batch_rng = jax.random.split(rng)
            batch = _make_synthetic_batch(config, batch_rng, mesh)
        batch = jax.tree.map(lambda x: jax.device_put(x, data_sharding), batch)

        # Train step (JIT-compiled)
        state, metrics = jit_train_step(state, batch)

        # Sync on loss scalar for accurate timing (lightweight vs full state sync)
        metrics["loss"].block_until_ready()

        step_time = time.time() - step_start

        # Loss-free expert bias update (outside JIT, excluded from step timing)
        if config.num_experts > 0 and metrics.get("moe_expert_counts") is not None:
            state = update_expert_biases(state, metrics["moe_expert_counts"], config)

        # Log
        if step % config.log_period == 0 or step == config.steps - 1:
            metric_logger.log_step(step, metrics, step_time)

        # Checkpoint
        if checkpoint_manager is not None:
            save_checkpoint(checkpoint_manager, step, state)

        profiler.maybe_deactivate(step)

    # Final checkpoint (force save at end if not already saved)
    if checkpoint_manager is not None:
        if start_step < config.steps and config.steps not in checkpoint_manager.all_steps():
            save_checkpoint(checkpoint_manager, config.steps, state, force=True)
        wait_for_checkpoint(checkpoint_manager)

    profiler.stop()
    metric_logger.close()
    logger.info("Training complete.")


def main():
    """CLI entry point.

    Usage:
        python -m mintext.train --config configs/base.yml steps=100 learning_rate=1e-4
        python -m mintext.train --help-config   # show all config fields
    """
    import argparse

    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser(description="MinText training")
    parser.add_argument("--config", type=str, default="configs/base.yml", help="Config YAML path")
    parser.add_argument("--help-config", action="store_true", help="Show all config fields and exit")
    args, unknown = parser.parse_known_args()

    if args.help_config:
        print_help_config()
        return

    # Use OmegaConf.from_cli() for proper type handling (lists, nested, int/float/bool)
    cli_cfg = OmegaConf.from_cli(unknown)
    overrides = OmegaConf.to_container(cli_cfg, resolve=True) if cli_cfg else {}

    config = load_config(args.config, overrides=overrides if overrides else None)
    train(config)


if __name__ == "__main__":
    main()
