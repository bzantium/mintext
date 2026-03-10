"""Optimizer factory and learning rate schedules."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
from optax.contrib._muon import MuonDimensionNumbers, muon

from mintext.config import MinTextConfig


def create_lr_schedule(config: MinTextConfig) -> optax.Schedule:
    """Create a learning rate schedule.

    Supported: cosine, linear, wsd (warmup-stable-decay).

    Args:
        config: MinText config with lr_schedule, warmup_steps_fraction, etc.

    Returns:
        Optax schedule function: step -> learning_rate.
    """
    total_steps = max(config.steps, 1)
    warmup_steps = max(int(total_steps * config.warmup_steps_fraction), 1)
    peak_lr = config.learning_rate
    final_lr = peak_lr * config.lr_final_fraction

    if config.lr_schedule == "cosine":
        decay_steps = total_steps - warmup_steps
        if decay_steps > 0:
            schedule = optax.join_schedules(
                schedules=[
                    optax.linear_schedule(0.0, peak_lr, warmup_steps),
                    optax.cosine_decay_schedule(peak_lr, decay_steps, alpha=final_lr / peak_lr if peak_lr > 0 else 0.0),
                ],
                boundaries=[warmup_steps],
            )
        else:
            schedule = optax.linear_schedule(0.0, peak_lr, warmup_steps)

    elif config.lr_schedule == "linear":
        decay_steps = total_steps - warmup_steps
        if decay_steps > 0:
            schedule = optax.join_schedules(
                schedules=[
                    optax.linear_schedule(0.0, peak_lr, warmup_steps),
                    optax.linear_schedule(peak_lr, final_lr, decay_steps),
                ],
                boundaries=[warmup_steps],
            )
        else:
            schedule = optax.linear_schedule(0.0, peak_lr, warmup_steps)

    elif config.lr_schedule == "wsd":
        decay_steps = max(int(total_steps * config.wsd_decay_steps_fraction), 1)
        stable_steps = total_steps - warmup_steps - decay_steps

        schedules = [optax.linear_schedule(0.0, peak_lr, warmup_steps)]
        boundaries = [warmup_steps]

        if stable_steps > 0:
            schedules.append(optax.constant_schedule(peak_lr))
            boundaries.append(warmup_steps + stable_steps)

        if config.wsd_decay_style == "cosine":
            schedules.append(optax.cosine_decay_schedule(
                peak_lr, decay_steps,
                alpha=final_lr / peak_lr if peak_lr > 0 else 0.0,
            ))
        else:
            schedules.append(optax.linear_schedule(peak_lr, final_lr, decay_steps))

        schedule = optax.join_schedules(schedules, boundaries)

    else:
        raise ValueError(f"Unknown lr_schedule: {config.lr_schedule}")

    return schedule


def _wd_mask(params):
    """Weight decay mask: exclude e_score_correction_bias."""
    def _should_decay(path, _):
        for p in path:
            if "e_score_correction_bias" in str(p):
                return False
        return True
    return jax.tree.map_with_path(_should_decay, params)


def _muon_dimension_numbers(params):
    """Map param pytree to MuonDimensionNumbers for Muon optimizer."""
    def _classify(path, param):
        path_str = ".".join(str(p) for p in path)
        # Exclude: 1D (norms, biases), embeddings, scalars, MoE bias
        if param.ndim <= 1:
            return None
        if "embedder" in path_str or "e_score_correction_bias" in path_str:
            return None
        if "scale" in path_str:
            return None
        # 3D attention kernels
        if param.ndim == 3:
            if any(k in path_str for k in ("query", "key", "value", "q_proj", "q_a_proj", "q_b_proj", "kv_a_proj", "kv_b_proj")):
                return MuonDimensionNumbers(0, (1, 2))
            if "out" in path_str:
                return MuonDimensionNumbers((0, 1), 2)
            if "expert" in path_str:
                return MuonDimensionNumbers(1, 2)
            return MuonDimensionNumbers(0, (1, 2))
        # Standard 2D
        return MuonDimensionNumbers(0, 1)
    return jax.tree.map_with_path(_classify, params)


def create_optimizer(config: MinTextConfig) -> optax.GradientTransformation:
    """Create the optimizer chain.

    Supports AdamW and Muon optimizers with gradient clipping.

    Args:
        config: MinText config.

    Returns:
        Optax optimizer transformation.
    """
    lr_schedule = create_lr_schedule(config)

    if config.optimizer == "muon":
        components = []
        if config.gradient_clip_threshold > 0:
            components.append(optax.clip_by_global_norm(config.gradient_clip_threshold))
        muon_kwargs = dict(
            learning_rate=lr_schedule,
            ns_steps=config.muon_newton_schulz_steps,
            beta=config.muon_beta,
            eps=config.adam_eps,
            adam_b1=config.adam_b1,
            adam_b2=config.adam_b2,
            weight_decay=config.weight_decay,
            weight_decay_mask=_wd_mask,
            muon_weight_dimension_numbers=_muon_dimension_numbers,
        )
        if config.muon_consistent_rms is not None:
            import inspect
            if "consistent_rms" in inspect.signature(muon).parameters:
                muon_kwargs["consistent_rms"] = config.muon_consistent_rms
        components.append(muon(**muon_kwargs))
        return optax.chain(*components)

    # Default: AdamW
    components = []
    if config.gradient_clip_threshold > 0:
        components.append(optax.clip_by_global_norm(config.gradient_clip_threshold))

    components.append(
        optax.adamw(
            learning_rate=lr_schedule,
            b1=config.adam_b1,
            b2=config.adam_b2,
            eps=config.adam_eps,
            weight_decay=config.weight_decay,
            mask=_wd_mask,
        )
    )

    return optax.chain(*components)
