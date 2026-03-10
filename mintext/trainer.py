"""Trainer: loss function, train step, eval step, and training loop."""

from __future__ import annotations

import logging
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from flax.training import train_state

from mintext.config import MinTextConfig
from mintext.models import Transformer, make_causal_mask
from mintext.optim.optimizer import create_optimizer, create_lr_schedule
from mintext.optim.grad_accumulation import accumulate_gradients
from mintext.distributed.sharding import add_data_axis_to_sharding

logger = logging.getLogger(__name__)

TrainState = train_state.TrainState


# --- Loss function ---


@jax.custom_vjp
def cross_entropy_with_z_loss(
    logits: jax.Array,
    targets: jax.Array,
    z_loss_weight: float,
) -> tuple[jax.Array, jax.Array]:
    """Cross-entropy with auxiliary z-loss for logit stabilization.

    The z-loss penalizes large log-partition values, improving training stability.
    Uses custom_vjp so z-loss affects gradients but is tracked separately.

    Args:
        logits: [batch, seq_len, vocab_size]
        targets: one-hot [batch, seq_len, vocab_size]
        z_loss_weight: Weight for z-loss penalty.

    Returns:
        (per_token_loss, per_token_z_loss)
    """
    logits = logits.astype(jnp.float32)  # Upcast for numerical stability
    logits_sum = jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
    log_softmax = logits - logits_sum
    loss = -jnp.sum(targets * log_softmax, axis=-1)
    log_z = jnp.squeeze(logits_sum, axis=-1)
    total_z_loss = z_loss_weight * jax.lax.square(log_z)
    loss = loss + total_z_loss
    return loss, total_z_loss


def _ce_z_fwd(logits, targets, z_loss_weight):
    logits = logits.astype(jnp.float32)
    max_logit = logits.max(axis=-1, keepdims=True)
    shifted = logits - max_logit
    exp_shifted = jnp.exp(shifted)
    sum_exp = jnp.sum(exp_shifted, axis=-1, keepdims=True)
    log_softmax = shifted - jnp.log(sum_exp)
    loss = -jnp.sum(targets * log_softmax, axis=-1)
    log_z = jnp.squeeze(jnp.log(sum_exp) + max_logit, axis=-1)
    total_z_loss = z_loss_weight * jax.lax.square(log_z)
    loss = loss + total_z_loss
    return (loss, total_z_loss), (logits, targets, z_loss_weight, exp_shifted, sum_exp, log_z)


def _ce_z_bwd(res, g):
    g = g[0]  # only backprop through loss, not z_loss
    logits, targets, z_loss_weight, exp_shifted, sum_exp, log_z = res
    deriv = jnp.expand_dims(1 + 2 * z_loss_weight * log_z, -1) * exp_shifted / sum_exp - targets
    g_logits = jnp.expand_dims(g, axis=-1) * deriv
    return (jnp.asarray(g_logits, logits.dtype), None, None)


cross_entropy_with_z_loss.defvjp(_ce_z_fwd, _ce_z_bwd)


@jax.custom_vjp
def _cross_entropy_with_integer_labels_and_z_loss(
    logits: jax.Array,
    targets: jax.Array,
    label_smoothing: float,
    z_loss_weight: float,
) -> tuple[jax.Array, jax.Array]:
    """Cross-entropy with z-loss using integer labels to avoid dense one-hot targets."""
    logits = logits.astype(jnp.float32)  # Upcast for numerical stability
    max_logit = logits.max(axis=-1, keepdims=True)
    shifted = logits - max_logit
    exp_shifted = jnp.exp(shifted)
    probs = exp_shifted / jnp.sum(exp_shifted, axis=-1, keepdims=True)
    logsumexp = jnp.log(jnp.sum(exp_shifted, axis=-1, keepdims=True)) + max_logit
    log_softmax = shifted - jnp.log(jnp.sum(exp_shifted, axis=-1, keepdims=True))
    target_log_probs = jnp.take_along_axis(log_softmax, targets[..., None], axis=-1).squeeze(-1)
    nll = -target_log_probs

    if label_smoothing > 0:
        smooth_loss = -jnp.mean(log_softmax, axis=-1)
        loss = (1.0 - label_smoothing) * nll + label_smoothing * smooth_loss
    else:
        loss = nll

    log_z = jnp.squeeze(logsumexp, axis=-1)
    total_z_loss = z_loss_weight * jax.lax.square(log_z)
    return loss + total_z_loss, total_z_loss


def _ce_int_z_fwd(logits, targets, label_smoothing, z_loss_weight):
    logits = logits.astype(jnp.float32)
    max_logit = logits.max(axis=-1, keepdims=True)
    shifted = logits - max_logit
    exp_shifted = jnp.exp(shifted)
    sum_exp = jnp.sum(exp_shifted, axis=-1, keepdims=True)
    probs = exp_shifted / sum_exp
    log_softmax = shifted - jnp.log(sum_exp)
    target_log_probs = jnp.take_along_axis(log_softmax, targets[..., None], axis=-1).squeeze(-1)
    nll = -target_log_probs

    if label_smoothing > 0:
        smooth_loss = -jnp.mean(log_softmax, axis=-1)
        loss = (1.0 - label_smoothing) * nll + label_smoothing * smooth_loss
    else:
        loss = nll

    log_z = jnp.squeeze(jnp.log(sum_exp) + max_logit, axis=-1)
    total_z_loss = z_loss_weight * jax.lax.square(log_z)
    return (loss + total_z_loss, total_z_loss), (
        probs,
        targets,
        label_smoothing,
        z_loss_weight,
        log_z,
    )


def _ce_int_z_bwd(res, g):
    g = g[0]  # only backprop through loss, not z_loss
    probs, targets, label_smoothing, z_loss_weight, log_z = res
    vocab_size = probs.shape[-1]

    deriv = probs * jnp.expand_dims(1.0 + 2.0 * z_loss_weight * log_z, axis=-1)
    if label_smoothing > 0:
        deriv = deriv - (label_smoothing / vocab_size)

    flat_deriv = deriv.reshape(-1, vocab_size)
    flat_targets = targets.reshape(-1)
    flat_deriv = flat_deriv.at[
        jnp.arange(flat_targets.shape[0]), flat_targets
    ].add(-(1.0 - label_smoothing))

    g_logits = jnp.expand_dims(g, axis=-1) * flat_deriv.reshape(probs.shape)
    return (jnp.asarray(g_logits, probs.dtype), None, None, None)


_cross_entropy_with_integer_labels_and_z_loss.defvjp(_ce_int_z_fwd, _ce_int_z_bwd)


def cross_entropy_loss(
    logits: jax.Array,
    targets: jax.Array,
    label_smoothing: float = 0.0,
    z_loss_weight: float = 0.0,
) -> tuple[jax.Array, jax.Array]:
    """Compute cross-entropy loss per token, optionally with z-loss.

    Args:
        logits: [batch, seq_len, vocab_size]
        targets: [batch, seq_len] integer token ids
        label_smoothing: Label smoothing coefficient (0 = no smoothing).
        z_loss_weight: Z-loss weight (0 = disabled).

    Returns:
        (per_token_loss, per_token_z_loss). z_loss is zeros if weight is 0.
    """
    if z_loss_weight > 0:
        return _cross_entropy_with_integer_labels_and_z_loss(
            logits, targets, label_smoothing, z_loss_weight
        )

    # Standard cross-entropy (no z-loss)
    logits = logits.astype(jnp.float32)  # Upcast for numerical stability
    log_softmax = jax.nn.log_softmax(logits, axis=-1)
    target_log_probs = jnp.take_along_axis(log_softmax, targets[..., None], axis=-1).squeeze(-1)
    per_token_loss = -target_log_probs
    if label_smoothing > 0:
        per_token_loss = (
            (1.0 - label_smoothing) * per_token_loss
            - label_smoothing * jnp.mean(log_softmax, axis=-1)
        )
    return per_token_loss, jnp.zeros_like(per_token_loss)


def chunked_cross_entropy_loss(
    hidden_states: jax.Array,
    targets: jax.Array,
    output_proj: nnx.Module,
    num_tiles: int,
    label_smoothing: float = 0.0,
    z_loss_weight: float = 0.0,
) -> tuple[jax.Array, jax.Array]:
    """Compute cross-entropy in chunks to reduce peak memory from logits.

    Instead of materializing [batch*seq, vocab_size] logits all at once,
    processes num_tiles chunks of [batch*seq/num_tiles, vocab_size] each.

    Args:
        hidden_states: [batch*seq, hidden_size] flattened hidden states.
        targets: [batch*seq] flattened target token ids.
        output_proj: Final output projection module (hidden -> vocab).
        num_tiles: Number of chunks to split over.
        label_smoothing: Label smoothing coefficient.
        z_loss_weight: Z-loss weight.

    Returns:
        (per_token_loss, per_token_z_loss) each of shape [batch*seq].
    """
    total_tokens = hidden_states.shape[0]
    chunk_size = total_tokens // num_tiles

    hidden_chunks = hidden_states.reshape(num_tiles, chunk_size, -1)
    target_chunks = targets.reshape(num_tiles, chunk_size)

    def compute_chunk_loss(carry, chunk_data):
        h_chunk, t_chunk = chunk_data
        logits_chunk = output_proj(h_chunk)  # [chunk_size, vocab_size]
        # cross_entropy_loss expects [batch, seq, vocab] so add batch dim
        loss_chunk, z_loss_chunk = cross_entropy_loss(
            logits_chunk[jnp.newaxis], t_chunk[jnp.newaxis],
            label_smoothing, z_loss_weight,
        )
        return carry, (loss_chunk.squeeze(0), z_loss_chunk.squeeze(0))

    _, (all_losses, all_z_losses) = jax.lax.scan(
        compute_chunk_loss, None, (hidden_chunks, target_chunks),
    )

    return all_losses.reshape(-1), all_z_losses.reshape(-1)


def compute_loss(
    params: Any,
    model: Transformer,
    config: MinTextConfig,
    batch: dict[str, jax.Array],
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Compute training loss.

    Args:
        params: Model parameter pytree (from nnx.split).
        model: NNX model (graph structure, used with nnx.merge).
        config: MinText config.
        batch: Dict with 'input_tokens' [B, S] and 'target_tokens' [B, S].

    Returns:
        (scalar_loss, aux_dict)
    """
    graph, _ = nnx.split(model)
    merged_model = nnx.merge(graph, params)

    tokens = batch["input_tokens"]
    targets = batch["target_tokens"]
    batch_size, seq_len = tokens.shape

    positions = jnp.broadcast_to(jnp.arange(seq_len), (batch_size, seq_len))
    # Only generate the explicit causal mask when needed (soft-capping path).
    # jax.nn.dot_product_attention handles causal masking via is_causal=True,
    # avoiding materialization of the O(S²) mask tensor.
    if config.attn_logit_softcapping is not None:
        mask = make_causal_mask(seq_len, dtype=getattr(jnp, config.dtype))
    else:
        mask = None

    if config.num_vocab_tiles > 1:
        hidden_states, all_aux = merged_model(tokens, positions, mask, return_hidden=True)
        hidden_flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        targets_flat = targets.reshape(-1)
        per_token_loss, per_token_z_loss = chunked_cross_entropy_loss(
            hidden_flat, targets_flat, merged_model.output_proj,
            config.num_vocab_tiles, config.label_smoothing, config.z_loss_weight,
        )
        per_token_loss = per_token_loss.reshape(batch_size, seq_len)
        per_token_z_loss = per_token_z_loss.reshape(batch_size, seq_len)
    else:
        logits, all_aux = merged_model(tokens, positions, mask)
        per_token_loss, per_token_z_loss = cross_entropy_loss(
            logits, targets, config.label_smoothing, config.z_loss_weight,
        )

    # Average over all tokens
    loss = jnp.mean(per_token_loss)
    z_loss = jnp.mean(per_token_z_loss)

    # Compute per-layer expert counts for loss-free bias updates (inside JIT)
    moe_expert_counts = None
    if config.num_experts > 0:
        counts_list = []
        for layer_aux in all_aux:
            if layer_aux is not None:
                topk_indices = layer_aux["topk_indices"]  # [N, K]
                counts_list.append(
                    jnp.bincount(topk_indices.reshape(-1), length=config.num_experts)
                )
        if counts_list:
            moe_expert_counts = jnp.stack(counts_list)  # [num_moe_layers, num_experts]

    aux = {
        "loss": loss,
        "z_loss": z_loss,
        "per_token_loss": per_token_loss,
        "moe_expert_counts": moe_expert_counts,
    }
    return loss, aux


# --- Train step ---


def train_step(
    state: TrainState,
    batch: dict[str, jax.Array],
    model: Transformer,
    config: MinTextConfig,
) -> tuple[TrainState, dict[str, jax.Array]]:
    """Single training step: forward, backward, optimizer update.

    Args:
        state: Current training state.
        batch: Training batch.
        model: NNX model (graph structure only, params come from state).
        config: MinText config.

    Returns:
        (new_state, metrics_dict)
    """
    loss_fn = partial(compute_loss, model=model, config=config, batch=batch)

    if config.gradient_accumulation_steps > 1:
        loss, aux, grads = accumulate_gradients(
            loss_fn=lambda params, data: compute_loss(params, model, config, data),
            params=state.params,
            data=batch,
            num_micro_steps=config.gradient_accumulation_steps,
        )
    else:
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, aux), grads = grad_fn(state.params)

    # Compute gradient norm before clipping (for logging)
    grad_norm = _l2_norm(grads)

    new_state = state.apply_gradients(grads=grads)

    # Get current learning rate
    lr = create_lr_schedule(config)(state.step)

    metrics = {
        "loss": loss,
        "z_loss": aux.get("z_loss", jnp.float32(0.0)),
        "learning_rate": lr,
        "grad_norm": grad_norm,
        "param_norm": _l2_norm(state.params),
        "moe_expert_counts": aux.get("moe_expert_counts"),
    }
    return new_state, metrics


def update_expert_biases(
    state: TrainState,
    expert_counts: jax.Array,
    config: MinTextConfig,
) -> TrainState:
    """Update e_score_correction_bias in state params based on expert load.

    Applies loss-free load balancing: experts that are under-utilized get a
    positive bias bump, over-utilized ones get a negative bump.

    Called outside JIT after train_step returns expert_counts.

    Args:
        state: Current training state with params to update.
        expert_counts: [num_moe_layers, num_experts] from JIT-compiled train_step.
        config: MinText config.

    Returns:
        New TrainState with updated bias values.
    """
    rate = config.e_score_correction_bias_update_rate

    # Compute per-layer bias deltas: [num_moe_layers, num_experts]
    avg_load = expert_counts.sum(axis=-1, keepdims=True) / config.num_experts
    direction = jnp.sign(avg_load - expert_counts)
    bias_deltas = direction * rate

    # Apply updates to matching entries in state.params via tree traversal
    moe_idx = [0]

    def _update_leaf(path, value):
        for p in path:
            if "e_score_correction_bias" in str(p):
                idx = moe_idx[0]
                moe_idx[0] += 1
                return value + bias_deltas[idx]
        return value

    new_params = jax.tree.map_with_path(_update_leaf, state.params)
    return state.replace(params=new_params)


def eval_step(
    state: TrainState,
    batch: dict[str, jax.Array],
    model: Transformer,
    config: MinTextConfig,
) -> dict[str, jax.Array]:
    """Single evaluation step: forward only, no gradients.

    Args:
        state: Current training state.
        batch: Eval batch.
        model: NNX model.
        config: MinText config.

    Returns:
        Metrics dict.
    """
    loss, aux = compute_loss(state.params, model, config, batch)
    return {"eval_loss": loss}


# --- Utilities ---


def _l2_norm(tree: Any) -> jax.Array:
    """Compute L2 norm of a pytree."""
    return optax.tree.norm(tree)


def create_train_state(
    model: Transformer,
    config: MinTextConfig,
) -> TrainState:
    """Create initial training state from model and config.

    When shard_optimizer_over_data=True, optimizer state is sharded over
    the data axis (ZeRO-1), reducing per-device optimizer memory by ~N×
    for N-way data parallelism.

    Args:
        model: Initialized NNX model.
        config: MinText config.

    Returns:
        TrainState with params and optimizer.
    """
    _, params = nnx.split(model)
    tx = create_optimizer(config)
    state = TrainState.create(
        apply_fn=None,
        params=params,
        tx=tx,
    )

    if config.shard_optimizer_over_data:
        # ZeRO-1: shard optimizer state over data axis
        from jax.sharding import NamedSharding

        def _maybe_add_data_axis(leaf):
            if isinstance(leaf, NamedSharding):
                return add_data_axis_to_sharding(leaf)
            return leaf

        opt_shardings = jax.tree.map(
            _maybe_add_data_axis,
            state.opt_state,
            is_leaf=lambda x: isinstance(x, NamedSharding),
        )
        state = state.replace(
            opt_state=jax.tree.map(
                lambda arr, sharding: jax.device_put(arr, sharding)
                if isinstance(sharding, NamedSharding)
                else arr,
                state.opt_state,
                opt_shardings,
                is_leaf=lambda x: isinstance(x, NamedSharding),
            ),
        )

    return state
