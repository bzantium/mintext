"""Gradient accumulation via lax.scan over micro-batches."""

from __future__ import annotations

from typing import Any, Callable

import jax


def accumulate_gradients(
    loss_fn: Callable,
    params: Any,
    data: dict[str, jax.Array],
    num_micro_steps: int,
) -> tuple[jax.Array, Any, Any]:
    """Accumulate gradients over micro-batches using lax.scan.

    Splits the batch dimension into micro-batches, computes gradients for each,
    and averages them. Returns the mean loss, aux data from last micro-batch,
    and accumulated gradients.

    Args:
        loss_fn: Function (params, data) -> (loss, aux). Loss should be
            per-token mean within the micro-batch.
        params: Model parameters.
        data: Dict with arrays of shape [batch, seq_len, ...].
        num_micro_steps: Number of micro-batches to split into.

    Returns:
        (mean_loss, last_aux, mean_grads)
    """
    batch_size = jax.tree.leaves(data)[0].shape[0]
    if batch_size % num_micro_steps != 0:
        raise ValueError(
            f"Batch size {batch_size} must be divisible by num_micro_steps={num_micro_steps}"
        )
    micro_batch_size = batch_size // num_micro_steps

    # Reshape data: [batch, ...] -> [num_micro_steps, micro_batch, ...]
    micro_data = jax.tree.map(
        lambda x: x.reshape((num_micro_steps, micro_batch_size, *x.shape[1:])),
        data,
    )

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    first_micro_batch = jax.tree.map(lambda x: x[0], micro_data)
    (first_loss, first_aux), first_grads = grad_fn(params, first_micro_batch)

    if num_micro_steps == 1:
        return first_loss, first_aux, first_grads

    def scan_step(carry, micro_batch):
        acc_grad, acc_loss, _last_aux = carry
        (loss, aux), grads = grad_fn(params, micro_batch)
        acc_grad = jax.tree.map(lambda a, g: a + g, acc_grad, grads)
        acc_loss = acc_loss + loss
        return (acc_grad, acc_loss, aux), None

    remaining_micro_data = jax.tree.map(lambda x: x[1:], micro_data)
    (total_grads, total_loss, last_aux), _ = jax.lax.scan(
        scan_step,
        (first_grads, first_loss, first_aux),
        remaining_micro_data,
    )

    # Average
    mean_grads = jax.tree.map(lambda g: g / num_micro_steps, total_grads)
    mean_loss = total_loss / num_micro_steps

    return mean_loss, last_aux, mean_grads
