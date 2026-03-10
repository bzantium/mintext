"""Qwen3-Next layer type generation."""

from __future__ import annotations

from typing import Any


def get_layer_types(num_hidden_layers: int, data: dict[str, Any]) -> list[str]:
    """Hybrid: full attention every N-th layer, linear attention otherwise."""
    interval = data.get("full_attention_interval", 4)
    return [
        "full_attention" if (i + 1) % interval == 0 else "linear_attention"
        for i in range(num_hidden_layers)
    ]
