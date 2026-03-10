"""Qwen3 layer type generation."""

from __future__ import annotations

from typing import Any


def get_layer_types(num_hidden_layers: int, data: dict[str, Any]) -> list[str]:
    """Sliding window for lower layers, full attention for upper layers."""
    max_window = data.get("max_window_layers", 28)
    use_sw = data.get("use_sliding_window", False)
    if use_sw:
        return [
            "sliding_attention" if i < max_window else "full_attention"
            for i in range(num_hidden_layers)
        ]
    return ["full_attention"] * num_hidden_layers
