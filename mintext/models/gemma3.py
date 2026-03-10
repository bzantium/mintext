"""Gemma3 layer type generation."""

from __future__ import annotations

from typing import Any


def get_layer_types(num_hidden_layers: int, data: dict[str, Any]) -> list[str]:
    """Interleaved sliding + global pattern (5:1 by default).

    Every ``sliding_window_pattern``-th layer (1-indexed) is global (full) attention;
    the rest use sliding window attention.
    """
    pattern = data.get("sliding_window_pattern", 6)
    return [
        "full_attention" if (i + 1) % pattern == 0 else "sliding_attention"
        for i in range(num_hidden_layers)
    ]
