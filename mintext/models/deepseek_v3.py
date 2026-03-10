"""DeepSeek-V3 layer type generation."""

from __future__ import annotations

from typing import Any


def get_layer_types(num_hidden_layers: int, data: dict[str, Any]) -> list[str]:
    """All layers use full attention."""
    return ["full_attention"] * num_hidden_layers
