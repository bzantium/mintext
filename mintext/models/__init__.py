"""MinText model components."""

from __future__ import annotations

from typing import Any

from mintext.models.base import Transformer, make_causal_mask, DecoderLayer, _all_layers_scannable
from mintext.models import llama3, qwen3, deepseek_v3, qwen3_next, gemma3

__all__ = ["Transformer", "make_causal_mask", "DecoderLayer", "_all_layers_scannable", "get_layer_types"]

_MODEL_REGISTRY = {
    "llama3": llama3,
    "qwen3": qwen3,
    "deepseek_v3": deepseek_v3,
    "qwen3_next": qwen3_next,
    "gemma3": gemma3,
}


def get_layer_types(
    model_type: str, num_hidden_layers: int, data: dict[str, Any]
) -> list[str]:
    """Dispatch layer type generation to the appropriate model module."""
    module = _MODEL_REGISTRY.get(model_type)
    if module is None:
        raise ValueError(
            f"Unknown model_type: {model_type!r}. "
            f"Available: {sorted(_MODEL_REGISTRY.keys())}"
        )
    return module.get_layer_types(num_hidden_layers, data)
