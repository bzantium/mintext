"""MinText configuration system.

Single flat Pydantic config with YAML loading via OmegaConf.
CLI overrides use OmegaConf.from_cli() for proper type handling.
Environment variable overrides use MINTEXT_<KEY> prefix.
"""

from __future__ import annotations

import enum
import logging
import os
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

logger = logging.getLogger(__name__)

_ENV_PREFIX = "MINTEXT_"


# --- Enums ---


class DType(str, enum.Enum):
    bfloat16 = "bfloat16"
    float32 = "float32"
    float16 = "float16"


class OptimizerType(str, enum.Enum):
    adamw = "adamw"
    muon = "muon"


class LRScheduleType(str, enum.Enum):
    cosine = "cosine"
    linear = "linear"
    wsd = "wsd"


class RematPolicy(str, enum.Enum):
    full = "full"
    minimal = "minimal"
    none = "none"
    save_qkv_proj = "save_qkv_proj"
    save_dot_except_mlp = "save_dot_except_mlp"
    qkv_proj_offloaded = "qkv_proj_offloaded"
    minimal_offloaded = "minimal_offloaded"


class DatasetType(str, enum.Enum):
    synthetic = "synthetic"
    mmap = "mmap"
    arecord = "arecord"
    auto = "auto"


class ModelType(str, enum.Enum):
    llama3 = "llama3"
    qwen3 = "qwen3"
    deepseek_v3 = "deepseek_v3"
    qwen3_next = "qwen3_next"
    gemma3 = "gemma3"


class RopeType(str, enum.Enum):
    default = "default"
    llama3 = "llama3"
    yarn = "yarn"
    linear = "linear"



# --- Default axis rules ---

DEFAULT_LOGICAL_AXIS_RULES = [
    ("batch", "data"),
    ("length", None),
    ("embed", "fsdp"),
    ("heads", "tensor"),
    ("q_heads", "tensor"),
    ("kv_heads", None),
    ("kv_head_dim", None),
    ("kv", None),
    ("mlp", "tensor"),
    ("vocab", "tensor"),
    ("norm", None),
    ("exp", None),
]


# --- Layer type generation ---


def _generate_layer_types(
    model_type: str, num_hidden_layers: int, data: dict[str, Any]
) -> list[str]:
    """Auto-generate layer_types list from model_type via model registry."""
    from mintext.models import get_layer_types
    return get_layer_types(model_type, num_hidden_layers, data)


# --- Config ---


class MinTextConfig(BaseModel):
    """Single flat configuration for MinText.

    Fields are grouped by concern. All fields have defaults so that
    `MinTextConfig()` produces a valid test-sized config.
    """

    model_config = ConfigDict(frozen=True, use_enum_values=True)

    @property
    def qk_head_dim(self) -> int:
        """Derived: total Q/K head dim for MLA (nope + rope)."""
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    # --- Run ---
    run_name: str = "default"
    model_name: str = "test-tiny"
    base_output_directory: str = "/tmp/mintext"
    seed: int = 0

    # --- Model Architecture ---
    num_hidden_layers: int = 2
    hidden_size: int = 128
    num_attention_heads: int = 4
    num_key_value_heads: int | None = None
    intermediate_size: int = 512
    head_dim: int | None = None
    vocab_size: int = 256
    rms_norm_eps: float = 1e-5
    max_position_embeddings: int = 128
    tie_word_embeddings: bool = False
    hidden_activation: str = "silu"

    # --- Model Type ---
    model_type: ModelType = ModelType.llama3

    # --- Qwen3 fields ---
    use_qk_norm: bool = False
    use_sliding_window: bool = False
    sliding_window: int = 4096
    max_window_layers: int = 28
    layer_types: list[str] = []

    # --- DeepSeek-V3 MLA fields ---
    attention_type: str = "mha"
    q_lora_rank: int = 0
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    rope_interleave: bool = False

    # --- MoE fields ---
    num_experts: int = 0
    num_experts_per_tok: int = 8
    moe_intermediate_size: int = 2048
    n_shared_experts: int = 1
    n_group: int = 8
    topk_group: int = 4
    first_k_dense_replace: int = 3
    routed_scaling_factor: float = 2.5
    norm_topk_prob: bool = True
    e_score_correction_bias_update_rate: float = 0.001

    # --- MoE kernel config ---
    moe_use_custom_vjp: bool = True
    moe_mosaic_fusion_group: bool = True
    moe_autotune: bool = False
    moe_autotune_cache_dir: str = ""
    moe_gate_up_tiling: list[int] = []
    moe_down_tiling: list[int] = []

    # --- Qwen3-Next fields ---
    full_attention_interval: int = 4
    partial_rotary_factor: float = 1.0
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_num_key_heads: int | None = None
    linear_num_value_heads: int | None = None
    linear_conv_kernel_dim: int = 4

    # --- Gemma3 fields ---
    query_pre_attn_scalar: float | None = None
    attn_logit_softcapping: float | None = None
    final_logit_softcapping: float | None = None
    sliding_window_pattern: int = 6
    rope_local_theta: float = 10_000.0
    use_post_ffw_norm: bool = False
    scale_embeddings: bool = False

    # --- Data Types ---
    dtype: DType = DType.bfloat16
    weight_dtype: DType = DType.float32

    # --- Positional Embeddings (RoPE) ---
    rope_type: RopeType = RopeType.default
    rope_theta: float = 10_000.0
    rope_scaling_factor: float = 1.0
    rope_original_max_position_embeddings: int = 0
    # YaRN-specific
    rope_yarn_beta_fast: float = 32.0
    rope_yarn_beta_slow: float = 1.0
    rope_yarn_mscale: float = 1.0
    rope_yarn_mscale_all_dim: float = 0.0
    # Llama3-specific
    rope_llama3_low_freq_factor: float = 1.0
    rope_llama3_high_freq_factor: float = 4.0

    # --- Training Loop ---
    num_vocab_tiles: int = 1
    steps: int = 10
    log_period: int = 1
    label_smoothing: float = 0.0

    # --- Optimizer ---
    shard_optimizer_over_data: bool = False
    optimizer: OptimizerType = OptimizerType.adamw
    learning_rate: float = 1e-3
    lr_schedule: LRScheduleType = LRScheduleType.cosine
    warmup_steps_fraction: float = 0.1
    lr_final_fraction: float = 0.1
    gradient_accumulation_steps: int = 1
    gradient_clip_threshold: float = 1.0
    adam_b1: float = 0.9
    adam_b2: float = 0.95
    adam_eps: float = 1e-8
    weight_decay: float = 0.1
    wsd_decay_steps_fraction: float = 0.1
    wsd_decay_style: str = "linear"
    z_loss_weight: float = 0.0
    muon_newton_schulz_steps: int = 5
    muon_beta: float = 0.95
    muon_consistent_rms: float | None = None

    # --- Remat ---
    remat_policy: RematPolicy = RematPolicy.none

    # --- Custom Kernels ---
    use_custom_kernels: bool = False

    # --- FP8 ---
    use_fp8: bool = False

    # --- Dataset ---
    dataset_type: DatasetType = DatasetType.synthetic
    per_device_batch_size: int = 2
    data_path: str = ""
    data_cache_dir: str = ""
    data_split: str = "99,1,0"
    num_data_epochs: int = 1
    add_extra_token: bool = True
    grain_worker_count: int = 4
    grain_prefetch_buffer_size: int = 500

    # --- Mesh / Parallelism ---
    ici_data_parallelism: int = 1
    ici_fsdp_parallelism: int = -1
    ici_tensor_parallelism: int = 1
    scan_layers: bool = True

    # --- DCN / Multi-node ---
    dcn_data_parallelism: int = 1
    dcn_fsdp_parallelism: int = 1
    dcn_tensor_parallelism: int = 1
    num_slices: int = 1
    jax_distributed_init_timeout: int = 300

    # --- Sharding ---
    logical_axis_rules: list[tuple[str, str | None]] = []

    # --- Checkpointing ---
    enable_checkpointing: bool = True
    async_checkpointing: bool = True
    checkpoint_period: int = 1000
    checkpoint_dir: str = ""
    max_checkpoints: int = 5
    load_parameters_from_path: str = ""
    load_full_state_from_path: str = ""

    # --- Performance ---
    tpu_type: str = "auto"  # TPU type for XLA flags: "auto", "v6e", "v6p", "v5e", "v5p", "v4"
    xla_flags: str = ""  # Additional LIBTPU_INIT_ARGS flags (appended to presets)
    peak_tflops_per_device: float = 0.0  # 0 = auto-detect

    # --- Logging ---
    enable_tensorboard: bool = True
    tensorboard_dir: str = ""
    log_dir: str = ""

    # --- Profiling ---
    enable_profiler: bool = False
    profiler_steps: int = 5
    skip_first_n_profiler_steps: int = 1

    # --- Validators ---

    @field_validator("rope_type", mode="before")
    @classmethod
    def _normalize_rope_type(cls, v: Any) -> Any:
        if v == "llama3_1":
            return "llama3"
        return v

    @field_validator("num_hidden_layers")
    @classmethod
    def num_hidden_layers_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"num_hidden_layers must be >= 1, got {v}")
        return v

    @field_validator("hidden_size")
    @classmethod
    def hidden_size_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"hidden_size must be >= 1, got {v}")
        return v

    @field_validator("num_attention_heads")
    @classmethod
    def num_attention_heads_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"num_attention_heads must be >= 1, got {v}")
        return v

    @field_validator("vocab_size")
    @classmethod
    def vocab_size_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"vocab_size must be >= 1, got {v}")
        return v

    @field_validator("steps")
    @classmethod
    def steps_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"steps must be >= 0, got {v}")
        return v

    @field_validator("per_device_batch_size")
    @classmethod
    def batch_size_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"per_device_batch_size must be >= 1, got {v}")
        return v

    @model_validator(mode="before")
    @classmethod
    def compute_derived(cls, data: dict[str, Any]) -> dict[str, Any]:
        if isinstance(data, dict):
            # head_dim defaults to hidden_size // num_attention_heads
            heads = data.get("num_attention_heads", 4)
            if data.get("head_dim") is None and heads and heads > 0:
                hidden = data.get("hidden_size", 128)
                data["head_dim"] = hidden // heads

            # num_key_value_heads defaults to num_attention_heads (MHA)
            if data.get("num_key_value_heads") is None:
                data["num_key_value_heads"] = data.get("num_attention_heads", 4)

            # linear_num_key_heads defaults to num_attention_heads
            if data.get("linear_num_key_heads") is None:
                data["linear_num_key_heads"] = data.get("num_attention_heads", 4)

            # linear_num_value_heads defaults to num_attention_heads
            if data.get("linear_num_value_heads") is None:
                data["linear_num_value_heads"] = data.get("num_attention_heads", 4)

            # Auto-generate layer_types from model_type if not provided
            if not data.get("layer_types"):
                model_type = data.get("model_type", "llama3")
                num_hidden_layers = data.get("num_hidden_layers", 2)
                data["layer_types"] = _generate_layer_types(model_type, num_hidden_layers, data)

            # logical_axis_rules defaults
            if not data.get("logical_axis_rules"):
                data["logical_axis_rules"] = DEFAULT_LOGICAL_AXIS_RULES

            # checkpoint_dir defaults to base_output_directory/checkpoints
            if not data.get("checkpoint_dir"):
                base = data.get("base_output_directory", "/tmp/mintext")
                run = data.get("run_name", "default")
                data["checkpoint_dir"] = os.path.join(base, run, "checkpoints")

            # tensorboard_dir defaults
            if not data.get("tensorboard_dir"):
                base = data.get("base_output_directory", "/tmp/mintext")
                run = data.get("run_name", "default")
                data["tensorboard_dir"] = os.path.join(base, run, "tensorboard")

            # log_dir defaults
            if not data.get("log_dir"):
                base = data.get("base_output_directory", "/tmp/mintext")
                run = data.get("run_name", "default")
                data["log_dir"] = os.path.join(base, run, "logs")

        return data

    @model_validator(mode="after")
    def _validate_vocab_tiling(self) -> MinTextConfig:
        if self.num_vocab_tiles > 1:
            total_tokens = self.per_device_batch_size * self.max_position_embeddings
            if total_tokens % self.num_vocab_tiles != 0:
                raise ValueError(
                    f"per_device_batch_size * max_position_embeddings ({total_tokens}) must be "
                    f"divisible by num_vocab_tiles ({self.num_vocab_tiles})"
                )
        return self


def _resolve_config_path(config_path: str | Path, base_dir: Path | None = None) -> Path:
    """Resolve a config path, handling relative paths."""
    p = Path(config_path)
    if not p.is_absolute() and base_dir is not None:
        p = base_dir / p
    return p.resolve()


def _load_yaml_chain(config_path: Path) -> dict[str, Any]:
    """Load a YAML config, recursively resolving base_config inheritance."""
    cfg = OmegaConf.load(config_path)
    data = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(data, dict):
        raise ValueError(f"Config file {config_path} must be a YAML mapping")

    base_config = data.pop("base_config", None)
    if base_config is not None:
        base_path = _resolve_config_path(base_config, config_path.parent)
        base_data = _load_yaml_chain(base_path)
        base_data.update(data)
        return base_data

    return data


def _validate_override_keys(overrides: dict[str, Any]) -> None:
    """Raise ValueError if any override key is not a known config field."""
    valid_keys = set(MinTextConfig.model_fields.keys())
    unknown = set(overrides.keys()) - valid_keys
    if unknown:
        # Suggest close matches
        import difflib

        suggestions = {}
        for key in unknown:
            matches = difflib.get_close_matches(key, valid_keys, n=1, cutoff=0.6)
            if matches:
                suggestions[key] = matches[0]
        parts = []
        for key in sorted(unknown):
            if key in suggestions:
                parts.append(f"  '{key}' (did you mean '{suggestions[key]}'?)")
            else:
                parts.append(f"  '{key}'")
        raise ValueError(f"Unknown config key(s):\n" + "\n".join(parts))


def _apply_env_overrides(
    data: dict[str, Any], cli_keys: frozenset[str]
) -> dict[str, Any]:
    """Apply MINTEXT_<KEY> environment variable overrides.

    Environment variables are lower priority than CLI overrides.
    If a key is set via both CLI and env var, a ValueError is raised.
    """
    valid_keys = set(MinTextConfig.model_fields.keys())
    for key in valid_keys:
        env_key = _ENV_PREFIX + key.upper()
        env_val = os.environ.get(env_key)
        if env_val is None:
            continue

        if key in cli_keys:
            raise ValueError(
                f"Key '{key}' is overridden by both CLI and environment "
                f"variable '{env_key}'. Remove one to avoid ambiguity."
            )

        # Parse env string to match the field's current type
        current = data.get(key)
        if isinstance(current, bool) or (current is None and key in data):
            # Bool needs special handling (avoid int("true"))
            env_val_lower = env_val.lower()
            if env_val_lower in ("true", "1", "yes"):
                data[key] = True
            elif env_val_lower in ("false", "0", "no"):
                data[key] = False
            else:
                raise ValueError(
                    f"Cannot parse '{env_val}' as bool for env var '{env_key}'"
                )
        elif isinstance(current, int):
            data[key] = int(env_val)
        elif isinstance(current, float):
            data[key] = float(env_val)
        else:
            data[key] = env_val

        logger.info("Config '%s' overridden by env var %s=%s", key, env_key, env_val)

    return data


def load_config(
    config_path: str | Path,
    overrides: dict[str, Any] | None = None,
) -> MinTextConfig:
    """Load a MinTextConfig from a YAML file with optional overrides.

    Priority (highest to lowest):
        CLI overrides > environment variables (MINTEXT_*) > YAML chain

    Args:
        config_path: Path to a YAML config file.
        overrides: Optional dict of field overrides applied after YAML loading.

    Returns:
        Validated, frozen MinTextConfig.
    """
    path = Path(config_path).resolve()
    data = _load_yaml_chain(path)

    # Validate override keys before applying
    cli_keys: frozenset[str] = frozenset()
    if overrides:
        _validate_override_keys(overrides)
        cli_keys = frozenset(overrides.keys())
        data.update(overrides)

    # Apply environment variable overrides
    data = _apply_env_overrides(data, cli_keys)

    return MinTextConfig(**data)


def print_help_config() -> None:
    """Print all config fields with types, defaults, and descriptions."""
    fields = MinTextConfig.model_fields
    print("MinText Configuration Fields")
    print("=" * 72)
    print(f"{'Field':<35} {'Type':<20} {'Default'}")
    print("-" * 72)
    for name, field_info in fields.items():
        annotation = field_info.annotation
        # Clean up type display
        type_str = getattr(annotation, "__name__", str(annotation))
        type_str = type_str.replace("typing.", "")
        if len(type_str) > 18:
            type_str = type_str[:17] + "…"
        default = field_info.default
        if isinstance(default, str) and len(default) > 30:
            default = default[:27] + "..."
        print(f"  {name:<33} {type_str:<20} {default}")
    print()
    print(f"Total: {len(fields)} fields")
    print()
    print("Environment variable override: MINTEXT_<FIELD_NAME_UPPER>=value")
    print("  Example: MINTEXT_LEARNING_RATE=1e-4")
