"""Tests for MinText config system."""

import os
from pathlib import Path

import pytest

from mintext.config import (
    DEFAULT_LOGICAL_AXIS_RULES,
    MinTextConfig,
    _validate_override_keys,
    _apply_env_overrides,
    load_config,
    print_help_config,
)

CONFIGS_DIR = Path(__file__).parent.parent / "configs"


class TestMinTextConfigDefaults:
    """Test default config creation."""

    def test_default_creates_valid_config(self):
        cfg = MinTextConfig()
        assert cfg.num_hidden_layers == 2
        assert cfg.hidden_size == 128
        assert cfg.num_attention_heads == 4

    def test_derived_head_dim(self):
        cfg = MinTextConfig()
        assert cfg.head_dim == 32  # 128 // 4

    def test_derived_num_key_value_heads_defaults_to_num_attention_heads(self):
        cfg = MinTextConfig()
        assert cfg.num_key_value_heads == cfg.num_attention_heads

    def test_explicit_head_dim(self):
        cfg = MinTextConfig(hidden_size=256, num_attention_heads=4, head_dim=128)
        assert cfg.head_dim == 128

    def test_explicit_num_key_value_heads(self):
        cfg = MinTextConfig(num_attention_heads=32, num_key_value_heads=8)
        assert cfg.num_key_value_heads == 8

    def test_default_logical_axis_rules(self):
        cfg = MinTextConfig()
        assert cfg.logical_axis_rules == DEFAULT_LOGICAL_AXIS_RULES

    def test_default_checkpoint_dir(self):
        cfg = MinTextConfig(base_output_directory="/tmp/test", run_name="run1")
        assert cfg.checkpoint_dir == "/tmp/test/run1/checkpoints"

    def test_default_tensorboard_dir(self):
        cfg = MinTextConfig(base_output_directory="/tmp/test", run_name="run1")
        assert cfg.tensorboard_dir == "/tmp/test/run1/tensorboard"


class TestMinTextConfigValidation:
    """Test Pydantic validation."""

    def test_negative_num_hidden_layers_rejected(self):
        with pytest.raises(ValueError, match="num_hidden_layers must be >= 1"):
            MinTextConfig(num_hidden_layers=-1)

    def test_zero_num_hidden_layers_rejected(self):
        with pytest.raises(ValueError, match="num_hidden_layers must be >= 1"):
            MinTextConfig(num_hidden_layers=0)

    def test_negative_hidden_size_rejected(self):
        with pytest.raises(ValueError, match="hidden_size must be >= 1"):
            MinTextConfig(hidden_size=-1)

    def test_negative_num_attention_heads_rejected(self):
        with pytest.raises(ValueError, match="num_attention_heads must be >= 1"):
            MinTextConfig(num_attention_heads=0)

    def test_negative_vocab_size_rejected(self):
        with pytest.raises(ValueError, match="vocab_size must be >= 1"):
            MinTextConfig(vocab_size=0)

    def test_negative_steps_rejected(self):
        with pytest.raises(ValueError, match="steps must be >= 0"):
            MinTextConfig(steps=-1)

    def test_zero_batch_size_rejected(self):
        with pytest.raises(ValueError, match="per_device_batch_size must be >= 1"):
            MinTextConfig(per_device_batch_size=0)


class TestMinTextConfigImmutability:
    """Test that config is frozen after creation."""

    def test_cannot_set_field(self):
        cfg = MinTextConfig()
        with pytest.raises(Exception):  # ValidationError for frozen model
            cfg.num_hidden_layers = 10

    def test_cannot_delete_field(self):
        cfg = MinTextConfig()
        with pytest.raises(Exception):
            del cfg.num_hidden_layers


class TestLoadConfig:
    """Test YAML config loading."""

    def test_load_base_yml(self):
        cfg = load_config(CONFIGS_DIR / "base.yml")
        assert cfg.model_name == "test-tiny"
        assert cfg.num_hidden_layers == 2
        assert cfg.hidden_size == 128

    def test_load_test_tiny_with_inheritance(self):
        cfg = load_config(CONFIGS_DIR / "models" / "test-tiny.yml")
        assert cfg.model_name == "test-tiny"
        assert cfg.num_hidden_layers == 2
        assert cfg.vocab_size == 256

    def test_load_llama3_8b_with_inheritance(self):
        cfg = load_config(CONFIGS_DIR / "models" / "meta-llama-3-8b.yml")
        assert cfg.model_name == "meta-llama-3-8b"
        assert cfg.num_hidden_layers == 32
        assert cfg.hidden_size == 4096
        assert cfg.num_attention_heads == 32
        assert cfg.num_key_value_heads == 8
        assert cfg.intermediate_size == 14336
        assert cfg.head_dim == 128
        assert cfg.vocab_size == 128256

    def test_overrides_applied(self):
        cfg = load_config(CONFIGS_DIR / "base.yml", overrides={"num_hidden_layers": 8, "steps": 100})
        assert cfg.num_hidden_layers == 8
        assert cfg.steps == 100

    def test_overrides_take_precedence_over_yaml(self):
        cfg = load_config(
            CONFIGS_DIR / "models" / "meta-llama-3-8b.yml",
            overrides={"num_hidden_layers": 4},
        )
        assert cfg.num_hidden_layers == 4

    def test_enum_fields_from_yaml(self):
        cfg = load_config(CONFIGS_DIR / "base.yml")
        assert cfg.dtype == "bfloat16"
        assert cfg.optimizer == "adamw"
        assert cfg.lr_schedule == "cosine"
        assert cfg.dataset_type == "synthetic"

    def test_invalid_yaml_path_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.yml")


class TestUnknownKeyDetection:
    """Test that unknown override keys are caught."""

    def test_typo_raises_with_suggestion(self):
        with pytest.raises(ValueError, match="lerning_rate.*did you mean.*learning_rate"):
            load_config(CONFIGS_DIR / "base.yml", overrides={"lerning_rate": 1e-4})

    def test_completely_unknown_key_raises(self):
        with pytest.raises(ValueError, match="Unknown config key"):
            load_config(CONFIGS_DIR / "base.yml", overrides={"zzz_nonexistent": 42})

    def test_valid_keys_accepted(self):
        cfg = load_config(
            CONFIGS_DIR / "base.yml",
            overrides={"learning_rate": 1e-4, "steps": 5},
        )
        assert cfg.learning_rate == 1e-4
        assert cfg.steps == 5

    def test_multiple_unknown_keys(self):
        with pytest.raises(ValueError, match="Unknown config key"):
            load_config(
                CONFIGS_DIR / "base.yml",
                overrides={"bad_key1": 1, "bad_key2": 2},
            )


class TestEnvOverrides:
    """Test MINTEXT_<KEY> environment variable overrides."""

    def test_env_overrides_int(self, monkeypatch):
        monkeypatch.setenv("MINTEXT_STEPS", "42")
        cfg = load_config(CONFIGS_DIR / "base.yml")
        assert cfg.steps == 42

    def test_env_overrides_float(self, monkeypatch):
        monkeypatch.setenv("MINTEXT_LEARNING_RATE", "3e-4")
        cfg = load_config(CONFIGS_DIR / "base.yml")
        assert cfg.learning_rate == 3e-4

    def test_env_overrides_string(self, monkeypatch):
        monkeypatch.setenv("MINTEXT_RUN_NAME", "my_run")
        cfg = load_config(CONFIGS_DIR / "base.yml")
        assert cfg.run_name == "my_run"

    def test_env_overrides_bool_true(self, monkeypatch):
        monkeypatch.setenv("MINTEXT_ENABLE_PROFILER", "true")
        cfg = load_config(CONFIGS_DIR / "base.yml")
        assert cfg.enable_profiler is True

    def test_env_overrides_bool_false(self, monkeypatch):
        monkeypatch.setenv("MINTEXT_ENABLE_TENSORBOARD", "false")
        cfg = load_config(CONFIGS_DIR / "base.yml")
        assert cfg.enable_tensorboard is False

    def test_cli_takes_precedence_over_env_raises(self, monkeypatch):
        """CLI and env var for same key should raise."""
        monkeypatch.setenv("MINTEXT_STEPS", "99")
        with pytest.raises(ValueError, match="overridden by both CLI and environment"):
            load_config(CONFIGS_DIR / "base.yml", overrides={"steps": 50})

    def test_env_ignored_when_not_set(self):
        # Ensure no MINTEXT_ vars leak from other tests
        cfg = load_config(CONFIGS_DIR / "base.yml")
        assert cfg.steps == 10  # base.yml default


class TestHelpConfig:
    """Test --help-config output."""

    def test_print_help_config(self, capsys):
        print_help_config()
        captured = capsys.readouterr()
        assert "MinText Configuration Fields" in captured.out
        assert "learning_rate" in captured.out
        assert "num_hidden_layers" in captured.out
        assert "MINTEXT_" in captured.out


class TestOmegaConfCLI:
    """Test that OmegaConf CLI parsing handles complex types."""

    def test_int_override_via_load_config(self):
        cfg = load_config(CONFIGS_DIR / "base.yml", overrides={"steps": 999})
        assert cfg.steps == 999
