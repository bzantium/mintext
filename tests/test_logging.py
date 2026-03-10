"""Tests for logging, profiling, and pytree utilities."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from mintext.config import MinTextConfig
from mintext.utils.pytree import (
    calculate_tflops_per_device,
    calculate_tokens_per_device,
    check_nan,
    count_params,
)
from mintext.utils.logging import MetricLogger
from mintext.utils.profiling import Profiler


# ============================================================
# Pytree utilities
# ============================================================


class TestCountParams:
    def test_simple(self):
        params = {"a": jnp.ones((3, 4)), "b": jnp.ones((5,))}
        assert count_params(params) == 17

    def test_nested(self):
        params = {"layer": {"w": jnp.ones((10, 10)), "b": jnp.ones((10,))}}
        assert count_params(params) == 110

    def test_empty(self):
        assert count_params({}) == 0


class TestCheckNan:
    def test_no_nan(self):
        params = {"a": jnp.ones((3,)), "b": jnp.zeros((2, 2))}
        assert check_nan(params) is False

    def test_with_nan(self):
        params = {"a": jnp.array([1.0, float("nan"), 3.0])}
        assert check_nan(params) is True

    def test_nested_nan(self):
        params = {"layer": {"w": jnp.array([float("nan")])}}
        assert check_nan(params) is True


class TestCalculateTflops:
    def test_basic_llama(self):
        config = MinTextConfig(
            num_hidden_layers=2, hidden_size=128, num_attention_heads=4,
            intermediate_size=512, vocab_size=256, seq_length=64,
            per_device_batch_size=2,
        )
        tflops = calculate_tflops_per_device(config)
        assert tflops > 0
        # Small model should have small TFLOP count
        assert tflops < 1.0

    def test_larger_model_more_tflops(self):
        small = MinTextConfig(
            num_hidden_layers=2, hidden_size=128, num_attention_heads=4,
            intermediate_size=512, vocab_size=256, seq_length=64,
        )
        large = MinTextConfig(
            num_hidden_layers=8, hidden_size=512, num_attention_heads=8,
            intermediate_size=2048, vocab_size=256, seq_length=64,
        )
        assert calculate_tflops_per_device(large) > calculate_tflops_per_device(small)

    def test_moe_model(self):
        config = MinTextConfig(
            num_hidden_layers=4, hidden_size=128, num_attention_heads=4, head_dim=32,
            intermediate_size=256, vocab_size=128, seq_length=64,
            num_experts=4, num_experts_per_tok=2,
            moe_intermediate_size=128, first_k_dense_replace=2,
        )
        tflops = calculate_tflops_per_device(config)
        assert tflops > 0

    def test_mla_model(self):
        config = MinTextConfig(
            num_hidden_layers=2, hidden_size=64, num_attention_heads=4, head_dim=16,
            intermediate_size=128, vocab_size=128, seq_length=64,
            attention_type="mla",
            q_lora_rank=32, kv_lora_rank=32,
            qk_nope_head_dim=8, qk_rope_head_dim=4, v_head_dim=8,
        )
        tflops = calculate_tflops_per_device(config)
        assert tflops > 0

    def test_gradient_accumulation_scales(self):
        base = MinTextConfig(
            num_hidden_layers=2, hidden_size=128, num_attention_heads=4,
            intermediate_size=512, vocab_size=256, seq_length=64,
            gradient_accumulation_steps=1,
        )
        ga4 = MinTextConfig(
            num_hidden_layers=2, hidden_size=128, num_attention_heads=4,
            intermediate_size=512, vocab_size=256, seq_length=64,
            gradient_accumulation_steps=4,
        )
        assert calculate_tflops_per_device(ga4) == pytest.approx(
            calculate_tflops_per_device(base) * 4
        )


class TestCalculateTokens:
    def test_basic(self):
        config = MinTextConfig(per_device_batch_size=4, seq_length=128)
        assert calculate_tokens_per_device(config) == 4 * 128

    def test_with_ga(self):
        config = MinTextConfig(
            per_device_batch_size=4, seq_length=128,
            gradient_accumulation_steps=2,
        )
        assert calculate_tokens_per_device(config) == 4 * 128 * 2


# ============================================================
# MetricLogger
# ============================================================


class TestMetricLogger:
    def test_creates_without_tensorboard(self):
        config = MinTextConfig(enable_tensorboard=False)
        ml = MetricLogger(config)
        assert ml.writer is None
        ml.close()

    def test_creates_with_tensorboard(self, tmp_path):
        config = MinTextConfig(
            enable_tensorboard=True,
            tensorboard_dir=str(tmp_path / "tb"),
        )
        ml = MetricLogger(config)
        assert ml.writer is not None
        ml.close()
        assert (tmp_path / "tb").exists()

    def test_log_step(self, tmp_path):
        config = MinTextConfig(
            enable_tensorboard=True,
            tensorboard_dir=str(tmp_path / "tb"),
            num_hidden_layers=2, hidden_size=128, num_attention_heads=4,
            intermediate_size=512, vocab_size=256, seq_length=64,
        )
        ml = MetricLogger(config)
        metrics = {
            "loss": jnp.array(5.0),
            "learning_rate": jnp.array(1e-3),
            "grad_norm": jnp.array(1.5),
            "param_norm": jnp.array(100.0),
        }
        ml.log_step(0, metrics, step_time=0.5)
        ml.close()

    def test_log_step_without_tensorboard(self):
        config = MinTextConfig(enable_tensorboard=False)
        ml = MetricLogger(config)
        metrics = {
            "loss": jnp.array(5.0),
            "learning_rate": jnp.array(1e-3),
            "grad_norm": jnp.array(1.5),
        }
        # Should not raise
        ml.log_step(0, metrics, step_time=0.5)
        ml.close()

    def test_with_params(self, tmp_path):
        config = MinTextConfig(
            enable_tensorboard=True,
            tensorboard_dir=str(tmp_path / "tb"),
        )
        params = {"w": jnp.ones((100, 100))}
        ml = MetricLogger(config, params=params)
        assert ml.tflops_per_device > 0
        assert ml.tokens_per_device > 0
        ml.close()


# ============================================================
# Profiler
# ============================================================


class TestProfiler:
    def test_disabled(self):
        config = MinTextConfig(enable_profiler=False)
        prof = Profiler(config)
        assert not prof.enabled
        # Should be no-ops
        prof.maybe_activate(0)
        prof.maybe_deactivate(0)
        prof.stop()

    def test_enabled_setup(self, tmp_path):
        config = MinTextConfig(
            enable_profiler=True,
            profiler_steps=3,
            skip_first_n_profiler_steps=2,
            base_output_directory=str(tmp_path),
            run_name="test_run",
        )
        prof = Profiler(config)
        assert prof.enabled
        assert prof.start_step == 2
        assert prof.end_step == 5
        assert not prof.active

    def test_activation_window(self, tmp_path):
        config = MinTextConfig(
            enable_profiler=True,
            profiler_steps=2,
            skip_first_n_profiler_steps=1,
            base_output_directory=str(tmp_path),
            run_name="test_run",
        )
        prof = Profiler(config)
        # Step 0: should not activate
        prof.maybe_activate(0)
        assert not prof.active
        # Step 1: should activate (start_step)
        prof.maybe_activate(1)
        assert prof.active
        # Step 2: should deactivate (end_step - 1)
        prof.maybe_deactivate(2)
        assert not prof.active
