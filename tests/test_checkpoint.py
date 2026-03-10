"""Tests for the MinText checkpointing module.

Tests cover:
- CheckpointManager creation
- Save/restore round-trip
- Training resume (start_step detection)
- Params-only loading (fine-tuning)
- HuggingFace SafeTensors export/import round-trip
- Key mapping correctness
- Forward validation: MinText <-> HuggingFace round-trip logit comparison
"""

from __future__ import annotations

import json
import types
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

# Monkey-patch jax.ad_checkpoint before importing model code (JAX 0.9.x removed it)
if not hasattr(jax, "ad_checkpoint"):
    _ad_mod = types.ModuleType("jax.ad_checkpoint")
    _ad_mod.checkpoint_name = lambda x, name: x  # no-op passthrough
    jax.ad_checkpoint = _ad_mod
    import sys
    sys.modules["jax.ad_checkpoint"] = _ad_mod

from mintext.config import MinTextConfig
from mintext.checkpoint.manager import (
    create_checkpoint_manager,
    save_checkpoint,
    restore_checkpoint,
    maybe_restore_checkpoint,
    wait_for_checkpoint,
)
from mintext.checkpoint.conversion import (
    _hf_to_mintext_key_map,
    _mintext_to_hf_key_map,
    _mintext_to_hf_config,
    _flatten_state,
    save_hf_checkpoint,
    load_hf_checkpoint,
)
from mintext.models import Transformer, make_causal_mask
from mintext.trainer import TrainState, create_train_state


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def ckpt_config(tmp_path) -> MinTextConfig:
    """Config with checkpointing pointed at tmp_path."""
    return MinTextConfig(
        checkpoint_dir=str(tmp_path / "checkpoints"),
        checkpoint_period=2,
        max_checkpoints=3,
        async_checkpointing=False,  # Sync for deterministic tests
        enable_checkpointing=True,
        steps=10,
    )


@pytest.fixture
def model_and_state(ckpt_config):
    """Create a tiny model and TrainState."""
    model = Transformer(ckpt_config, rngs=nnx.Rngs(params=0))
    state = create_train_state(model, ckpt_config)
    return model, state


# ============================================================
# CheckpointManager tests
# ============================================================


class TestCheckpointManager:
    def test_create_manager(self, ckpt_config):
        manager = create_checkpoint_manager(ckpt_config)
        assert manager is not None
        assert Path(ckpt_config.checkpoint_dir).exists()

    def test_save_and_restore_round_trip(self, ckpt_config, model_and_state):
        model, state = model_and_state
        manager = create_checkpoint_manager(ckpt_config)

        # Save at step 0
        saved = save_checkpoint(manager, 0, state, force=True)
        assert saved
        wait_for_checkpoint(manager)

        # Restore
        restored, step = restore_checkpoint(manager, state, step=0)
        assert step == 0

        # Compare params
        orig_leaves = jax.tree.leaves(state.params)
        rest_leaves = jax.tree.leaves(restored.params)
        assert len(orig_leaves) == len(rest_leaves)
        for orig, rest in zip(orig_leaves, rest_leaves):
            np.testing.assert_array_equal(np.asarray(orig), np.asarray(rest))

    def test_restore_latest(self, ckpt_config, model_and_state):
        model, state = model_and_state
        manager = create_checkpoint_manager(ckpt_config)

        # Save at steps 0, 2, 4
        for step in [0, 2, 4]:
            save_checkpoint(manager, step, state, force=True)
        wait_for_checkpoint(manager)

        # Restore latest (should be step 4)
        restored, step = restore_checkpoint(manager, state)
        assert step == 4

    def test_max_checkpoints_kept(self, ckpt_config, model_and_state):
        model, state = model_and_state
        manager = create_checkpoint_manager(ckpt_config)

        # Save 5 checkpoints, max_checkpoints=3
        for step in [0, 2, 4, 6, 8]:
            save_checkpoint(manager, step, state, force=True)
        wait_for_checkpoint(manager)

        # Only latest 3 should be available
        assert manager.latest_step() == 8
        all_steps = manager.all_steps()
        assert len(all_steps) <= 3

    def test_restore_no_checkpoint_raises(self, ckpt_config, model_and_state):
        model, state = model_and_state
        manager = create_checkpoint_manager(ckpt_config)

        with pytest.raises(FileNotFoundError):
            restore_checkpoint(manager, state)


# ============================================================
# maybe_restore_checkpoint tests
# ============================================================


class TestMaybeRestore:
    def test_no_checkpoint_returns_initial(self, ckpt_config, model_and_state):
        model, state = model_and_state
        manager = create_checkpoint_manager(ckpt_config)

        restored, start_step = maybe_restore_checkpoint(manager, state, ckpt_config)
        assert start_step == 0

    def test_resume_from_existing(self, ckpt_config, model_and_state):
        model, state = model_and_state
        manager = create_checkpoint_manager(ckpt_config)

        # Simulate training: apply gradients to get step=5
        for _ in range(5):
            # Zero grads just to increment step
            zero_grads = jax.tree.map(jnp.zeros_like, state.params)
            state = state.apply_gradients(grads=zero_grads)

        save_checkpoint(manager, 5, state, force=True)
        wait_for_checkpoint(manager)

        # Now "restart" with fresh state
        fresh_state = create_train_state(model, ckpt_config)
        restored, start_step = maybe_restore_checkpoint(manager, fresh_state, ckpt_config)
        assert start_step == 5

    def test_checkpointing_disabled(self, model_and_state):
        model, state = model_and_state
        config = MinTextConfig(enable_checkpointing=False)
        restored, start_step = maybe_restore_checkpoint(None, state, config)
        assert start_step == 0

    def test_load_checkpoint_full_state(self, ckpt_config, model_and_state, tmp_path):
        model, state = model_and_state

        # Save a checkpoint to an external location
        ext_dir = tmp_path / "external_ckpt"
        ext_config = MinTextConfig(
            checkpoint_dir=str(ext_dir),
            async_checkpointing=False,
        )
        ext_manager = create_checkpoint_manager(ext_config)

        # Step forward and save
        for _ in range(3):
            zero_grads = jax.tree.map(jnp.zeros_like, state.params)
            state = state.apply_gradients(grads=zero_grads)
        save_checkpoint(ext_manager, 3, state, force=True)
        wait_for_checkpoint(ext_manager)

        # Load full state from external path
        cfg = MinTextConfig(
            load_checkpoint=str(ext_dir),
            async_checkpointing=False,
        )
        fresh_state = create_train_state(model, cfg)
        restored, start_step = maybe_restore_checkpoint(None, fresh_state, cfg)
        assert start_step == 3

    def test_load_checkpoint_params_only(self, ckpt_config, model_and_state, tmp_path):
        model, state = model_and_state

        # Save a checkpoint to an external location
        ext_dir = tmp_path / "external_ckpt"
        ext_config = MinTextConfig(
            checkpoint_dir=str(ext_dir),
            async_checkpointing=False,
        )
        ext_manager = create_checkpoint_manager(ext_config)

        # Step forward and save
        for _ in range(3):
            zero_grads = jax.tree.map(jnp.zeros_like, state.params)
            state = state.apply_gradients(grads=zero_grads)
        save_checkpoint(ext_manager, 3, state, force=True)
        wait_for_checkpoint(ext_manager)

        # Load params only — optimizer reset, step=0
        cfg = MinTextConfig(
            load_checkpoint=str(ext_dir),
            load_params_only=True,
            async_checkpointing=False,
        )
        fresh_state = create_train_state(model, cfg)
        restored, start_step = maybe_restore_checkpoint(None, fresh_state, cfg)
        assert start_step == 0

    def test_restart_resumes_full_state(self, tmp_path):
        """Simulate full restart: train 20 steps, save, create fresh state, resume."""
        config = MinTextConfig(
            checkpoint_dir=str(tmp_path / "checkpoints"),
            checkpoint_period=20,
            async_checkpointing=False,
            enable_checkpointing=True,
            steps=100,
        )

        # Phase 1: train from scratch and save at step 20
        model = Transformer(config, rngs=nnx.Rngs(params=0))
        state = create_train_state(model, config)
        manager = create_checkpoint_manager(config)

        # Apply non-zero gradients so params diverge from init
        for _ in range(20):
            ones_grads = jax.tree.map(jnp.ones_like, state.params)
            state = state.apply_gradients(grads=ones_grads)

        save_checkpoint(manager, 20, state, force=True)
        wait_for_checkpoint(manager)

        # Snapshot the trained params for comparison
        trained_leaves = [np.asarray(l) for l in jax.tree.leaves(state.params)]

        # Phase 2: simulate script restart — fresh model, fresh state, fresh manager
        fresh_model = Transformer(config, rngs=nnx.Rngs(params=0))
        fresh_state = create_train_state(fresh_model, config)
        fresh_manager = create_checkpoint_manager(config)

        # Sanity: fresh init params differ from trained params
        fresh_leaves = [np.asarray(l) for l in jax.tree.leaves(fresh_state.params)]
        any_differ = any(
            not np.array_equal(f, t) for f, t in zip(fresh_leaves, trained_leaves)
        )
        assert any_differ, "Gradients should have changed params from init"

        # Phase 3: resume from checkpoint_dir
        restored, start_step = maybe_restore_checkpoint(
            fresh_manager, fresh_state, config
        )
        assert start_step == 20

        # Restored params must match the trained params, not fresh init
        restored_leaves = [np.asarray(l) for l in jax.tree.leaves(restored.params)]
        assert len(restored_leaves) == len(trained_leaves)
        for i, (rest, trained) in enumerate(zip(restored_leaves, trained_leaves)):
            np.testing.assert_array_equal(
                rest, trained, err_msg=f"Param leaf {i} not restored correctly"
            )


# ============================================================
# HF conversion tests
# ============================================================


class TestHFKeyMapping:
    def test_forward_mapping_covers_all_layers(self):
        config = MinTextConfig(num_hidden_layers=2)
        key_map = _hf_to_mintext_key_map(config)

        # Should have embedding, per-layer (9 keys each), norm, output
        expected_per_layer = 9  # q,k,v,o,gate,up,down,attn_norm,ffn_norm
        expected_total = 1 + config.num_hidden_layers * expected_per_layer + 2  # embed + layers + norm + output
        assert len(key_map) == expected_total

    def test_reverse_mapping_same_size(self):
        config = MinTextConfig(num_hidden_layers=2)
        forward = _hf_to_mintext_key_map(config)
        reverse = _mintext_to_hf_key_map(config)
        assert len(forward) == len(reverse)

    def test_round_trip_key_coverage(self):
        config = MinTextConfig(num_hidden_layers=1)
        forward = _hf_to_mintext_key_map(config)
        reverse = _mintext_to_hf_key_map(config)
        # Every HF key in forward should map back through reverse
        for hf_key, (mt_key, _) in forward.items():
            assert mt_key in reverse
            assert reverse[mt_key][0] == hf_key


class TestHFConfigExport:
    def test_basic_fields(self):
        config = MinTextConfig(
            hidden_size=256, num_hidden_layers=4, num_attention_heads=8,
            vocab_size=32000, max_position_embeddings=2048,
        )
        hf_cfg = _mintext_to_hf_config(config)
        assert hf_cfg["hidden_size"] == 256
        assert hf_cfg["num_hidden_layers"] == 4
        assert hf_cfg["num_attention_heads"] == 8
        assert hf_cfg["vocab_size"] == 32000
        assert hf_cfg["model_type"] == "llama"


class TestHFSafeTensorsRoundTrip:
    def test_export_import_round_trip(self, tmp_path):
        config = MinTextConfig(num_hidden_layers=1, hidden_size=32, num_attention_heads=2, intermediate_size=64, vocab_size=64)
        model = Transformer(config, rngs=nnx.Rngs(params=0))

        # Get original params
        _, orig_state = nnx.split(model)
        orig_flat = _flatten_state(orig_state)

        # Export
        hf_dir = str(tmp_path / "hf_export")
        save_hf_checkpoint(model, config, hf_dir)

        # Verify files exist
        assert (Path(hf_dir) / "model.safetensors").exists()
        assert (Path(hf_dir) / "config.json").exists()

        # Import into a fresh model
        fresh_model = Transformer(config, rngs=nnx.Rngs(params=1))
        loaded_model = load_hf_checkpoint(hf_dir, config, fresh_model)

        # Compare params
        _, loaded_state = nnx.split(loaded_model)
        loaded_flat = _flatten_state(loaded_state)

        key_map = _hf_to_mintext_key_map(config)
        mapped_mt_keys = {mt_key for _, (mt_key, _) in key_map.items()}

        for key in mapped_mt_keys:
            if key in orig_flat and key in loaded_flat:
                np.testing.assert_allclose(
                    orig_flat[key], loaded_flat[key], rtol=1e-5,
                    err_msg=f"Mismatch for key {key}",
                )


# ============================================================
# Per-model conversion tests
# ============================================================


class TestQwen3Conversion:
    def test_key_mapping_has_qk_norm(self):
        config = MinTextConfig(model_type="qwen3", num_hidden_layers=2, use_qk_norm=True)
        key_map = _hf_to_mintext_key_map(config)
        assert "model.layers.0.self_attn.q_norm.weight" in key_map
        assert "model.layers.0.self_attn.k_norm.weight" in key_map

    def test_round_trip_key_coverage(self):
        config = MinTextConfig(model_type="qwen3", num_hidden_layers=1, use_qk_norm=True)
        forward = _hf_to_mintext_key_map(config)
        reverse = _mintext_to_hf_key_map(config)
        for hf_key, (mt_key, _) in forward.items():
            assert mt_key in reverse, f"{mt_key} not in reverse map"
            assert reverse[mt_key][0] == hf_key

    def test_hf_config_type(self):
        config = MinTextConfig(model_type="qwen3")
        hf_cfg = _mintext_to_hf_config(config)
        assert hf_cfg["model_type"] == "qwen3"
        assert hf_cfg["architectures"] == ["Qwen3ForCausalLM"]


class TestDeepSeekV3Conversion:
    def test_mla_key_mapping(self):
        config = MinTextConfig(
            model_type="deepseek_v3", num_hidden_layers=2, hidden_size=64, num_attention_heads=4,
            attention_type="mla", q_lora_rank=32, kv_lora_rank=32,
            qk_nope_head_dim=8, qk_rope_head_dim=4, v_head_dim=8,
            num_experts=4, num_experts_per_tok=2, moe_intermediate_size=64,
            n_group=2, topk_group=1, first_k_dense_replace=1,
        )
        key_map = _hf_to_mintext_key_map(config)
        # MLA keys
        assert "model.layers.0.self_attn.q_a_proj.weight" in key_map
        assert "model.layers.0.self_attn.kv_a_proj_with_mqa.weight" in key_map
        assert "model.layers.0.self_attn.kv_b_proj.weight" in key_map
        # MoE keys for layer 1 (>= first_k_dense_replace)
        assert "model.layers.1.mlp.gate.weight" in key_map
        # Expert weights handled by _import/_export_expert_weights, not in key map
        # Dense MLP for layer 0
        assert "model.layers.0.mlp.gate_proj.weight" in key_map

    def test_round_trip_key_coverage(self):
        config = MinTextConfig(
            model_type="deepseek_v3", num_hidden_layers=2, hidden_size=64, num_attention_heads=4,
            attention_type="mla", q_lora_rank=32, kv_lora_rank=32,
            qk_nope_head_dim=8, qk_rope_head_dim=4, v_head_dim=8,
            num_experts=4, num_experts_per_tok=2, moe_intermediate_size=64,
            n_group=2, topk_group=1, first_k_dense_replace=1,
        )
        forward = _hf_to_mintext_key_map(config)
        reverse = _mintext_to_hf_key_map(config)
        for hf_key, (mt_key, _) in forward.items():
            assert mt_key in reverse, f"{mt_key} not in reverse"

    def test_hf_config_type(self):
        config = MinTextConfig(model_type="deepseek_v3", num_experts=4)
        hf_cfg = _mintext_to_hf_config(config)
        assert hf_cfg["model_type"] == "deepseek_v3"
        assert hf_cfg["n_routed_experts"] == 4


class TestQwen3NextConversion:
    def test_key_mapping_hybrid_layers(self):
        config = MinTextConfig(
            model_type="qwen3_next", num_hidden_layers=4, hidden_size=64, num_attention_heads=4, head_dim=16,
            intermediate_size=128, vocab_size=128, use_qk_norm=True,
            full_attention_interval=4,
            linear_key_head_dim=8, linear_value_head_dim=8,
        )
        key_map = _hf_to_mintext_key_map(config)
        # Layer 0 = linear attention
        assert "model.layers.0.linear_attn.in_proj_qkvz.weight" in key_map
        assert "model.layers.0.linear_attn.A_log" in key_map
        # Layer 3 = full attention
        assert "model.layers.3.self_attn.q_proj.weight" in key_map
        if config.use_qk_norm:
            assert "model.layers.3.self_attn.q_norm.weight" in key_map

    def test_hf_config_type(self):
        config = MinTextConfig(model_type="qwen3_next")
        hf_cfg = _mintext_to_hf_config(config)
        assert hf_cfg["model_type"] == "qwen3_next"
        assert "layer_types" in hf_cfg


# ============================================================
# Forward validation tests
# ============================================================


class TestForwardValidation:
    """Test round-trip: MinText -> HF export -> HF load -> compare forward."""

    @pytest.fixture
    def tiny_llama_config(self):
        return MinTextConfig(
            model_type="llama3",
            num_hidden_layers=1,
            hidden_size=64,
            num_attention_heads=2,
            intermediate_size=128,
            vocab_size=256,
            max_position_embeddings=32,
            dtype="float32",
            weight_dtype="float32",
        )

    def _compare_logits(self, mt_logits, hf_logits, atol=1e-4):
        """Compare logits with tolerance."""
        mt_f64 = mt_logits.astype(np.float64)
        hf_f64 = hf_logits.astype(np.float64)

        max_err = np.max(np.abs(mt_f64 - hf_f64))
        assert max_err < atol, f"Max abs error {max_err:.6e} exceeds tolerance {atol}"

        # Cosine similarity
        mt_flat = mt_f64.flatten()
        hf_flat = hf_f64.flatten()
        cos_sim = np.dot(mt_flat, hf_flat) / (
            np.linalg.norm(mt_flat) * np.linalg.norm(hf_flat) + 1e-12
        )
        assert cos_sim > 0.999, f"Cosine similarity {cos_sim:.6f} too low"

    def _run_mintext_forward(self, model, config, input_ids):
        """Run MinText forward and return numpy logits."""
        dtype = getattr(jnp, config.dtype)
        tokens = jnp.array(input_ids)
        batch_size, seq_len = tokens.shape
        positions = jnp.broadcast_to(jnp.arange(seq_len), (batch_size, seq_len))
        mask = make_causal_mask(seq_len, dtype=dtype)
        logits, _ = model(tokens, positions, mask)
        return np.asarray(logits)

    @pytest.fixture
    def tiny_qwen3_config(self):
        return MinTextConfig(
            model_type="qwen3",
            num_hidden_layers=1,
            hidden_size=64,
            num_attention_heads=2,
            intermediate_size=128,
            vocab_size=256,
            max_position_embeddings=32,
            use_qk_norm=True,
            dtype="float32",
            weight_dtype="float32",
        )

    def _validate_mintext_to_hf(self, config, tmp_path):
        """Export MinText -> HF safetensors, load with AutoModel, compare forward."""
        pytest.importorskip("transformers")
        torch = pytest.importorskip("torch")
        from transformers import AutoModelForCausalLM

        mt_model = Transformer(config, rngs=nnx.Rngs(params=42))
        hf_dir = str(tmp_path / "hf_export")
        save_hf_checkpoint(mt_model, config, hf_dir)

        hf_model = AutoModelForCausalLM.from_pretrained(hf_dir, dtype=torch.float32)
        hf_model.eval()

        rng = np.random.RandomState(123)
        input_ids = rng.randint(0, config.vocab_size, (1, 16))

        mt_logits = self._run_mintext_forward(mt_model, config, input_ids)
        with torch.no_grad():
            hf_out = hf_model(torch.tensor(input_ids, dtype=torch.long))
            hf_logits = hf_out.logits.float().numpy()

        self._compare_logits(mt_logits, hf_logits, atol=1e-3)

    def _validate_hf_to_mintext(self, config, hf_config_cls, hf_model_cls, tmp_path):
        """Create HF model, save safetensors, load into MinText, compare forward."""
        torch = pytest.importorskip("torch")

        hf_cfg_dict = _mintext_to_hf_config(config)
        hf_cfg_dict["tie_word_embeddings"] = False
        hf_config = hf_config_cls(**hf_cfg_dict)
        hf_model = hf_model_cls(hf_config)
        hf_model.eval()

        hf_dir = tmp_path / "hf_model"
        hf_model.save_pretrained(hf_dir)

        mt_model = Transformer(config, rngs=nnx.Rngs(params=0))
        mt_model = load_hf_checkpoint(str(hf_dir), config, mt_model)

        rng = np.random.RandomState(456)
        input_ids = rng.randint(0, config.vocab_size, (1, 16))

        with torch.no_grad():
            hf_out = hf_model(torch.tensor(input_ids, dtype=torch.long))
            hf_logits = hf_out.logits.float().numpy()

        mt_logits = self._run_mintext_forward(mt_model, config, input_ids)
        self._compare_logits(mt_logits, hf_logits, atol=1e-3)

    def test_llama_mintext_to_hf_forward(self, tiny_llama_config, tmp_path):
        self._validate_mintext_to_hf(tiny_llama_config, tmp_path)

    def test_qwen3_mintext_to_hf_forward(self, tiny_qwen3_config, tmp_path):
        try:
            from transformers import Qwen3ForCausalLM  # noqa: F401
        except ImportError:
            pytest.skip("Qwen3 requires transformers>=4.51")
        self._validate_mintext_to_hf(tiny_qwen3_config, tmp_path)

    def test_llama_hf_to_mintext_forward(self, tiny_llama_config, tmp_path):
        pytest.importorskip("transformers")
        from transformers import LlamaConfig, LlamaForCausalLM
        self._validate_hf_to_mintext(tiny_llama_config, LlamaConfig, LlamaForCausalLM, tmp_path)

    def test_qwen3_hf_to_mintext_forward(self, tiny_qwen3_config, tmp_path):
        pytest.importorskip("transformers")
        try:
            from transformers import Qwen3Config, Qwen3ForCausalLM
        except ImportError:
            pytest.skip("Qwen3 requires transformers>=4.51")
        self._validate_hf_to_mintext(tiny_qwen3_config, Qwen3Config, Qwen3ForCausalLM, tmp_path)

    # --- DeepSeek-V3 ---

    @pytest.fixture
    def tiny_deepseek_v3_config(self):
        return MinTextConfig(
            model_type="deepseek_v3",
            attention_type="mla",
            num_hidden_layers=4,
            first_k_dense_replace=2,
            q_lora_rank=16,
            kv_lora_rank=16,
            qk_nope_head_dim=8,
            qk_rope_head_dim=8,
            v_head_dim=8,
            num_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=32,
            n_shared_experts=1,
            n_group=2,
            topk_group=1,
            hidden_size=64,
            num_attention_heads=2,
            intermediate_size=128,
            vocab_size=256,
            max_position_embeddings=32,
            dtype="float32",
            weight_dtype="float32",
        )

    def test_deepseek_v3_mintext_to_hf_forward(self, tiny_deepseek_v3_config, tmp_path):
        try:
            from transformers import DeepseekV3ForCausalLM  # noqa: F401
        except ImportError:
            pytest.skip("DeepseekV3 requires transformers>=4.51")
        self._validate_mintext_to_hf(tiny_deepseek_v3_config, tmp_path)

    def test_deepseek_v3_hf_to_mintext_forward(self, tiny_deepseek_v3_config, tmp_path):
        pytest.importorskip("transformers")
        try:
            from transformers import DeepseekV3Config, DeepseekV3ForCausalLM
        except ImportError:
            pytest.skip("DeepseekV3 requires transformers>=4.51")
        self._validate_hf_to_mintext(
            tiny_deepseek_v3_config, DeepseekV3Config, DeepseekV3ForCausalLM, tmp_path
        )

    # --- Qwen3-Next ---

    @pytest.fixture
    def tiny_qwen3_next_config(self):
        return MinTextConfig(
            model_type="qwen3_next",
            use_qk_norm=True,
            num_hidden_layers=4,
            full_attention_interval=2,
            partial_rotary_factor=0.5,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
            hidden_size=64,
            num_attention_heads=2,
            intermediate_size=128,
            vocab_size=256,
            max_position_embeddings=32,
            dtype="float32",
            weight_dtype="float32",
        )

    def test_qwen3_next_mintext_to_hf_forward(self, tiny_qwen3_next_config, tmp_path):
        try:
            from transformers import Qwen3NextForCausalLM  # noqa: F401
        except ImportError:
            pytest.skip("Qwen3Next requires transformers>=4.57")
        self._validate_mintext_to_hf(tiny_qwen3_next_config, tmp_path)

    def test_qwen3_next_hf_to_mintext_forward(self, tiny_qwen3_next_config, tmp_path):
        pytest.importorskip("transformers")
        try:
            from transformers import Qwen3NextConfig, Qwen3NextForCausalLM
        except ImportError:
            pytest.skip("Qwen3Next requires transformers>=4.57")
        self._validate_hf_to_mintext(
            tiny_qwen3_next_config, Qwen3NextConfig, Qwen3NextForCausalLM, tmp_path
        )
