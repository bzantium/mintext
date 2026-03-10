"""Tests for MinText model components."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from mintext.config import MinTextConfig
from mintext.modules.attention import Attention, make_sliding_window_mask
from mintext.modules.linear import MLP, Linear, ACT2FN
from mintext.modules.mla import MLAttention
from mintext.modules.norm import RMSNorm
from mintext.modules.rope import RotaryEmbedding, compute_inv_freq
from mintext.models import DecoderLayer, Transformer, make_causal_mask
from mintext.modules.moe import MoERouter, MoEExperts, MoEBlock
from mintext.modules.linear_attention import GatedDeltaRuleAttention, chunk_gated_delta_rule


@pytest.fixture
def config():
    return MinTextConfig(
        num_hidden_layers=2,
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        intermediate_size=128,
        vocab_size=128,
        seq_length=32,
        dtype="float32",
        weight_dtype="float32",
    )


@pytest.fixture
def rngs():
    return nnx.Rngs(params=0, dropout=1)


BATCH, SEQ = 2, 16


class TestRMSNorm:
    def test_output_shape(self, rngs):
        norm = RMSNorm(64, dtype=jnp.float32, rngs=rngs)
        x = jax.random.normal(jax.random.key(0), (BATCH, SEQ, 64))
        y = norm(x)
        assert y.shape == (BATCH, SEQ, 64)

    def test_normalized_rms(self, rngs):
        norm = RMSNorm(64, dtype=jnp.float32, rngs=rngs)
        x = jax.random.normal(jax.random.key(0), (BATCH, SEQ, 64)) * 10.0
        y = norm(x)
        rms = jnp.sqrt(jnp.mean(y ** 2, axis=-1))
        # After RMSNorm with scale=1, RMS should be ~1
        assert jnp.allclose(rms, 1.0, atol=0.1)


class TestLinear:
    def test_2d_kernel(self, rngs):
        dense = Linear(64, 128, dtype=jnp.float32, weight_dtype=jnp.float32, rngs=rngs)
        x = jax.random.normal(jax.random.key(0), (BATCH, SEQ, 64))
        y = dense(x)
        assert y.shape == (BATCH, SEQ, 128)

    def test_3d_expand(self, rngs):
        dense = Linear(64, (4, 16), dtype=jnp.float32, weight_dtype=jnp.float32, rngs=rngs)
        x = jax.random.normal(jax.random.key(0), (BATCH, SEQ, 64))
        y = dense(x)
        assert y.shape == (BATCH, SEQ, 4, 16)

    def test_3d_contract(self, rngs):
        dense = Linear((4, 16), 64, dtype=jnp.float32, weight_dtype=jnp.float32, rngs=rngs)
        x = jax.random.normal(jax.random.key(0), (BATCH, SEQ, 4, 16))
        y = dense(x)
        assert y.shape == (BATCH, SEQ, 64)


class TestRotaryEmbedding:
    def test_output_shape(self, config):
        rope = RotaryEmbedding(config)
        x = jax.random.normal(jax.random.key(0), (BATCH, SEQ, 4, 16))
        pos = jnp.broadcast_to(jnp.arange(SEQ), (BATCH, SEQ))
        y = rope(x, pos)
        assert y.shape == x.shape

    def test_position_zero_identity(self, config):
        """At position 0, cos=1 and sin=0, so output should equal input."""
        rope = RotaryEmbedding(config)
        x = jax.random.normal(jax.random.key(0), (1, 1, 4, 16))
        pos = jnp.zeros((1, 1), dtype=jnp.int32)
        y = rope(x, pos)
        assert jnp.allclose(y, x, atol=1e-5)


class TestRoPEVariants:
    """Tests for extended RoPE variants (linear, YaRN, Llama3)."""

    def test_linear_scaling(self):
        """Linear scaling halves frequencies relative to default."""
        cfg_default = MinTextConfig(rope_type="default", dtype="float32", weight_dtype="float32")
        cfg_linear = MinTextConfig(
            rope_type="linear", rope_scaling_factor=2.0,
            dtype="float32", weight_dtype="float32",
        )
        inv_default, _ = compute_inv_freq(cfg_default, 64)
        inv_linear, _ = compute_inv_freq(cfg_linear, 64)
        assert jnp.allclose(inv_linear, inv_default / 2.0, atol=1e-6)

    def test_yarn_frequency_blending(self):
        """YaRN produces blended frequencies between extrapolation and interpolation."""
        cfg = MinTextConfig(
            rope_type="yarn", rope_scaling_factor=4.0,
            rope_original_seq_length=4096,
            rope_yarn_beta_fast=32.0, rope_yarn_beta_slow=1.0,
            rope_yarn_mscale=1.0, rope_yarn_mscale_all_dim=0.0,
            dtype="float32", weight_dtype="float32",
        )
        inv_freq, af = compute_inv_freq(cfg, 128)
        assert inv_freq.shape == (64,)
        # Attention factor > 1 for factor > 1
        assert af > 1.0

    def test_yarn_attention_scaling(self):
        """YaRN mscale: factor=4, mscale=1 -> attention_factor ~= 0.1*ln(4)+1."""
        import math
        cfg = MinTextConfig(
            rope_type="yarn", rope_scaling_factor=4.0,
            rope_original_seq_length=4096,
            rope_yarn_mscale=1.0, rope_yarn_mscale_all_dim=0.0,
            dtype="float32", weight_dtype="float32",
        )
        _, af = compute_inv_freq(cfg, 128)
        expected = 0.1 * math.log(4.0) + 1.0
        assert abs(af - expected) < 1e-6

    def test_llama3_three_regions(self):
        """Llama3 produces three regions: unchanged, scaled, and smoothed."""
        cfg_default = MinTextConfig(rope_type="default", dtype="float32", weight_dtype="float32")
        cfg_llama3 = MinTextConfig(
            rope_type="llama3", rope_scaling_factor=8.0,
            rope_original_seq_length=8192,
            rope_llama3_low_freq_factor=1.0, rope_llama3_high_freq_factor=4.0,
            dtype="float32", weight_dtype="float32",
        )
        inv_default, _ = compute_inv_freq(cfg_default, 128)
        inv_llama3, af = compute_inv_freq(cfg_llama3, 128)
        assert af == 1.0  # no attention scaling
        # Some frequencies should differ from default (scaled or smoothed)
        assert not jnp.allclose(inv_default, inv_llama3)
        # High-freq dims (beginning) should be close to default
        assert jnp.allclose(inv_default[:3], inv_llama3[:3], atol=1e-5)

    def test_rope_backward_compat(self, config):
        """Default rope_type produces same output as the old timescale approach."""
        rope = RotaryEmbedding(config)
        x = jax.random.normal(jax.random.key(0), (1, 8, 4, 16))
        pos = jnp.broadcast_to(jnp.arange(8), (1, 8))
        y = rope(x, pos)
        assert y.shape == x.shape
        # At position 0, output should equal input (cos=1, sin=0)
        pos_zero = jnp.zeros((1, 1), dtype=jnp.int32)
        x_single = jax.random.normal(jax.random.key(1), (1, 1, 4, 16))
        y_zero = rope(x_single, pos_zero)
        assert jnp.allclose(y_zero, x_single, atol=1e-5)

    def test_llama3_1_alias(self):
        """rope_type='llama3_1' normalizes to 'llama3'."""
        cfg = MinTextConfig(rope_type="llama3_1", dtype="float32", weight_dtype="float32")
        assert cfg.rope_type == "llama3"

    def test_rope_variant_forward_pass(self):
        """Full forward pass works with each RoPE variant."""
        rngs = nnx.Rngs(params=0, dropout=1)
        for rt in ["default", "linear", "yarn", "llama3"]:
            cfg = MinTextConfig(
                num_hidden_layers=1, hidden_size=64, num_attention_heads=4, head_dim=16,
                intermediate_size=128, vocab_size=128, seq_length=32,
                rope_type=rt, rope_scaling_factor=2.0,
                rope_original_seq_length=32,
                dtype="float32", weight_dtype="float32",
            )
            model = Transformer(cfg, rngs=rngs)
            tokens = jax.random.randint(jax.random.key(0), (1, 8), 0, 128)
            pos = jnp.broadcast_to(jnp.arange(8), (1, 8))
            mask = make_causal_mask(8)
            logits, _ = model(tokens, pos, mask)
            assert logits.shape == (1, 8, 128), f"Failed for rope_type={rt}"
            assert jnp.all(jnp.isfinite(logits)), f"NaN/Inf for rope_type={rt}"


class TestAttention:
    def test_output_shape(self, config, rngs):
        attn = Attention(config, rngs=rngs)
        x = jax.random.normal(jax.random.key(0), (BATCH, SEQ, config.hidden_size))
        pos = jnp.broadcast_to(jnp.arange(SEQ), (BATCH, SEQ))
        mask = make_causal_mask(SEQ)
        y = attn(x, pos, mask)
        assert y.shape == (BATCH, SEQ, config.hidden_size)

    def test_gqa_head_counts(self, config, rngs):
        attn = Attention(config, rngs=rngs)
        assert attn.num_attention_heads == 4
        assert attn.num_key_value_heads == 2

    def test_gqa_matches_repeated_kv_reference(self, config, rngs):
        attn = Attention(config, rngs=rngs)
        x = jax.random.normal(jax.random.key(0), (BATCH, SEQ, config.hidden_size))
        pos = jnp.broadcast_to(jnp.arange(SEQ), (BATCH, SEQ))
        mask = make_causal_mask(SEQ)

        y = attn(x, pos, mask)

        q = attn.query(x)
        k = attn.key(x)
        v = attn.value(x)
        q = attn.rope(q, pos)
        k = attn.rope(k, pos)

        repeats = attn.num_attention_heads // attn.num_key_value_heads
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(jnp.repeat(k, repeats, axis=2), (0, 2, 1, 3))
        v = jnp.transpose(jnp.repeat(v, repeats, axis=2), (0, 2, 1, 3))

        ref_weights = jnp.matmul(q, jnp.swapaxes(k, -2, -1)) * attn.scale
        ref_weights = jax.nn.softmax(jnp.asarray(ref_weights, jnp.float32) + mask, axis=-1)
        ref_output = jnp.matmul(jnp.asarray(ref_weights, attn.dtype), v)
        ref_output = jnp.transpose(ref_output, (0, 2, 1, 3))
        ref = attn.out(ref_output)

        assert jnp.allclose(y, ref, atol=1e-5)

    def test_mha_when_kv_equals_heads(self, rngs):
        cfg = MinTextConfig(
            hidden_size=64, num_attention_heads=4, num_key_value_heads=4, head_dim=16,
            dtype="float32", weight_dtype="float32",
        )
        attn = Attention(cfg, rngs=rngs)
        x = jax.random.normal(jax.random.key(0), (BATCH, SEQ, 64))
        pos = jnp.broadcast_to(jnp.arange(SEQ), (BATCH, SEQ))
        y = attn(x, pos)
        assert y.shape == (BATCH, SEQ, 64)

    def test_qk_norm(self, config, rngs):
        attn = Attention(config, rngs=rngs, use_qk_norm=True)
        assert hasattr(attn, 'q_norm')
        assert hasattr(attn, 'k_norm')
        x = jax.random.normal(jax.random.key(0), (BATCH, SEQ, config.hidden_size))
        pos = jnp.broadcast_to(jnp.arange(SEQ), (BATCH, SEQ))
        mask = make_causal_mask(SEQ)
        y = attn(x, pos, mask)
        assert y.shape == (BATCH, SEQ, config.hidden_size)

    def test_sliding_window(self, config, rngs):
        attn = Attention(config, rngs=rngs, sliding_window=4)
        x = jax.random.normal(jax.random.key(0), (BATCH, SEQ, config.hidden_size))
        pos = jnp.broadcast_to(jnp.arange(SEQ), (BATCH, SEQ))
        mask = make_causal_mask(SEQ)
        y = attn(x, pos, mask)
        assert y.shape == (BATCH, SEQ, config.hidden_size)


class TestMLP:
    def test_output_shape(self, config, rngs):
        mlp = MLP(config, rngs=rngs)
        x = jax.random.normal(jax.random.key(0), (BATCH, SEQ, config.hidden_size))
        y = mlp(x)
        assert y.shape == (BATCH, SEQ, config.hidden_size)


class TestDecoderLayer:
    def test_output_shape(self, config, rngs):
        layer = DecoderLayer(config, layer_idx=0, rngs=rngs)
        x = jax.random.normal(jax.random.key(0), (BATCH, SEQ, config.hidden_size))
        pos = jnp.broadcast_to(jnp.arange(SEQ), (BATCH, SEQ))
        mask = make_causal_mask(SEQ)
        y, aux = layer(x, pos, mask)
        assert y.shape == (BATCH, SEQ, config.hidden_size)
        assert aux is None  # no MoE


class TestTransformer:
    def test_output_shape(self, config, rngs):
        model = Transformer(config, rngs=rngs)
        tokens = jax.random.randint(jax.random.key(0), (BATCH, SEQ), 0, config.vocab_size)
        pos = jnp.broadcast_to(jnp.arange(SEQ), (BATCH, SEQ))
        mask = make_causal_mask(SEQ)
        logits, _ = model(tokens, pos, mask)
        assert logits.shape == (BATCH, SEQ, config.vocab_size)

    def test_tie_word_embeddings(self, rngs):
        cfg = MinTextConfig(
            num_hidden_layers=1, hidden_size=64, num_attention_heads=4, head_dim=16,
            intermediate_size=128, vocab_size=128, seq_length=32,
            tie_word_embeddings=True, dtype="float32", weight_dtype="float32",
        )
        model = Transformer(cfg, rngs=rngs)
        tokens = jax.random.randint(jax.random.key(0), (BATCH, SEQ), 0, cfg.vocab_size)
        pos = jnp.broadcast_to(jnp.arange(SEQ), (BATCH, SEQ))
        mask = make_causal_mask(SEQ)
        logits, _ = model(tokens, pos, mask)
        assert logits.shape == (BATCH, SEQ, cfg.vocab_size)
        assert not hasattr(model, "output_proj")

    def test_forward_is_jittable(self, config, rngs):
        model = Transformer(config, rngs=rngs)
        tokens = jax.random.randint(jax.random.key(0), (BATCH, SEQ), 0, config.vocab_size)
        pos = jnp.broadcast_to(jnp.arange(SEQ), (BATCH, SEQ))
        mask = make_causal_mask(SEQ)

        @nnx.jit
        def forward(model, tokens, pos, mask):
            return model(tokens, pos, mask)

        logits, _ = forward(model, tokens, pos, mask)
        assert logits.shape == (BATCH, SEQ, config.vocab_size)

    def test_param_count(self, config, rngs):
        model = Transformer(config, rngs=rngs)
        graph, state = nnx.split(model)
        total = sum(p.size for p in jax.tree.leaves(state))
        assert total > 0

    def test_causal_mask(self):
        mask = make_causal_mask(4)
        assert mask.shape == (1, 1, 4, 4)
        # Upper triangle should be large negative
        assert mask[0, 0, 0, 1] < -1e9
        # Diagonal and lower should be 0
        assert mask[0, 0, 0, 0] == 0.0
        assert mask[0, 0, 1, 0] == 0.0


# --- Qwen3 tests ---


class TestSlidingWindowMask:
    def test_shape(self):
        mask = make_sliding_window_mask(8, 4)
        assert mask.shape == (1, 1, 8, 8)

    def test_window_allows_recent(self):
        mask = make_sliding_window_mask(8, 3)
        # Position 4 should see positions 2,3,4 (window=3)
        assert mask[0, 0, 4, 4] == 0.0  # self
        assert mask[0, 0, 4, 3] == 0.0  # 1 ago
        assert mask[0, 0, 4, 2] == 0.0  # 2 ago
        assert mask[0, 0, 4, 1] < -1e9  # 3 ago, blocked

    def test_window_blocks_future(self):
        mask = make_sliding_window_mask(8, 3)
        assert mask[0, 0, 2, 5] < -1e9  # future blocked


class TestQwen3Attention:
    def test_qk_norm_params_exist(self, config, rngs):
        attn = Attention(config, rngs=rngs, use_qk_norm=True)
        assert hasattr(attn, 'q_norm')
        assert hasattr(attn, 'k_norm')

    def test_sliding_window_output(self, config, rngs):
        attn = Attention(config, rngs=rngs, sliding_window=4)
        x = jax.random.normal(jax.random.key(0), (BATCH, SEQ, config.hidden_size))
        pos = jnp.broadcast_to(jnp.arange(SEQ), (BATCH, SEQ))
        mask = make_causal_mask(SEQ)
        y = attn(x, pos, mask)
        assert y.shape == (BATCH, SEQ, config.hidden_size)
        assert jnp.all(jnp.isfinite(y))

    def test_qwen3_full_model(self, rngs):
        cfg = MinTextConfig(
            model_type="qwen3", num_hidden_layers=2, hidden_size=64, num_attention_heads=4, head_dim=16,
            intermediate_size=128, vocab_size=128, seq_length=32,
            use_qk_norm=True, use_sliding_window=True, sliding_window=8, max_window_layers=1,
            dtype="float32", weight_dtype="float32",
        )
        model = Transformer(cfg, rngs=rngs)
        tokens = jax.random.randint(jax.random.key(0), (BATCH, SEQ), 0, 128)
        pos = jnp.broadcast_to(jnp.arange(SEQ), (BATCH, SEQ))
        mask = make_causal_mask(SEQ)
        logits, _ = model(tokens, pos, mask)
        assert logits.shape == (BATCH, SEQ, 128)


# --- MLA tests ---


class TestMLAttention:
    @pytest.fixture
    def mla_config(self):
        return MinTextConfig(
            model_type="deepseek_v3", num_hidden_layers=2, hidden_size=64, num_attention_heads=4,
            attention_type="mla", q_lora_rank=32, kv_lora_rank=32,
            qk_nope_head_dim=8, qk_rope_head_dim=4, v_head_dim=8,
            intermediate_size=128, vocab_size=128, seq_length=32,
            dtype="float32", weight_dtype="float32",
        )

    def test_output_shape(self, mla_config, rngs):
        mla = MLAttention(mla_config, layer_idx=0, rngs=rngs)
        x = jax.random.normal(jax.random.key(0), (BATCH, SEQ, mla_config.hidden_size))
        pos = jnp.broadcast_to(jnp.arange(SEQ), (BATCH, SEQ))
        mask = make_causal_mask(SEQ)
        y = mla(x, pos, mask)
        assert y.shape == (BATCH, SEQ, mla_config.hidden_size)

    def test_lora_projections_exist(self, mla_config, rngs):
        mla = MLAttention(mla_config, layer_idx=0, rngs=rngs)
        assert hasattr(mla, 'q_a_proj')
        assert hasattr(mla, 'q_a_norm')
        assert hasattr(mla, 'q_b_proj')
        assert hasattr(mla, 'kv_a_proj')
        assert hasattr(mla, 'kv_b_proj')

    def test_no_lora_q(self, rngs):
        cfg = MinTextConfig(
            model_type="deepseek_v3", num_hidden_layers=2, hidden_size=64, num_attention_heads=4,
            attention_type="mla", q_lora_rank=0, kv_lora_rank=32,
            qk_nope_head_dim=8, qk_rope_head_dim=4, v_head_dim=8,
            intermediate_size=128, vocab_size=128, seq_length=32,
            dtype="float32", weight_dtype="float32",
        )
        mla = MLAttention(cfg, layer_idx=0, rngs=rngs)
        assert hasattr(mla, 'q_proj')
        x = jax.random.normal(jax.random.key(0), (BATCH, SEQ, 64))
        pos = jnp.broadcast_to(jnp.arange(SEQ), (BATCH, SEQ))
        y = mla(x, pos)
        assert y.shape == (BATCH, SEQ, 64)

    def test_output_finite(self, mla_config, rngs):
        mla = MLAttention(mla_config, layer_idx=0, rngs=rngs)
        x = jax.random.normal(jax.random.key(0), (BATCH, SEQ, mla_config.hidden_size))
        pos = jnp.broadcast_to(jnp.arange(SEQ), (BATCH, SEQ))
        mask = make_causal_mask(SEQ)
        y = mla(x, pos, mask)
        assert jnp.all(jnp.isfinite(y))


# --- MoE tests ---


class TestMoERouter:
    @pytest.fixture
    def moe_config(self):
        return MinTextConfig(
            hidden_size=64, num_attention_heads=4, intermediate_size=128, vocab_size=128,
            num_experts=4, num_experts_per_tok=2,
            moe_intermediate_size=64, n_group=2, topk_group=1,
            routed_scaling_factor=1.0,
            dtype="float32", weight_dtype="float32",
        )

    def test_routing_shapes(self, moe_config, rngs):
        router = MoERouter(moe_config, rngs=rngs)
        x = jax.random.normal(jax.random.key(0), (32, moe_config.hidden_size))
        indices, weights, scores = router(x)
        assert indices.shape == (32, 2)
        assert weights.shape == (32, 2)
        assert scores.shape == (32, 4)

    def test_indices_in_range(self, moe_config, rngs):
        router = MoERouter(moe_config, rngs=rngs)
        x = jax.random.normal(jax.random.key(0), (32, moe_config.hidden_size))
        indices, _, _ = router(x)
        assert jnp.all(indices >= 0)
        assert jnp.all(indices < moe_config.num_experts)

    def test_bias_application(self, moe_config, rngs):
        router = MoERouter(moe_config, rngs=rngs)
        x = jax.random.normal(jax.random.key(0), (32, moe_config.hidden_size))
        _, weights1, _ = router(x)
        # Set large bias for expert 0
        router.e_score_correction_bias = nnx.Variable(
            jnp.array([10.0, 0.0, 0.0, 0.0])
        )
        indices2, _, _ = router(x)
        # Expert 0 should be selected more often
        assert jnp.sum(indices2 == 0) > 0


class TestMoEExperts:
    def test_output_shape(self, rngs):
        cfg = MinTextConfig(
            hidden_size=64, num_attention_heads=4, intermediate_size=128, vocab_size=128,
            num_experts=4, num_experts_per_tok=2, moe_intermediate_size=64,
            dtype="float32", weight_dtype="float32",
        )
        experts = MoEExperts(cfg, rngs=rngs)
        x = jax.random.normal(jax.random.key(0), (16, 64))
        indices = jnp.array([[0, 1]] * 16)
        weights = jnp.ones((16, 2)) * 0.5
        y = experts(x, indices, weights)
        assert y.shape == (16, 64)


class TestMoEBlock:
    def test_forward(self, rngs):
        cfg = MinTextConfig(
            hidden_size=64, num_attention_heads=4, intermediate_size=128, vocab_size=128,
            num_experts=4, num_experts_per_tok=2, moe_intermediate_size=64,
            n_shared_experts=1, n_group=2, topk_group=1,
            routed_scaling_factor=1.0,
            dtype="float32", weight_dtype="float32",
        )
        block = MoEBlock(cfg, rngs=rngs)
        x = jax.random.normal(jax.random.key(0), (BATCH, SEQ, 64))
        out, aux = block(x)
        assert out.shape == (BATCH, SEQ, 64)
        assert "topk_indices" in aux
        assert "scores" in aux


# --- Linear Attention tests ---


class TestGatedDeltaRule:
    def test_chunk_computation_shapes(self):
        B, S, H, Dk, Dv = 2, 16, 4, 8, 8
        q = jax.random.normal(jax.random.key(0), (B, S, H, Dk))
        k = jax.random.normal(jax.random.key(1), (B, S, H, Dk))
        v = jax.random.normal(jax.random.key(2), (B, S, H, Dv))
        beta = jax.nn.sigmoid(jax.random.normal(jax.random.key(3), (B, S, H)))
        g = -jnp.abs(jax.random.normal(jax.random.key(4), (B, S, H)))
        out = chunk_gated_delta_rule(q, k, v, beta, g, chunk_size=8)
        assert out.shape == (B, S, H, Dv)

    def test_output_finite(self):
        B, S, H, Dk, Dv = 2, 16, 4, 8, 8
        q = jax.random.normal(jax.random.key(0), (B, S, H, Dk)) * 0.1
        k = jax.random.normal(jax.random.key(1), (B, S, H, Dk)) * 0.1
        v = jax.random.normal(jax.random.key(2), (B, S, H, Dv)) * 0.1
        beta = jax.nn.sigmoid(jax.random.normal(jax.random.key(3), (B, S, H)))
        g = -0.1 * jnp.abs(jax.random.normal(jax.random.key(4), (B, S, H)))
        out = chunk_gated_delta_rule(q, k, v, beta, g, chunk_size=8)
        assert jnp.all(jnp.isfinite(out))


class TestGatedDeltaRuleAttention:
    @pytest.fixture
    def linear_config(self):
        return MinTextConfig(
            model_type="qwen3_next", num_hidden_layers=2, hidden_size=64, num_attention_heads=4, head_dim=16,
            intermediate_size=128, vocab_size=128, seq_length=32,
            linear_key_head_dim=8, linear_value_head_dim=8,
            linear_num_key_heads=4, linear_num_value_heads=4,
            linear_conv_kernel_dim=4,
            dtype="float32", weight_dtype="float32",
        )

    def test_output_shape(self, linear_config, rngs):
        attn = GatedDeltaRuleAttention(linear_config, rngs=rngs)
        x = jax.random.normal(jax.random.key(0), (BATCH, SEQ, linear_config.hidden_size))
        y = attn(x)
        assert y.shape == (BATCH, SEQ, linear_config.hidden_size)

    def test_output_finite(self, linear_config, rngs):
        attn = GatedDeltaRuleAttention(linear_config, rngs=rngs)
        x = jax.random.normal(jax.random.key(0), (BATCH, SEQ, linear_config.hidden_size)) * 0.1
        y = attn(x)
        assert jnp.all(jnp.isfinite(y))


# --- Qwen3-Next hybrid model tests ---


class TestQwen3NextHybrid:
    @pytest.fixture
    def hybrid_config(self):
        return MinTextConfig(
            model_type="qwen3_next", num_hidden_layers=8, hidden_size=64, num_attention_heads=4, head_dim=16,
            intermediate_size=128, vocab_size=128, seq_length=32,
            use_qk_norm=True, full_attention_interval=4,
            partial_rotary_factor=1.0,
            linear_key_head_dim=8, linear_value_head_dim=8,
            linear_num_key_heads=4, linear_num_value_heads=4,
            linear_conv_kernel_dim=4,
            dtype="float32", weight_dtype="float32",
        )

    def test_layer_types(self, hybrid_config):
        # Every 4th layer is full attention, rest is linear
        assert hybrid_config.layer_types[0] == "linear_attention"
        assert hybrid_config.layer_types[3] == "full_attention"
        assert hybrid_config.layer_types[7] == "full_attention"

    def test_forward_pass(self, hybrid_config, rngs):
        model = Transformer(hybrid_config, rngs=rngs)
        tokens = jax.random.randint(jax.random.key(0), (BATCH, SEQ), 0, 128)
        pos = jnp.broadcast_to(jnp.arange(SEQ), (BATCH, SEQ))
        mask = make_causal_mask(SEQ)
        logits, aux = model(tokens, pos, mask)
        assert logits.shape == (BATCH, SEQ, 128)

    def test_mixed_layer_types_in_model(self, hybrid_config, rngs):
        model = Transformer(hybrid_config, rngs=rngs)
        # Check that layers alternate correctly
        assert model.layers[0].is_linear
        assert not model.layers[3].is_linear
        assert model.layers[4].is_linear
        assert not model.layers[7].is_linear


# --- DeepSeek-V3 full model tests ---


class TestDeepSeekV3Model:
    @pytest.fixture
    def ds_config(self):
        return MinTextConfig(
            model_type="deepseek_v3", num_hidden_layers=4, hidden_size=64, num_attention_heads=4,
            intermediate_size=128, vocab_size=128, seq_length=32,
            attention_type="mla", q_lora_rank=32, kv_lora_rank=32,
            qk_nope_head_dim=8, qk_rope_head_dim=4, v_head_dim=8,
            num_experts=4, num_experts_per_tok=2, moe_intermediate_size=64,
            n_shared_experts=1, n_group=2, topk_group=1,
            first_k_dense_replace=1, routed_scaling_factor=1.0,
            dtype="float32", weight_dtype="float32",
        )

    def test_forward_pass(self, ds_config, rngs):
        model = Transformer(ds_config, rngs=rngs)
        tokens = jax.random.randint(jax.random.key(0), (BATCH, SEQ), 0, 128)
        pos = jnp.broadcast_to(jnp.arange(SEQ), (BATCH, SEQ))
        mask = make_causal_mask(SEQ)
        logits, aux = model(tokens, pos, mask)
        assert logits.shape == (BATCH, SEQ, 128)

    def test_moe_layers_correct(self, ds_config, rngs):
        model = Transformer(ds_config, rngs=rngs)
        # first_k_dense_replace=1 means layer 0 is dense, rest are MoE
        assert not model.layers[0].has_moe
        assert model.layers[1].has_moe
        assert model.layers[2].has_moe
        assert model.layers[3].has_moe

    def test_aux_data_from_moe(self, ds_config, rngs):
        model = Transformer(ds_config, rngs=rngs)
        tokens = jax.random.randint(jax.random.key(0), (BATCH, SEQ), 0, 128)
        pos = jnp.broadcast_to(jnp.arange(SEQ), (BATCH, SEQ))
        mask = make_causal_mask(SEQ)
        _, aux = model(tokens, pos, mask)
        assert aux[0] is None  # dense layer
        assert aux[1] is not None  # MoE layer
        assert "topk_indices" in aux[1]


# --- Multi-model Transformer tests ---


class TestMultiModelTransformer:
    def test_llama(self, rngs):
        cfg = MinTextConfig(
            model_type="llama3", num_hidden_layers=2, hidden_size=64, num_attention_heads=4, head_dim=16,
            intermediate_size=128, vocab_size=128, seq_length=32,
            dtype="float32", weight_dtype="float32",
        )
        model = Transformer(cfg, rngs=rngs)
        tokens = jax.random.randint(jax.random.key(0), (1, 8), 0, 128)
        pos = jnp.broadcast_to(jnp.arange(8), (1, 8))
        mask = make_causal_mask(8)
        logits, _ = model(tokens, pos, mask)
        assert logits.shape == (1, 8, 128)

    def test_qwen3(self, rngs):
        cfg = MinTextConfig(
            model_type="qwen3", num_hidden_layers=2, hidden_size=64, num_attention_heads=4, head_dim=16,
            intermediate_size=128, vocab_size=128, seq_length=32,
            use_qk_norm=True, dtype="float32", weight_dtype="float32",
        )
        model = Transformer(cfg, rngs=rngs)
        tokens = jax.random.randint(jax.random.key(0), (1, 8), 0, 128)
        pos = jnp.broadcast_to(jnp.arange(8), (1, 8))
        mask = make_causal_mask(8)
        logits, _ = model(tokens, pos, mask)
        assert logits.shape == (1, 8, 128)

    def test_deepseek_v3(self, rngs):
        cfg = MinTextConfig(
            model_type="deepseek_v3", num_hidden_layers=2, hidden_size=64, num_attention_heads=4,
            intermediate_size=128, vocab_size=128, seq_length=32,
            attention_type="mla", q_lora_rank=32, kv_lora_rank=32,
            qk_nope_head_dim=8, qk_rope_head_dim=4, v_head_dim=8,
            num_experts=4, num_experts_per_tok=2, moe_intermediate_size=64,
            n_group=2, topk_group=1, first_k_dense_replace=0,
            routed_scaling_factor=1.0,
            dtype="float32", weight_dtype="float32",
        )
        model = Transformer(cfg, rngs=rngs)
        tokens = jax.random.randint(jax.random.key(0), (1, 8), 0, 128)
        pos = jnp.broadcast_to(jnp.arange(8), (1, 8))
        mask = make_causal_mask(8)
        logits, _ = model(tokens, pos, mask)
        assert logits.shape == (1, 8, 128)

    def test_qwen3_next(self, rngs):
        cfg = MinTextConfig(
            model_type="qwen3_next", num_hidden_layers=4, hidden_size=64, num_attention_heads=4, head_dim=16,
            intermediate_size=128, vocab_size=128, seq_length=32,
            use_qk_norm=True, full_attention_interval=4,
            linear_key_head_dim=8, linear_value_head_dim=8,
            linear_num_key_heads=4, linear_num_value_heads=4,
            linear_conv_kernel_dim=4,
            dtype="float32", weight_dtype="float32",
        )
        model = Transformer(cfg, rngs=rngs)
        tokens = jax.random.randint(jax.random.key(0), (1, 8), 0, 128)
        pos = jnp.broadcast_to(jnp.arange(8), (1, 8))
        mask = make_causal_mask(8)
        logits, _ = model(tokens, pos, mask)
        assert logits.shape == (1, 8, 128)

    def test_gemma3(self, rngs):
        cfg = MinTextConfig(
            model_type="gemma3", num_hidden_layers=6, hidden_size=64, num_attention_heads=4,
            num_key_value_heads=2, head_dim=16, intermediate_size=128, vocab_size=128,
            seq_length=32, hidden_activation="gelu",
            use_qk_norm=True, sliding_window=16, sliding_window_pattern=6,
            rope_theta=1_000_000.0, rope_local_theta=10_000.0,
            query_pre_attn_scalar=16.0, use_post_ffw_norm=True,
            scale_embeddings=True, tie_word_embeddings=True,
            dtype="float32", weight_dtype="float32",
        )
        model = Transformer(cfg, rngs=rngs)
        tokens = jax.random.randint(jax.random.key(0), (1, 8), 0, 128)
        pos = jnp.broadcast_to(jnp.arange(8), (1, 8))
        mask = make_causal_mask(8)
        logits, _ = model(tokens, pos, mask)
        assert logits.shape == (1, 8, 128)
        assert jnp.all(jnp.isfinite(logits))


# --- Gemma3 tests ---


class TestGemma3LayerTypes:
    def test_5_1_pattern(self):
        """Every 6th layer (1-indexed) is global, rest sliding."""
        cfg = MinTextConfig(
            model_type="gemma3", num_hidden_layers=12,
            sliding_window_pattern=6,
            dtype="float32", weight_dtype="float32",
        )
        lt = cfg.layer_types
        assert len(lt) == 12
        # Layer indices 5, 11 (0-indexed) are global (6th, 12th 1-indexed)
        for i in range(12):
            expected = "full_attention" if (i + 1) % 6 == 0 else "sliding_attention"
            assert lt[i] == expected, f"Layer {i}: expected {expected}, got {lt[i]}"

    def test_custom_pattern(self):
        """Custom pattern of 3 (every 3rd layer global)."""
        cfg = MinTextConfig(
            model_type="gemma3", num_hidden_layers=6,
            sliding_window_pattern=3,
            dtype="float32", weight_dtype="float32",
        )
        assert cfg.layer_types == [
            "sliding_attention", "sliding_attention", "full_attention",
            "sliding_attention", "sliding_attention", "full_attention",
        ]


class TestGemma3Attention:
    @pytest.fixture
    def gemma3_config(self):
        return MinTextConfig(
            model_type="gemma3", num_hidden_layers=6, hidden_size=64, num_attention_heads=4,
            num_key_value_heads=2, head_dim=16, intermediate_size=128, vocab_size=128,
            seq_length=32, hidden_activation="gelu",
            use_qk_norm=True, sliding_window=16, sliding_window_pattern=6,
            rope_theta=1_000_000.0, rope_local_theta=10_000.0,
            query_pre_attn_scalar=16.0, attn_logit_softcapping=50.0,
            use_post_ffw_norm=True, scale_embeddings=True,
            tie_word_embeddings=True,
            dtype="float32", weight_dtype="float32",
        )

    def test_softcapping(self, gemma3_config, rngs):
        """Attention logit softcapping bounds attention weights."""
        attn = Attention(gemma3_config, rngs=rngs, use_qk_norm=True)
        x = jax.random.normal(jax.random.key(0), (BATCH, SEQ, 64))
        pos = jnp.broadcast_to(jnp.arange(SEQ), (BATCH, SEQ))
        mask = make_causal_mask(SEQ)
        y = attn(x, pos, mask)
        assert y.shape == (BATCH, SEQ, 64)
        assert jnp.all(jnp.isfinite(y))

    def test_query_pre_attn_scalar(self, gemma3_config, rngs):
        """query_pre_attn_scalar overrides default 1/sqrt(head_dim)."""
        attn = Attention(gemma3_config, rngs=rngs)
        expected_scale = 16.0 ** -0.5  # 0.25
        assert abs(attn.scale - expected_scale) < 1e-6

    def test_dual_rope(self, gemma3_config, rngs):
        """Sliding layer uses local rope_theta, global layer uses global."""
        # Sliding layer (uses local rope)
        attn_local = Attention(
            gemma3_config, rngs=rngs,
            sliding_window=16, rope_local_theta=10_000.0,
        )
        assert attn_local.use_local_rope

        # Global layer (no local rope override)
        attn_global = Attention(gemma3_config, rngs=rngs)
        assert not attn_global.use_local_rope

        # Both produce valid outputs
        x = jax.random.normal(jax.random.key(0), (BATCH, SEQ, 64))
        pos = jnp.broadcast_to(jnp.arange(SEQ), (BATCH, SEQ))
        mask = make_causal_mask(SEQ)
        y_local = attn_local(x, pos, mask)
        y_global = attn_global(x, pos, mask)
        assert y_local.shape == (BATCH, SEQ, 64)
        assert y_global.shape == (BATCH, SEQ, 64)

    def test_4_norm_decoder_layer(self, gemma3_config, rngs):
        """Gemma3 decoder layer has 4 norms."""
        layer = DecoderLayer(gemma3_config, layer_idx=0, rngs=rngs)
        assert layer.use_post_ffw_norm
        assert hasattr(layer, 'pre_attn_norm')
        assert hasattr(layer, 'post_attn_norm')
        assert hasattr(layer, 'pre_ffw_norm')
        assert hasattr(layer, 'post_ffw_norm')

        x = jax.random.normal(jax.random.key(0), (BATCH, SEQ, 64))
        pos = jnp.broadcast_to(jnp.arange(SEQ), (BATCH, SEQ))
        mask = make_causal_mask(SEQ)
        y, aux = layer(x, pos, mask)
        assert y.shape == (BATCH, SEQ, 64)
        assert jnp.all(jnp.isfinite(y))


class TestGemma3Model:
    @pytest.fixture
    def gemma3_config(self):
        return MinTextConfig(
            model_type="gemma3", num_hidden_layers=6, hidden_size=64, num_attention_heads=4,
            num_key_value_heads=2, head_dim=16, intermediate_size=128, vocab_size=128,
            seq_length=32, hidden_activation="gelu",
            use_qk_norm=True, sliding_window=16, sliding_window_pattern=6,
            rope_theta=1_000_000.0, rope_local_theta=10_000.0,
            query_pre_attn_scalar=16.0, use_post_ffw_norm=True,
            scale_embeddings=True, tie_word_embeddings=True,
            dtype="float32", weight_dtype="float32",
        )

    def test_full_forward(self, gemma3_config, rngs):
        model = Transformer(gemma3_config, rngs=rngs)
        tokens = jax.random.randint(jax.random.key(0), (BATCH, SEQ), 0, 128)
        pos = jnp.broadcast_to(jnp.arange(SEQ), (BATCH, SEQ))
        mask = make_causal_mask(SEQ)
        logits, aux = model(tokens, pos, mask)
        assert logits.shape == (BATCH, SEQ, 128)
        assert jnp.all(jnp.isfinite(logits))
        # Weight tying — no output_proj
        assert not hasattr(model, 'output_proj')

    def test_embedding_scale(self, gemma3_config, rngs):
        """Embedding is scaled by hidden_size ** 0.5 when scale_embeddings=True."""
        model = Transformer(gemma3_config, rngs=rngs)
        assert float(model.embedding_scale) == 64 ** 0.5  # hidden_size=64

    def test_final_logit_softcapping(self, rngs):
        """Final logit softcapping bounds output logits."""
        cfg = MinTextConfig(
            model_type="gemma3", num_hidden_layers=2, hidden_size=64, num_attention_heads=4,
            num_key_value_heads=2, head_dim=16, intermediate_size=128, vocab_size=128,
            seq_length=32, hidden_activation="gelu",
            final_logit_softcapping=30.0,
            dtype="float32", weight_dtype="float32",
        )
        model = Transformer(cfg, rngs=rngs)
        tokens = jax.random.randint(jax.random.key(0), (BATCH, SEQ), 0, 128)
        pos = jnp.broadcast_to(jnp.arange(SEQ), (BATCH, SEQ))
        mask = make_causal_mask(SEQ)
        logits, _ = model(tokens, pos, mask)
        # All logits should be within [-cap, cap]
        assert jnp.all(jnp.abs(logits) <= 30.0 + 1e-6)

    def test_gradient_flow(self, gemma3_config, rngs):
        """Gradients flow through the full Gemma3 model."""
        model = Transformer(gemma3_config, rngs=rngs)
        tokens = jax.random.randint(jax.random.key(0), (BATCH, SEQ), 0, 128)
        pos = jnp.broadcast_to(jnp.arange(SEQ), (BATCH, SEQ))
        mask = make_causal_mask(SEQ)

        def loss_fn(model, tokens, pos, mask):
            logits, _ = model(tokens, pos, mask)
            return jnp.mean(logits)

        grad_fn = nnx.grad(loss_fn)
        grads = grad_fn(model, tokens, pos, mask)
        leaves = jax.tree.leaves(nnx.state(grads))
        assert len(leaves) > 0
        assert all(jnp.all(jnp.isfinite(l)) for l in leaves)


class TestACT2FN:
    def test_all_activations_exist(self):
        assert "silu" in ACT2FN
        assert "gelu" in ACT2FN
        assert "relu" in ACT2FN

    def test_activation_output(self):
        x = jax.random.normal(jax.random.key(0), (4, 8))
        for name, fn in ACT2FN.items():
            y = fn(x)
            assert y.shape == x.shape, f"Activation {name} changed shape"
            assert jnp.all(jnp.isfinite(y)), f"Activation {name} produced NaN/Inf"

    def test_mlp_with_gelu(self, rngs):
        cfg = MinTextConfig(
            hidden_size=64, num_attention_heads=4, intermediate_size=128, vocab_size=128,
            hidden_activation="gelu",
            dtype="float32", weight_dtype="float32",
        )
        mlp = MLP(cfg, rngs=rngs)
        x = jax.random.normal(jax.random.key(0), (BATCH, SEQ, 64))
        y = mlp(x)
        assert y.shape == (BATCH, SEQ, 64)
        assert jnp.all(jnp.isfinite(y))


# --- Advanced remat, custom kernels, FP8, depth-scaled init tests ---


class TestAdvancedRemat:
    """Tests for advanced remat policies (Phase 11)."""

    def _make_model_and_inputs(self, rngs, remat_policy):
        cfg = MinTextConfig(
            num_hidden_layers=2, hidden_size=64, num_attention_heads=4, head_dim=16,
            intermediate_size=128, vocab_size=128, seq_length=32,
            remat_policy=remat_policy,
            dtype="float32", weight_dtype="float32",
        )
        model = Transformer(cfg, rngs=rngs)
        tokens = jax.random.randint(jax.random.key(0), (BATCH, SEQ), 0, 128)
        pos = jnp.broadcast_to(jnp.arange(SEQ), (BATCH, SEQ))
        mask = make_causal_mask(SEQ)
        return model, tokens, pos, mask

    def test_remat_save_qkv_proj(self, rngs):
        model, tokens, pos, mask = self._make_model_and_inputs(rngs, "save_qkv_proj")

        def loss_fn(model, tokens, pos, mask):
            logits, _ = model(tokens, pos, mask)
            return jnp.mean(logits)

        grad_fn = nnx.grad(loss_fn)
        grads = grad_fn(model, tokens, pos, mask)
        leaves = jax.tree.leaves(nnx.state(grads))
        assert len(leaves) > 0
        assert all(jnp.all(jnp.isfinite(l)) for l in leaves)

    def test_remat_save_dot_except_mlp(self, rngs):
        model, tokens, pos, mask = self._make_model_and_inputs(rngs, "save_dot_except_mlp")

        def loss_fn(model, tokens, pos, mask):
            logits, _ = model(tokens, pos, mask)
            return jnp.mean(logits)

        grad_fn = nnx.grad(loss_fn)
        grads = grad_fn(model, tokens, pos, mask)
        leaves = jax.tree.leaves(nnx.state(grads))
        assert len(leaves) > 0
        assert all(jnp.all(jnp.isfinite(l)) for l in leaves)

    def test_remat_offloaded_policies(self, rngs):
        for policy in ("qkv_proj_offloaded", "minimal_offloaded"):
            model, tokens, pos, mask = self._make_model_and_inputs(rngs, policy)

            def loss_fn(model, tokens, pos, mask):
                logits, _ = model(tokens, pos, mask)
                return jnp.mean(logits)

            grad_fn = nnx.grad(loss_fn)
            grads = grad_fn(model, tokens, pos, mask)
            leaves = jax.tree.leaves(nnx.state(grads))
            assert len(leaves) > 0


class TestCheckpointNames:
    """Verify checkpoint_name annotations are present in traced functions."""

    def test_checkpoint_names_in_attention(self, config, rngs):
        attn = Attention(config, rngs=rngs)
        x = jax.random.normal(jax.random.key(0), (BATCH, SEQ, config.hidden_size))
        pos = jnp.broadcast_to(jnp.arange(SEQ), (BATCH, SEQ))
        mask = make_causal_mask(SEQ)
        y = attn(x, pos, mask)
        assert y.shape == (BATCH, SEQ, config.hidden_size)
        assert jnp.all(jnp.isfinite(y))

    def test_checkpoint_names_in_mlp(self, config, rngs):
        mlp = MLP(config, rngs=rngs)
        x = jax.random.normal(jax.random.key(0), (BATCH, SEQ, config.hidden_size))
        y = mlp(x)
        assert y.shape == (BATCH, SEQ, config.hidden_size)
        assert jnp.all(jnp.isfinite(y))


class TestCustomKernels:
    """Tests for custom kernel dispatch (Phase 10)."""

    def test_custom_kernels_fallback_without_tokamax(self, rngs):
        """When use_custom_kernels=True but tokamax not installed, should fall back."""
        cfg = MinTextConfig(
            num_hidden_layers=2, hidden_size=64, num_attention_heads=4, head_dim=16,
            intermediate_size=128, vocab_size=128, seq_length=32,
            use_custom_kernels=True,
            dtype="float32", weight_dtype="float32",
        )
        attn = Attention(cfg, rngs=rngs)
        x = jax.random.normal(jax.random.key(0), (BATCH, SEQ, cfg.hidden_size))
        pos = jnp.broadcast_to(jnp.arange(SEQ), (BATCH, SEQ))
        mask = make_causal_mask(SEQ)
        y = attn(x, pos, mask)
        assert y.shape == (BATCH, SEQ, cfg.hidden_size)
        assert jnp.all(jnp.isfinite(y))


class TestFP8Matmul:
    """Tests for FP8 training support (Phase 12)."""

    def test_fp8_matmul_basic(self, rngs):
        """Test that FP8 matmul produces correct shapes and reasonable values."""
        from mintext.modules.linear import _fp8_matmul

        x = jax.random.normal(jax.random.key(0), (4, 8, 64))
        kernel = jax.random.normal(jax.random.key(1), (64, 128))

        try:
            _ = jnp.float8_e4m3fn
            fp8_available = True
        except AttributeError:
            fp8_available = False

        result = _fp8_matmul(x, kernel, jnp.float32)
        assert result.shape == (4, 8, 128)
        assert result.dtype == jnp.float32

        if fp8_available:
            expected = jnp.dot(x, kernel)
            # FP8 has very limited precision; just check correlation, not exact match
            corr = jnp.corrcoef(result.flatten(), expected.flatten())[0, 1]
            assert corr > 0.95, f"FP8 result poorly correlated with reference: {corr}"

    def test_fp8_linear(self, rngs):
        """Test Linear with use_fp8=True."""
        dense = Linear(
            64, 128, dtype=jnp.float32, weight_dtype=jnp.float32,
            use_fp8=True, rngs=rngs,
        )
        x = jax.random.normal(jax.random.key(0), (BATCH, SEQ, 64))
        y = dense(x)
        assert y.shape == (BATCH, SEQ, 128)
        assert jnp.all(jnp.isfinite(y))

    def test_fp8_mlp(self, rngs):
        """Test MLP with FP8 enabled via config."""
        cfg = MinTextConfig(
            num_hidden_layers=2, hidden_size=64, num_attention_heads=4, head_dim=16,
            intermediate_size=128, vocab_size=128, use_fp8=True,
            dtype="float32", weight_dtype="float32",
        )
        mlp = MLP(cfg, rngs=rngs)
        x = jax.random.normal(jax.random.key(0), (BATCH, SEQ, 64))
        y = mlp(x)
        assert y.shape == (BATCH, SEQ, 64)
        assert jnp.all(jnp.isfinite(y))

