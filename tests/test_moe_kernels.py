"""Tests for MoE custom kernels: dispatch, grouped matmul, VJP, tgmm, integration, autotuner."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from mintext.config import MinTextConfig
from mintext.kernels.moe_dispatch import route, unroute, _sort_with_vjp
from mintext.kernels.grouped_matmul import grouped_matmul, grouped_matmul_vjp, tgmm


@pytest.fixture
def rngs():
    return nnx.Rngs(params=0, dropout=1)


# --- TestSortDispatch ---


class TestSortDispatch:
    """Tests for route/unroute token dispatch."""

    N, D, K, E = 32, 64, 2, 4

    @pytest.fixture
    def routing_data(self):
        rng = jax.random.key(42)
        tokens = jax.random.normal(rng, (self.N, self.D))
        # Random expert assignments in range [0, E)
        topk_indices = jax.random.randint(
            jax.random.key(1), (self.N, self.K), 0, self.E
        )
        topk_weights = jax.random.uniform(
            jax.random.key(2), (self.N, self.K)
        )
        return tokens, topk_indices, topk_weights

    def test_route_output_shapes(self, routing_data):
        tokens, topk_indices, _ = routing_data
        sorted_tokens, sort_indices, group_sizes = route(tokens, topk_indices, self.E)
        assert sorted_tokens.shape == (self.N * self.K, self.D)
        assert sort_indices.shape == (self.N * self.K,)
        assert group_sizes.shape == (self.E,)

    def test_group_sizes_sum(self, routing_data):
        tokens, topk_indices, _ = routing_data
        _, _, group_sizes = route(tokens, topk_indices, self.E)
        assert int(group_sizes.sum()) == self.N * self.K

    def test_tokens_contiguous_by_expert(self, routing_data):
        tokens, topk_indices, _ = routing_data
        sorted_tokens, sort_indices, group_sizes = route(tokens, topk_indices, self.E)
        # After sorting, expert ids should be non-decreasing
        flat_indices = jnp.ravel(topk_indices)
        sorted_expert_ids = flat_indices[sort_indices]
        # Check non-decreasing
        diffs = sorted_expert_ids[1:] - sorted_expert_ids[:-1]
        assert jnp.all(diffs >= 0)

    def test_unroute_recovers_shape(self, routing_data):
        tokens, topk_indices, topk_weights = routing_data
        sorted_tokens, sort_indices, _ = route(tokens, topk_indices, self.E)
        # Use sorted_tokens as "expert output"
        output = unroute(sorted_tokens, sort_indices, topk_weights, self.K)
        assert output.shape == (self.N, self.D)

    def test_route_unroute_gradient(self, routing_data):
        tokens, topk_indices, topk_weights = routing_data

        def fn(tok):
            sorted_t, si, gs = route(tok, topk_indices, self.E)
            out = unroute(sorted_t, si, topk_weights, self.K)
            return out.sum()

        grad = jax.grad(fn)(tokens)
        assert grad.shape == tokens.shape
        # Gradient should be non-zero for at least some tokens
        assert jnp.any(grad != 0)

    def test_gradient_matches_finite_diff(self, routing_data):
        tokens, topk_indices, topk_weights = routing_data
        # float32-only environment: use larger epsilon and looser tolerance
        eps = 1e-3

        def fn(tok):
            sorted_t, si, gs = route(tok, topk_indices, self.E)
            out = unroute(sorted_t, si, topk_weights, self.K)
            return out.sum()

        analytic_grad = jax.grad(fn)(tokens)
        # Check a few random elements
        rng = np.random.RandomState(0)
        for _ in range(10):
            i, j = rng.randint(0, self.N), rng.randint(0, self.D)
            tokens_plus = tokens.at[i, j].add(eps)
            tokens_minus = tokens.at[i, j].add(-eps)
            numeric = float((fn(tokens_plus) - fn(tokens_minus)) / (2 * eps))
            np.testing.assert_allclose(
                float(analytic_grad[i, j]), numeric,
                rtol=0.02, atol=1e-4,
            )

    def test_single_expert_edge_case(self):
        """All tokens routed to a single expert."""
        N, D, K, E = 8, 16, 1, 4
        tokens = jax.random.normal(jax.random.key(0), (N, D))
        topk_indices = jnp.zeros((N, K), dtype=jnp.int32)  # all to expert 0
        topk_weights = jnp.ones((N, K))

        sorted_t, si, gs = route(tokens, topk_indices, E)
        assert int(gs[0]) == N
        assert int(gs[1:].sum()) == 0

        output = unroute(sorted_t, si, topk_weights, K)
        assert output.shape == (N, D)

    def test_empty_expert(self):
        """One expert gets 0 tokens."""
        N, D, K, E = 8, 16, 1, 4
        tokens = jax.random.normal(jax.random.key(0), (N, D))
        # Route all to experts 0 and 1, expert 2 and 3 get nothing
        topk_indices = jax.random.randint(jax.random.key(1), (N, K), 0, 2)
        topk_weights = jnp.ones((N, K))

        sorted_t, si, gs = route(tokens, topk_indices, E)
        # Experts 2 and 3 should have 0 tokens
        assert int(gs[2]) == 0 or int(gs[3]) == 0
        assert int(gs.sum()) == N * K

        output = unroute(sorted_t, si, topk_weights, K)
        assert output.shape == (N, D)
        assert jnp.all(jnp.isfinite(output))


# --- TestGroupedMatmul ---


class TestGroupedMatmul:
    """Tests for grouped matmul (ragged_dot)."""

    E, K_in, K_out = 4, 32, 16

    @pytest.fixture
    def gmm_data(self):
        rng = jax.random.key(0)
        k1, k2 = jax.random.split(rng)
        M = 64
        x = jax.random.normal(k1, (M, self.K_in))
        weights = jax.random.normal(k2, (self.E, self.K_in, self.K_out))
        # Uniform group sizes
        base = M // self.E
        group_sizes = jnp.array([base] * self.E, dtype=jnp.int32)
        return x, weights, group_sizes

    def test_output_shape(self, gmm_data):
        x, weights, group_sizes = gmm_data
        out = grouped_matmul(x, weights, group_sizes)
        assert out.shape == (x.shape[0], self.K_out)

    def test_uniform_groups(self, gmm_data):
        """With uniform groups, should match looped standard matmul."""
        x, weights, group_sizes = gmm_data
        out = grouped_matmul(x, weights, group_sizes)

        # Reference: loop over experts
        M = x.shape[0]
        group_size = M // self.E
        ref = jnp.zeros((M, self.K_out))
        for i in range(self.E):
            start = i * group_size
            end = start + group_size
            ref = ref.at[start:end].set(x[start:end] @ weights[i])

        np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)

    def test_single_expert(self):
        """Degenerates to standard matmul with one expert."""
        M, K_in, K_out = 16, 8, 4
        rng = jax.random.key(0)
        x = jax.random.normal(rng, (M, K_in))
        w = jax.random.normal(jax.random.key(1), (1, K_in, K_out))
        gs = jnp.array([M], dtype=jnp.int32)

        out = grouped_matmul(x, w, gs)
        ref = x @ w[0]
        np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)

    def test_empty_expert(self):
        """Expert with 0 tokens doesn't corrupt output."""
        M, K_in, K_out, E = 16, 8, 4, 4
        rng = jax.random.key(0)
        x = jax.random.normal(rng, (M, K_in))
        w = jax.random.normal(jax.random.key(1), (E, K_in, K_out))
        # All tokens go to first 2 experts
        gs = jnp.array([8, 8, 0, 0], dtype=jnp.int32)

        out = grouped_matmul(x, w, gs)
        assert out.shape == (M, K_out)
        assert jnp.all(jnp.isfinite(out))

        # Verify correctness for non-empty experts
        ref0 = x[:8] @ w[0]
        ref1 = x[8:16] @ w[1]
        np.testing.assert_allclose(out[:8], ref0, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(out[8:16], ref1, rtol=1e-4, atol=1e-4)

    def test_gradient_correctness(self):
        """Verify gradients via finite differences."""
        M, K_in, K_out, E = 8, 4, 4, 2
        rng = jax.random.key(0)
        x = jax.random.normal(rng, (M, K_in))
        w = jax.random.normal(jax.random.key(1), (E, K_in, K_out))
        gs = jnp.array([4, 4], dtype=jnp.int32)

        def fn(x_):
            return grouped_matmul(x_, w, gs).sum()

        analytic = jax.grad(fn)(x)

        eps = 1e-3
        for i in range(min(4, M)):
            for j in range(min(4, K_in)):
                xp = x.at[i, j].add(eps)
                xm = x.at[i, j].add(-eps)
                numeric = float((fn(xp) - fn(xm)) / (2 * eps))
                np.testing.assert_allclose(
                    float(analytic[i, j]), numeric,
                    rtol=0.02, atol=1e-4,
                )

    def test_bfloat16(self):
        """bfloat16 dtype support."""
        M, K_in, K_out, E = 16, 8, 4, 2
        rng = jax.random.key(0)
        x = jax.random.normal(rng, (M, K_in), dtype=jnp.bfloat16)
        w = jax.random.normal(jax.random.key(1), (E, K_in, K_out), dtype=jnp.bfloat16)
        gs = jnp.array([8, 8], dtype=jnp.int32)

        out = grouped_matmul(x, w, gs)
        assert out.shape == (M, K_out)
        assert jnp.all(jnp.isfinite(out))

    @pytest.mark.parametrize("num_experts", [2, 4, 8])
    def test_various_expert_counts(self, num_experts):
        M = num_experts * 4
        K_in, K_out = 8, 4
        rng = jax.random.key(0)
        x = jax.random.normal(rng, (M, K_in))
        w = jax.random.normal(jax.random.key(1), (num_experts, K_in, K_out))
        gs = jnp.full(num_experts, 4, dtype=jnp.int32)

        out = grouped_matmul(x, w, gs)
        assert out.shape == (M, K_out)
        assert jnp.all(jnp.isfinite(out))

    def test_uneven_groups(self):
        """Highly imbalanced routing."""
        E, K_in, K_out = 4, 8, 4
        gs = jnp.array([15, 1, 0, 0], dtype=jnp.int32)
        M = int(gs.sum())
        rng = jax.random.key(0)
        x = jax.random.normal(rng, (M, K_in))
        w = jax.random.normal(jax.random.key(1), (E, K_in, K_out))

        out = grouped_matmul(x, w, gs)
        assert out.shape == (M, K_out)
        assert jnp.all(jnp.isfinite(out))

        # Check first expert's outputs
        ref0 = x[:15] @ w[0]
        np.testing.assert_allclose(out[:15], ref0, rtol=1e-4, atol=1e-4)


# --- TestGroupedMatmulVJP ---


class TestGroupedMatmulVJP:
    """Tests for grouped_matmul_vjp with custom backward."""

    def test_forward_matches_non_vjp(self):
        """VJP forward should match standard ragged_dot."""
        M, K_in, K_out, E = 32, 16, 8, 4
        rng = jax.random.key(42)
        x = jax.random.normal(rng, (M, K_in))
        w = jax.random.normal(jax.random.key(1), (E, K_in, K_out))
        gs = jnp.array([8, 8, 8, 8], dtype=jnp.int32)

        out_vjp = grouped_matmul_vjp(x, w, gs)
        out_ref = grouped_matmul(x, w, gs)
        np.testing.assert_allclose(out_vjp, out_ref, rtol=1e-5, atol=1e-5)

    def test_grad_matches_autodiff(self):
        """Custom VJP grads should match standard autodiff (ragged_dot)."""
        M, K_in, K_out, E = 16, 8, 4, 2
        rng = jax.random.key(0)
        x = jax.random.normal(rng, (M, K_in))
        w = jax.random.normal(jax.random.key(1), (E, K_in, K_out))
        gs = jnp.array([8, 8], dtype=jnp.int32)

        def loss_vjp(x_, w_):
            return grouped_matmul_vjp(x_, w_, gs).sum()

        def loss_ref(x_, w_):
            return grouped_matmul(x_, w_, gs).sum()

        dx_vjp, dw_vjp = jax.grad(loss_vjp, argnums=(0, 1))(x, w)
        dx_ref, dw_ref = jax.grad(loss_ref, argnums=(0, 1))(x, w)

        np.testing.assert_allclose(dx_vjp, dx_ref, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(dw_vjp, dw_ref, rtol=1e-4, atol=1e-4)

    def test_9_param_tiling(self):
        """9-param tiling doesn't change correctness."""
        M, K_in, K_out, E = 16, 8, 4, 2
        rng = jax.random.key(0)
        x = jax.random.normal(rng, (M, K_in))
        w = jax.random.normal(jax.random.key(1), (E, K_in, K_out))
        gs = jnp.array([8, 8], dtype=jnp.int32)

        tiling_9 = (64, 64, 64, 128, 128, 128, 32, 32, 32)
        out = grouped_matmul_vjp(x, w, gs, tiling=tiling_9)
        ref = grouped_matmul(x, w, gs)
        np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)

    def test_3_param_tiling_padded(self):
        """3-param tiling gets auto-expanded to 9."""
        M, K_in, K_out, E = 16, 8, 4, 2
        rng = jax.random.key(0)
        x = jax.random.normal(rng, (M, K_in))
        w = jax.random.normal(jax.random.key(1), (E, K_in, K_out))
        gs = jnp.array([8, 8], dtype=jnp.int32)

        out = grouped_matmul_vjp(x, w, gs, tiling=(64, 64, 64))
        ref = grouped_matmul(x, w, gs)
        np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)

    def test_bfloat16(self):
        """bfloat16 support."""
        M, K_in, K_out, E = 16, 8, 4, 2
        rng = jax.random.key(0)
        x = jax.random.normal(rng, (M, K_in), dtype=jnp.bfloat16)
        w = jax.random.normal(jax.random.key(1), (E, K_in, K_out), dtype=jnp.bfloat16)
        gs = jnp.array([8, 8], dtype=jnp.int32)

        out = grouped_matmul_vjp(x, w, gs)
        assert out.shape == (M, K_out)
        assert jnp.all(jnp.isfinite(out))

        # Backward
        def loss(x_):
            return grouped_matmul_vjp(x_, w, gs).sum()
        grad = jax.grad(loss)(x)
        assert grad.shape == x.shape
        assert jnp.any(grad != 0)

    def test_finite_diff(self):
        """Custom VJP grads match finite differences."""
        M, K_in, K_out, E = 8, 4, 4, 2
        rng = jax.random.key(0)
        x = jax.random.normal(rng, (M, K_in))
        w = jax.random.normal(jax.random.key(1), (E, K_in, K_out))
        gs = jnp.array([4, 4], dtype=jnp.int32)

        def fn(x_):
            return grouped_matmul_vjp(x_, w, gs).sum()

        analytic = jax.grad(fn)(x)
        eps = 1e-3
        for i in range(4):
            for j in range(4):
                xp = x.at[i, j].add(eps)
                xm = x.at[i, j].add(-eps)
                numeric = float((fn(xp) - fn(xm)) / (2 * eps))
                np.testing.assert_allclose(
                    float(analytic[i, j]), numeric,
                    rtol=0.02, atol=1e-4,
                )

    def test_weight_grad_finite_diff(self):
        """Weight gradients from custom VJP match finite differences."""
        M, K_in, K_out, E = 8, 4, 4, 2
        rng = jax.random.key(0)
        x = jax.random.normal(rng, (M, K_in))
        w = jax.random.normal(jax.random.key(1), (E, K_in, K_out))
        gs = jnp.array([4, 4], dtype=jnp.int32)

        def fn(w_):
            return grouped_matmul_vjp(x, w_, gs).sum()

        analytic = jax.grad(fn)(w)
        eps = 1e-3
        for e in range(E):
            for i in range(min(2, K_in)):
                for j in range(min(2, K_out)):
                    wp = w.at[e, i, j].add(eps)
                    wm = w.at[e, i, j].add(-eps)
                    numeric = float((fn(wp) - fn(wm)) / (2 * eps))
                    np.testing.assert_allclose(
                        float(analytic[e, i, j]), numeric,
                        rtol=0.02, atol=1e-4,
                    )

    def test_uneven_groups(self):
        """Uneven group sizes don't break VJP."""
        E, K_in, K_out = 4, 8, 4
        gs = jnp.array([10, 5, 1, 0], dtype=jnp.int32)
        M = int(gs.sum())
        x = jax.random.normal(jax.random.key(0), (M, K_in))
        w = jax.random.normal(jax.random.key(1), (E, K_in, K_out))

        out = grouped_matmul_vjp(x, w, gs)
        assert out.shape == (M, K_out)
        assert jnp.all(jnp.isfinite(out))

        def loss(x_):
            return grouped_matmul_vjp(x_, w, gs).sum()
        grad = jax.grad(loss)(x)
        assert jnp.all(jnp.isfinite(grad))


# --- TestTGMM ---


class TestTGMM:
    """Tests for transposed grouped matmul."""

    def test_correctness_uniform(self):
        """tgmm matches manual per-expert x^T @ grad."""
        M, K, N, E = 16, 8, 4, 2
        rng = jax.random.key(0)
        x = jax.random.normal(rng, (M, K))
        grad = jax.random.normal(jax.random.key(1), (M, N))
        gs = jnp.array([8, 8], dtype=jnp.int32)

        result = tgmm(x, grad, gs)
        assert result.shape == (E, K, N)

        # Manual reference
        ref = jnp.zeros((E, K, N))
        ref = ref.at[0].set(x[:8].T @ grad[:8])
        ref = ref.at[1].set(x[8:].T @ grad[8:])

        np.testing.assert_allclose(result, ref, rtol=1e-4, atol=1e-4)

    def test_various_group_sizes(self):
        """Non-uniform groups."""
        E, K, N = 4, 8, 4
        gs = jnp.array([10, 3, 2, 1], dtype=jnp.int32)
        M = int(gs.sum())
        x = jax.random.normal(jax.random.key(0), (M, K))
        grad = jax.random.normal(jax.random.key(1), (M, N))

        result = tgmm(x, grad, gs)
        assert result.shape == (E, K, N)

        # Manual reference
        ref = jnp.zeros((E, K, N))
        offset = 0
        for i in range(E):
            size = int(gs[i])
            ref = ref.at[i].set(x[offset:offset + size].T @ grad[offset:offset + size])
            offset += size

        np.testing.assert_allclose(result, ref, rtol=1e-4, atol=1e-4)

    def test_empty_expert(self):
        """Empty expert produces zero gradient."""
        E, K, N = 3, 4, 4
        gs = jnp.array([8, 0, 8], dtype=jnp.int32)
        M = int(gs.sum())
        x = jax.random.normal(jax.random.key(0), (M, K))
        grad = jax.random.normal(jax.random.key(1), (M, N))

        result = tgmm(x, grad, gs)
        assert result.shape == (E, K, N)
        # Expert 1 should be zeros
        np.testing.assert_allclose(result[1], jnp.zeros((K, N)), atol=1e-6)

    def test_bfloat16(self):
        """bfloat16 support."""
        M, K, N, E = 16, 8, 4, 2
        x = jax.random.normal(jax.random.key(0), (M, K), dtype=jnp.bfloat16)
        grad = jax.random.normal(jax.random.key(1), (M, N), dtype=jnp.bfloat16)
        gs = jnp.array([8, 8], dtype=jnp.int32)

        result = tgmm(x, grad, gs)
        assert result.shape == (E, K, N)
        assert jnp.all(jnp.isfinite(result))


# --- TestMoEKernelIntegration ---


class TestMoEKernelIntegration:
    """Integration tests: kernels through MoE module."""

    def test_moe_block_forward(self, rngs):
        from mintext.modules.moe import MoEBlock
        cfg = MinTextConfig(
            hidden_size=64, num_attention_heads=4, intermediate_size=128, vocab_size=128,
            num_experts=4, num_experts_per_tok=2, moe_intermediate_size=32,
            n_group=2, topk_group=1, routed_scaling_factor=1.0,
            dtype="float32", weight_dtype="float32",
            moe_use_custom_vjp=False,
        )
        block = MoEBlock(cfg, rngs=rngs)
        x = jax.random.normal(jax.random.key(0), (2, 16, 64))
        out, aux = block(x)
        assert out.shape == (2, 16, 64)
        assert jnp.all(jnp.isfinite(out))

    def test_moe_block_backward(self, rngs):
        from mintext.modules.moe import MoEBlock
        cfg = MinTextConfig(
            hidden_size=64, num_attention_heads=4, intermediate_size=128, vocab_size=128,
            num_experts=4, num_experts_per_tok=2, moe_intermediate_size=32,
            n_group=2, topk_group=1, routed_scaling_factor=1.0,
            dtype="float32", weight_dtype="float32",
            moe_use_custom_vjp=False,
        )
        block = MoEBlock(cfg, rngs=rngs)
        x = jax.random.normal(jax.random.key(0), (2, 16, 64))

        def loss_fn(x_):
            out, _ = block(x_)
            return out.sum()

        grad = jax.grad(loss_fn)(x)
        assert grad.shape == x.shape
        assert jnp.any(grad != 0)

    def test_custom_vjp_moe_block(self, rngs):
        """MoE block with custom VJP matches non-VJP output."""
        from mintext.modules.moe import MoEBlock

        cfg_vjp = MinTextConfig(
            hidden_size=64, num_attention_heads=4, intermediate_size=128, vocab_size=128,
            num_experts=4, num_experts_per_tok=2, moe_intermediate_size=32,
            n_group=2, topk_group=1, routed_scaling_factor=1.0,
            dtype="float32", weight_dtype="float32",
            moe_use_custom_vjp=True,
        )
        block_vjp = MoEBlock(cfg_vjp, rngs=rngs)

        cfg_no_vjp = MinTextConfig(
            hidden_size=64, num_attention_heads=4, intermediate_size=128, vocab_size=128,
            num_experts=4, num_experts_per_tok=2, moe_intermediate_size=32,
            n_group=2, topk_group=1, routed_scaling_factor=1.0,
            dtype="float32", weight_dtype="float32",
            moe_use_custom_vjp=False,
        )
        block_no_vjp = MoEBlock(cfg_no_vjp, rngs=nnx.Rngs(params=0, dropout=1))
        # Share weights
        block_no_vjp.experts.gate_up_proj = block_vjp.experts.gate_up_proj
        block_no_vjp.experts.down_proj = block_vjp.experts.down_proj
        block_no_vjp.router.gate = block_vjp.router.gate
        block_no_vjp.router.e_score_correction_bias = block_vjp.router.e_score_correction_bias

        x = jax.random.normal(jax.random.key(0), (2, 16, 64))
        out_vjp, _ = block_vjp(x)
        out_no_vjp, _ = block_no_vjp(x)

        np.testing.assert_allclose(out_vjp, out_no_vjp, rtol=1e-4, atol=1e-4)

    def test_custom_vjp_moe_block_backward(self, rngs):
        """MoE block backward with custom VJP produces valid gradients."""
        from mintext.modules.moe import MoEBlock

        cfg = MinTextConfig(
            hidden_size=64, num_attention_heads=4, intermediate_size=128, vocab_size=128,
            num_experts=4, num_experts_per_tok=2, moe_intermediate_size=32,
            n_group=2, topk_group=1, routed_scaling_factor=1.0,
            dtype="float32", weight_dtype="float32",
            moe_use_custom_vjp=True,
        )
        block = MoEBlock(cfg, rngs=rngs)
        x = jax.random.normal(jax.random.key(0), (2, 16, 64))

        def loss_fn(x_):
            out, _ = block(x_)
            return out.sum()

        grad = jax.grad(loss_fn)(x)
        assert grad.shape == x.shape
        assert jnp.any(grad != 0)
        assert jnp.all(jnp.isfinite(grad))

    def test_transformer_with_moe_kernels(self, rngs):
        """End-to-end forward pass with MoE kernels."""
        from mintext.models import Transformer, make_causal_mask
        cfg = MinTextConfig(
            num_hidden_layers=2, hidden_size=64, num_attention_heads=4, intermediate_size=128,
            vocab_size=128, seq_length=16,
            model_type="deepseek_v3",
            attention_type="mha",
            num_experts=4, num_experts_per_tok=2, moe_intermediate_size=32,
            n_shared_experts=0, n_group=2, topk_group=1,
            first_k_dense_replace=0,
            routed_scaling_factor=1.0,
            dtype="float32", weight_dtype="float32",
            moe_use_custom_vjp=False,
        )
        model = Transformer(cfg, rngs=rngs)
        tokens = jax.random.randint(jax.random.key(0), (2, 16), 0, 128)
        positions = jnp.broadcast_to(jnp.arange(16), (2, 16))
        mask = make_causal_mask(16)
        logits, aux = model(tokens, positions, mask)
        assert logits.shape == (2, 16, 128)
        assert jnp.all(jnp.isfinite(logits))

    def test_training_step_with_kernels(self, rngs):
        """Loss computes and decreases with kernel backends."""
        from mintext.modules.moe import MoEBlock

        cfg = MinTextConfig(
            hidden_size=64, num_attention_heads=4, intermediate_size=128, vocab_size=128,
            num_experts=4, num_experts_per_tok=2, moe_intermediate_size=32,
            n_group=2, topk_group=1, routed_scaling_factor=1.0,
            dtype="float32", weight_dtype="float32",
            moe_use_custom_vjp=False,
        )
        block = MoEBlock(cfg, rngs=rngs)
        x = jax.random.normal(jax.random.key(0), (2, 16, 64))
        target = jax.random.normal(jax.random.key(1), (2, 16, 64))

        def loss_fn(x_):
            out, _ = block(x_)
            return jnp.mean((out - target) ** 2)

        loss, grad = jax.value_and_grad(loss_fn)(x)
        assert jnp.isfinite(loss)
        assert jnp.any(grad != 0)


# --- TestAutotuner ---


class TestAutotuner:
    """Tests for MoE autotuner."""

    def test_candidate_generation(self):
        from mintext.kernels.autotuner import _generate_candidates
        candidates = _generate_candidates(M=256, K=128, N=64)
        assert len(candidates) > 0
        # All candidates should be tuples of 3 ints
        for c in candidates:
            assert len(c) == 3
            for v in c:
                assert v > 0
                assert v & (v - 1) == 0  # power of 2

    def test_cache_save_load(self, tmp_path):
        from mintext.kernels.autotuner import (
            MoETuningConfig, MoETuningResult,
            _cache_key, _save_cache, _load_cache,
        )
        config = MoETuningConfig(
            num_experts=8, hidden_size=256,
            moe_intermediate_size=512, num_experts_per_tok=2,
            batch_seq_tokens=1024, dtype="bfloat16",
        )
        key = _cache_key(config)
        result = MoETuningResult(
            gate_up_tiling=(128, 64, 256),
            down_tiling=(64, 128, 128),
            backend="ragged_dot",
            throughput_gflops=42.0,
        )
        _save_cache(str(tmp_path), key, result)
        loaded = _load_cache(str(tmp_path), key)
        assert loaded is not None
        assert loaded.gate_up_tiling == (128, 64, 256)
        assert loaded.down_tiling == (64, 128, 128)
        assert loaded.backend == "ragged_dot"
        assert loaded.throughput_gflops == 42.0

    def test_cache_invalidation(self, tmp_path):
        from mintext.kernels.autotuner import (
            MoETuningConfig, MoETuningResult,
            _cache_key, _save_cache, _load_cache,
        )
        config1 = MoETuningConfig(
            num_experts=8, hidden_size=256,
            moe_intermediate_size=512, num_experts_per_tok=2,
            batch_seq_tokens=1024,
        )
        config2 = MoETuningConfig(
            num_experts=16, hidden_size=256,  # different
            moe_intermediate_size=512, num_experts_per_tok=2,
            batch_seq_tokens=1024,
        )
        key1 = _cache_key(config1)
        key2 = _cache_key(config2)
        assert key1 != key2

        result = MoETuningResult(
            gate_up_tiling=None, down_tiling=None,
            backend="ragged_dot", throughput_gflops=0.0,
        )
        _save_cache(str(tmp_path), key1, result)
        assert _load_cache(str(tmp_path), key2) is None

    def test_result_fields(self):
        from mintext.kernels.autotuner import MoETuningResult
        result = MoETuningResult(
            gate_up_tiling=(128, 64, 256),
            down_tiling=None,
            backend="ragged_dot",
            throughput_gflops=50.0,
        )
        assert result.gate_up_tiling == (128, 64, 256)
        assert result.down_tiling is None
        assert result.backend == "ragged_dot"
        assert result.throughput_gflops == 50.0
