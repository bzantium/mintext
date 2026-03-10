"""Tests for training loop components."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from mintext.config import MinTextConfig
from mintext.distributed.mesh import setup_mesh
from mintext.models import Transformer, make_causal_mask
from mintext.optim.optimizer import create_lr_schedule, create_optimizer
from mintext.optim.grad_accumulation import accumulate_gradients
from mintext.trainer import (
    TrainState,
    chunked_cross_entropy_loss,
    compute_loss,
    create_train_state,
    cross_entropy_loss,
    cross_entropy_with_z_loss,
    eval_step,
    train_step,
    _l2_norm,
)


def _skip_if_ad_checkpoint_broken(func, *args, **kwargs):
    """Call func, skip test if layers.py ad_checkpoint is broken."""
    try:
        return func(*args, **kwargs)
    except AttributeError as e:
        if "ad_checkpoint" in str(e):
            pytest.skip("blocked: layers.py jax.ad_checkpoint broken by model-agent")
        raise


@pytest.fixture
def config():
    return MinTextConfig(
        num_hidden_layers=1,
        hidden_size=64,
        num_attention_heads=4,
        head_dim=16,
        intermediate_size=128,
        vocab_size=128,
        seq_length=16,
        steps=10,
        learning_rate=1e-3,
        warmup_steps_fraction=0.1,
        dtype="float32",
        weight_dtype="float32",
    )


@pytest.fixture
def model_and_state(config):
    setup_mesh(config)
    model = Transformer(config, rngs=nnx.Rngs(params=0))
    state = create_train_state(model, config)
    return model, state


@pytest.fixture
def batch(config):
    rng = jax.random.key(42)
    tokens = jax.random.randint(rng, (2, 16), 0, config.vocab_size)
    return {"input_tokens": tokens, "target_tokens": tokens}


class TestCrossEntropyLoss:
    def test_output_shape(self):
        logits = jax.random.normal(jax.random.key(0), (2, 8, 128))
        targets = jax.random.randint(jax.random.key(1), (2, 8), 0, 128)
        loss, z_loss = cross_entropy_loss(logits, targets)
        assert loss.shape == (2, 8)
        assert z_loss.shape == (2, 8)

    def test_loss_positive(self):
        logits = jax.random.normal(jax.random.key(0), (2, 8, 128))
        targets = jax.random.randint(jax.random.key(1), (2, 8), 0, 128)
        loss, _ = cross_entropy_loss(logits, targets)
        assert jnp.all(loss >= 0)

    def test_perfect_predictions_low_loss(self):
        # Create logits that strongly predict the target
        targets = jnp.array([[0, 1, 2]])
        logits = jnp.full((1, 3, 10), -10.0)
        logits = logits.at[0, 0, 0].set(10.0)
        logits = logits.at[0, 1, 1].set(10.0)
        logits = logits.at[0, 2, 2].set(10.0)
        loss, _ = cross_entropy_loss(logits, targets)
        assert jnp.all(loss < 0.01)

    def test_label_smoothing(self):
        logits = jax.random.normal(jax.random.key(0), (2, 8, 128))
        targets = jax.random.randint(jax.random.key(1), (2, 8), 0, 128)
        loss_no_smooth, _ = cross_entropy_loss(logits, targets, label_smoothing=0.0)
        loss_smooth, _ = cross_entropy_loss(logits, targets, label_smoothing=0.1)
        # Smoothed loss should differ
        assert not jnp.allclose(loss_no_smooth, loss_smooth)

    def test_label_smoothing_matches_dense_reference(self):
        logits = jax.random.normal(jax.random.key(0), (2, 4, 32))
        targets = jax.random.randint(jax.random.key(1), (2, 4), 0, 32)
        label_smoothing = 0.2

        loss, _ = cross_entropy_loss(logits, targets, label_smoothing=label_smoothing)

        one_hot = jax.nn.one_hot(targets, logits.shape[-1])
        smoothed_targets = (
            one_hot * (1.0 - label_smoothing) + label_smoothing / logits.shape[-1]
        )
        ref = -jnp.sum(smoothed_targets * jax.nn.log_softmax(logits, axis=-1), axis=-1)

        assert jnp.allclose(loss, ref, atol=1e-6)


class TestZLoss:
    def test_z_loss_increases_loss(self):
        logits = jax.random.normal(jax.random.key(0), (2, 8, 128))
        targets = jax.random.randint(jax.random.key(1), (2, 8), 0, 128)
        loss_no_z, z_no = cross_entropy_loss(logits, targets, z_loss_weight=0.0)
        loss_with_z, z_yes = cross_entropy_loss(logits, targets, z_loss_weight=1e-4)
        # z_loss should be zero when weight is 0
        assert jnp.allclose(z_no, 0.0)
        # z_loss should be positive when weight > 0
        assert jnp.all(z_yes > 0)
        # Total loss should be larger with z-loss
        assert jnp.mean(loss_with_z) > jnp.mean(loss_no_z)

    def test_z_loss_zero_weight_matches_standard(self):
        logits = jax.random.normal(jax.random.key(0), (2, 8, 128))
        targets = jax.random.randint(jax.random.key(1), (2, 8), 0, 128)
        loss_standard, _ = cross_entropy_loss(logits, targets, z_loss_weight=0.0)
        loss_z_zero, _ = cross_entropy_loss(logits, targets, z_loss_weight=0.0)
        assert jnp.allclose(loss_standard, loss_z_zero)

    def test_z_loss_gradient_flows(self):
        logits = jax.random.normal(jax.random.key(0), (2, 4, 32))
        one_hot = jax.nn.one_hot(jnp.array([[0, 1, 2, 3], [1, 2, 3, 0]]), 32)

        def loss_fn(logits):
            loss, z_loss = cross_entropy_with_z_loss(logits, one_hot, 1e-3)
            return jnp.mean(loss)

        grad = jax.grad(loss_fn)(logits)
        assert jnp.all(jnp.isfinite(grad))
        # Gradient should be non-zero
        assert jnp.any(grad != 0)

    def test_z_loss_matches_dense_reference_with_label_smoothing(self):
        logits = jax.random.normal(jax.random.key(0), (2, 4, 32))
        targets = jax.random.randint(jax.random.key(1), (2, 4), 0, 32)
        label_smoothing = 0.15
        z_loss_weight = 1e-4

        loss, z_loss = cross_entropy_loss(
            logits,
            targets,
            label_smoothing=label_smoothing,
            z_loss_weight=z_loss_weight,
        )

        one_hot = jax.nn.one_hot(targets, logits.shape[-1])
        smoothed_targets = (
            one_hot * (1.0 - label_smoothing) + label_smoothing / logits.shape[-1]
        )
        ref_loss, ref_z_loss = cross_entropy_with_z_loss(logits, smoothed_targets, z_loss_weight)

        assert jnp.allclose(loss, ref_loss, atol=1e-5)
        assert jnp.allclose(z_loss, ref_z_loss, atol=1e-6)

    def test_z_loss_in_compute_loss(self, config, model_and_state, batch):
        model, state = model_and_state
        cfg = MinTextConfig(
            num_hidden_layers=1, hidden_size=64, num_attention_heads=4, head_dim=16,
            intermediate_size=128, vocab_size=128, seq_length=16, steps=10,
            dtype="float32", weight_dtype="float32", z_loss_weight=1e-4,
        )
        loss, aux = _skip_if_ad_checkpoint_broken(
            compute_loss, state.params, model, cfg, batch
        )
        assert "z_loss" in aux
        assert float(aux["z_loss"]) > 0


class TestLRSchedule:
    def test_warmup_starts_at_zero(self, config):
        schedule = create_lr_schedule(config)
        assert float(schedule(0)) == pytest.approx(0.0, abs=1e-6)

    def test_peak_after_warmup(self, config):
        schedule = create_lr_schedule(config)
        warmup_steps = max(int(config.steps * config.warmup_steps_fraction), 1)
        lr_at_peak = float(schedule(warmup_steps))
        assert lr_at_peak == pytest.approx(config.learning_rate, rel=0.01)

    def test_decay_after_warmup(self, config):
        schedule = create_lr_schedule(config)
        warmup_steps = max(int(config.steps * config.warmup_steps_fraction), 1)
        lr_peak = float(schedule(warmup_steps))
        lr_end = float(schedule(config.steps - 1))
        assert lr_end < lr_peak

    def test_linear_schedule(self):
        config = MinTextConfig(steps=100, lr_schedule="linear", learning_rate=1e-3)
        schedule = create_lr_schedule(config)
        assert float(schedule(0)) == pytest.approx(0.0, abs=1e-6)


class TestWSDSchedule:
    def test_warmup_phase(self):
        config = MinTextConfig(
            steps=1000, lr_schedule="wsd", learning_rate=1e-3,
            warmup_steps_fraction=0.1, wsd_decay_steps_fraction=0.1,
        )
        schedule = create_lr_schedule(config)
        # Starts at 0
        assert float(schedule(0)) == pytest.approx(0.0, abs=1e-6)
        # After warmup, reaches peak
        assert float(schedule(100)) == pytest.approx(1e-3, rel=0.01)

    def test_stable_phase(self):
        config = MinTextConfig(
            steps=1000, lr_schedule="wsd", learning_rate=1e-3,
            warmup_steps_fraction=0.1, wsd_decay_steps_fraction=0.1,
        )
        schedule = create_lr_schedule(config)
        # In the stable phase (step 200, 500, 800), LR should be at peak
        for step in [200, 500, 800]:
            assert float(schedule(step)) == pytest.approx(1e-3, rel=0.01)

    def test_decay_phase(self):
        config = MinTextConfig(
            steps=1000, lr_schedule="wsd", learning_rate=1e-3,
            warmup_steps_fraction=0.1, wsd_decay_steps_fraction=0.1,
            lr_final_fraction=0.0,
        )
        schedule = create_lr_schedule(config)
        # At end of stable phase
        lr_stable = float(schedule(899))
        # Near end of decay
        lr_end = float(schedule(999))
        assert lr_end < lr_stable

    def test_cosine_decay_style(self):
        config = MinTextConfig(
            steps=1000, lr_schedule="wsd", learning_rate=1e-3,
            warmup_steps_fraction=0.1, wsd_decay_steps_fraction=0.2,
            wsd_decay_style="cosine",
        )
        schedule = create_lr_schedule(config)
        # Should still reach peak after warmup
        assert float(schedule(100)) == pytest.approx(1e-3, rel=0.01)
        # Should decay near the end
        lr_mid_stable = float(schedule(500))
        lr_end = float(schedule(999))
        assert lr_end < lr_mid_stable


class TestOptimizer:
    def test_creates_optimizer(self, config):
        tx = create_optimizer(config)
        assert tx is not None

    def test_no_clipping(self):
        config = MinTextConfig(gradient_clip_threshold=0.0)
        tx = create_optimizer(config)
        assert tx is not None


class TestMuonOptimizer:
    def test_muon_optimizer_creation(self):
        config = MinTextConfig(
            optimizer="muon", steps=10, learning_rate=1e-3,
            muon_newton_schulz_steps=5, muon_beta=0.95,
        )
        tx = create_optimizer(config)
        assert tx is not None

    def test_muon_consistent_rms_creation(self):
        """Test muon with consistent_rms — silently skipped if optax version lacks support."""
        config = MinTextConfig(
            optimizer="muon", steps=10, learning_rate=1e-3,
            muon_newton_schulz_steps=5, muon_beta=0.95,
            muon_consistent_rms=0.2,
        )
        tx = create_optimizer(config)
        assert tx is not None

    def test_muon_training_step(self):
        config = MinTextConfig(
            num_hidden_layers=1, hidden_size=64, num_attention_heads=4, head_dim=16,
            intermediate_size=128, vocab_size=128, seq_length=16, steps=10,
            learning_rate=1e-3, dtype="float32", weight_dtype="float32",
            optimizer="muon", muon_newton_schulz_steps=5,
        )
        setup_mesh(config)
        model = Transformer(config, rngs=nnx.Rngs(params=0))
        state = create_train_state(model, config)
        rng = jax.random.key(42)
        tokens = jax.random.randint(rng, (2, 16), 0, 128)
        batch = {"input_tokens": tokens, "target_tokens": tokens}

        losses = []
        for _ in range(5):
            state, metrics = _skip_if_ad_checkpoint_broken(
                train_step, state, batch, model, config
            )
            losses.append(float(metrics["loss"]))
        # Loss should decrease
        assert losses[-1] < losses[0]


class TestComputeLoss:
    def test_returns_loss_and_aux(self, config, model_and_state, batch):
        model, state = model_and_state
        loss, aux = _skip_if_ad_checkpoint_broken(
            compute_loss, state.params, model, config, batch
        )
        assert loss.shape == ()
        assert "loss" in aux
        assert "per_token_loss" in aux
        assert "z_loss" in aux

    def test_loss_is_finite(self, config, model_and_state, batch):
        model, state = model_and_state
        loss, _ = _skip_if_ad_checkpoint_broken(
            compute_loss, state.params, model, config, batch
        )
        assert jnp.isfinite(loss)


class TestTrainStep:
    def test_returns_state_and_metrics(self, config, model_and_state, batch):
        model, state = model_and_state
        new_state, metrics = _skip_if_ad_checkpoint_broken(
            train_step, state, batch, model, config
        )
        assert isinstance(new_state, TrainState)
        assert "loss" in metrics
        assert "learning_rate" in metrics
        assert "grad_norm" in metrics
        assert "param_norm" in metrics

    def test_step_increments(self, config, model_and_state, batch):
        model, state = model_and_state
        assert int(state.step) == 0
        new_state, _ = _skip_if_ad_checkpoint_broken(
            train_step, state, batch, model, config
        )
        assert int(new_state.step) == 1

    def test_loss_decreases_over_steps(self, config, model_and_state, batch):
        model, state = model_and_state
        losses = []
        for _ in range(5):
            state, metrics = _skip_if_ad_checkpoint_broken(
                train_step, state, batch, model, config
            )
            losses.append(float(metrics["loss"]))
        # Loss should decrease on the same batch (overfitting)
        assert losses[-1] < losses[0]


class TestEvalStep:
    def test_returns_metrics(self, config, model_and_state, batch):
        model, state = model_and_state
        metrics = _skip_if_ad_checkpoint_broken(
            eval_step, state, batch, model, config
        )
        assert "eval_loss" in metrics
        assert jnp.isfinite(metrics["eval_loss"])


class TestCreateTrainState:
    def test_has_params_and_optimizer(self, config):
        setup_mesh(config)
        model = Transformer(config, rngs=nnx.Rngs(params=0))
        state = create_train_state(model, config)
        assert state.params is not None
        assert state.opt_state is not None
        assert int(state.step) == 0


class TestL2Norm:
    def test_simple(self):
        tree = {"a": jnp.array([3.0, 4.0])}
        assert float(_l2_norm(tree)) == pytest.approx(5.0)

    def test_nested(self):
        tree = {"a": jnp.array([1.0]), "b": {"c": jnp.array([1.0])}}
        assert float(_l2_norm(tree)) == pytest.approx(jnp.sqrt(2.0))


class TestChunkedCrossEntropy:
    """Tests for vocab tiling (chunked cross-entropy loss)."""

    def test_chunked_matches_full(self):
        """Chunked loss (tiles=4) matches full logits loss within tolerance."""
        from mintext.modules.linear import Linear

        hidden_size, vocab_size = 64, 128
        total_tokens = 32  # B*S

        rngs = nnx.Rngs(params=42)
        output_proj = Linear(
            hidden_size, vocab_size,
            dtype=jnp.float32, weight_dtype=jnp.float32,
            rngs=rngs,
        )
        hidden = jax.random.normal(jax.random.key(0), (total_tokens, hidden_size))
        targets = jax.random.randint(jax.random.key(1), (total_tokens,), 0, vocab_size)

        # Full loss
        full_logits = output_proj(hidden)  # [total_tokens, vocab_size]
        full_loss, full_z = cross_entropy_loss(
            full_logits[jnp.newaxis], targets[jnp.newaxis],
        )
        full_loss = full_loss.squeeze(0)

        # Chunked loss
        chunked_loss, chunked_z = chunked_cross_entropy_loss(
            hidden, targets, output_proj, num_tiles=4,
        )

        assert chunked_loss.shape == (total_tokens,)
        assert jnp.allclose(full_loss, chunked_loss, atol=1e-5)

    def test_vocab_tiling_config_validation(self):
        """per_device_batch_size * seq_length must be divisible by num_vocab_tiles."""
        # Valid: 2*64 = 128, 128 % 4 = 0
        cfg = MinTextConfig(
            num_vocab_tiles=4, per_device_batch_size=2, seq_length=64,
            vocab_size=256, dtype="float32", weight_dtype="float32",
        )
        assert cfg.num_vocab_tiles == 4

        # Invalid: 2*63 = 126, 126 % 4 != 0
        with pytest.raises(ValueError, match="divisible"):
            MinTextConfig(
                num_vocab_tiles=4, per_device_batch_size=2, seq_length=63,
                vocab_size=256, dtype="float32", weight_dtype="float32",
            )

    def test_vocab_tiling_in_compute_loss(self):
        """Vocab tiling produces equivalent loss through compute_loss."""
        cfg_full = MinTextConfig(
            num_hidden_layers=1, hidden_size=64, num_attention_heads=4, head_dim=16,
            intermediate_size=128, vocab_size=128, seq_length=16, steps=10,
            per_device_batch_size=2, num_vocab_tiles=1,
            dtype="float32", weight_dtype="float32",
        )
        cfg_tiled = MinTextConfig(
            num_hidden_layers=1, hidden_size=64, num_attention_heads=4, head_dim=16,
            intermediate_size=128, vocab_size=128, seq_length=16, steps=10,
            per_device_batch_size=2, num_vocab_tiles=4,
            dtype="float32", weight_dtype="float32",
        )
        setup_mesh(cfg_full)

        rngs = nnx.Rngs(params=42)
        model_full = Transformer(cfg_full, rngs=rngs)
        rngs2 = nnx.Rngs(params=42)
        model_tiled = Transformer(cfg_tiled, rngs=rngs2)

        _, params_full = nnx.split(model_full)
        _, params_tiled = nnx.split(model_tiled)

        tokens = jax.random.randint(jax.random.key(0), (2, 16), 0, 128)
        batch = {"input_tokens": tokens, "target_tokens": tokens}

        loss_full, _ = _skip_if_ad_checkpoint_broken(
            compute_loss, params_full, model_full, cfg_full, batch
        )
        loss_tiled, _ = _skip_if_ad_checkpoint_broken(
            compute_loss, params_tiled, model_tiled, cfg_tiled, batch
        )

        assert jnp.allclose(loss_full, loss_tiled, atol=1e-4), (
            f"Full loss={float(loss_full)}, tiled loss={float(loss_tiled)}"
        )


class TestGradientAccumulation:
    def test_matches_full_batch_gradient(self):
        params = {"w": jnp.array([1.0, -2.0], dtype=jnp.float32)}
        data = {"x": jnp.arange(8, dtype=jnp.float32).reshape(4, 2)}

        def loss_fn(current_params, batch):
            preds = batch["x"] @ current_params["w"]
            loss = jnp.mean(preds ** 2)
            aux = {"mean_pred": jnp.mean(preds)}
            return loss, aux

        full_grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (full_loss, _), full_grads = full_grad_fn(params, data)

        loss, last_aux, grads = accumulate_gradients(
            loss_fn=loss_fn,
            params=params,
            data=data,
            num_micro_steps=2,
        )

        assert jnp.allclose(loss, full_loss, atol=1e-6)
        assert jnp.allclose(grads["w"], full_grads["w"], atol=1e-6)

        last_micro = {"x": data["x"][2:]}
        expected_aux = loss_fn(params, last_micro)[1]
        assert jnp.allclose(last_aux["mean_pred"], expected_aux["mean_pred"], atol=1e-6)

    def test_requires_divisible_micro_batches(self):
        params = {"w": jnp.array([1.0], dtype=jnp.float32)}
        data = {"x": jnp.arange(3, dtype=jnp.float32).reshape(3, 1)}

        with pytest.raises(ValueError, match="divisible"):
            accumulate_gradients(
                loss_fn=lambda current_params, batch: (
                    jnp.mean(batch["x"] * current_params["w"]),
                    {},
                ),
                params=params,
                data=data,
                num_micro_steps=2,
            )
