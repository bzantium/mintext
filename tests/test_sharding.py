"""Tests for distributed infrastructure: mesh, sharding, partition, scan."""

from unittest import mock

import numpy as np
import jax
import jax.numpy as jnp
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from flax import nnx

from mintext.config import MinTextConfig
from mintext.distributed.mesh import (
    _resolve_auto_parallelism,
    _detect_num_slices,
    create_mesh,
    initialize_distributed,
    setup_mesh,
)
from mintext.distributed.sharding import (
    DEFAULT_LOGICAL_AXIS_RULES,
    add_data_axis_to_sharding,
    create_named_sharding,
    get_input_data_sharding,
    get_logical_axis_rules,
    get_model_param_shardings,
    logical_to_pspec,
)
from mintext.distributed.partition import create_sharded_model
from mintext.models import Transformer, _all_layers_scannable, make_causal_mask


class TestResolveAutoParallelism:
    def test_all_explicit(self):
        assert _resolve_auto_parallelism(1, 1, 1, 1) == (1, 1, 1)

    def test_auto_fsdp(self):
        assert _resolve_auto_parallelism(1, -1, 1, 8) == (1, 8, 1)

    def test_auto_data(self):
        assert _resolve_auto_parallelism(-1, 2, 2, 8) == (2, 2, 2)

    def test_auto_tensor(self):
        assert _resolve_auto_parallelism(2, 2, -1, 8) == (2, 2, 2)

    def test_product_mismatch_raises(self):
        with pytest.raises(ValueError, match="product"):
            _resolve_auto_parallelism(2, 2, 2, 7)

    def test_multiple_auto_raises(self):
        with pytest.raises(ValueError, match="At most one"):
            _resolve_auto_parallelism(-1, -1, 1, 4)

    def test_auto_not_divisible_raises(self):
        with pytest.raises(ValueError, match="not divisible"):
            _resolve_auto_parallelism(1, -1, 3, 8)


class TestCreateMesh:
    def test_single_cpu_device(self):
        config = MinTextConfig(
            ici_data_parallelism=1,
            ici_fsdp_parallelism=1,
            ici_tensor_parallelism=1,
        )
        devices = jax.devices()[:1]
        mesh = create_mesh(config, devices)
        assert mesh.shape == {"data": 1, "fsdp": 1, "tensor": 1}
        assert mesh.axis_names == ("data", "fsdp", "tensor")

    def test_auto_fsdp_single_device(self):
        config = MinTextConfig(
            ici_data_parallelism=1,
            ici_fsdp_parallelism=-1,
            ici_tensor_parallelism=1,
        )
        devices = jax.devices()[:1]
        mesh = create_mesh(config, devices)
        assert mesh.shape["fsdp"] == len(devices)


class TestSetupMesh:
    def test_sets_global_mesh(self):
        config = MinTextConfig(
            ici_data_parallelism=1,
            ici_fsdp_parallelism=-1,
            ici_tensor_parallelism=1,
        )
        mesh = setup_mesh(config)
        abstract_mesh = jax.sharding.get_abstract_mesh()
        assert not abstract_mesh.empty
        assert "data" in abstract_mesh.axis_names


class TestLogicalAxisRules:
    def test_default_rules(self):
        rules = get_logical_axis_rules()
        assert ("batch", "data") in rules
        assert ("embed", "fsdp") in rules
        assert ("heads", "tensor") in rules
        assert ("norm", None) in rules

    def test_rules_from_config(self):
        rules = get_logical_axis_rules(MinTextConfig())
        assert len(rules) > 0
        # Should contain all essential mappings
        rule_names = {r[0] for r in rules}
        assert "batch" in rule_names
        assert "embed" in rule_names


class TestLogicalToPspec:
    @pytest.fixture
    def mesh(self):
        devices = np.array(jax.devices()[:1])
        return Mesh(devices.reshape(1, 1, 1), ("data", "fsdp", "tensor"))

    def test_embed_mlp(self, mesh):
        pspec = logical_to_pspec(("embed", "mlp"), DEFAULT_LOGICAL_AXIS_RULES, mesh)
        # On 1-device mesh, all axes are size 1, so everything becomes None
        assert pspec == P(None, None)

    def test_none_axes(self, mesh):
        pspec = logical_to_pspec((None, None), DEFAULT_LOGICAL_AXIS_RULES, mesh)
        assert pspec == P(None, None)

    def test_norm_axis(self, mesh):
        pspec = logical_to_pspec(("norm",), DEFAULT_LOGICAL_AXIS_RULES, mesh)
        assert pspec == P(None)


class TestCreateNamedSharding:
    @pytest.fixture
    def mesh(self):
        devices = np.array(jax.devices()[:1])
        return Mesh(devices.reshape(1, 1, 1), ("data", "fsdp", "tensor"))

    def test_returns_named_sharding(self, mesh):
        sharding = create_named_sharding(mesh, ("batch", "length"))
        assert isinstance(sharding, NamedSharding)


class TestGetInputDataSharding:
    def test_data_sharding(self):
        config = MinTextConfig()
        devices = np.array(jax.devices()[:1])
        mesh = Mesh(devices.reshape(1, 1, 1), ("data", "fsdp", "tensor"))
        sharding = get_input_data_sharding(config, mesh)
        assert isinstance(sharding, NamedSharding)


class TestCreateShardedModel:
    def test_creates_model(self):
        config = MinTextConfig(
            num_hidden_layers=1,
            hidden_size=64,
            num_attention_heads=4,
            head_dim=16,
            intermediate_size=128,
            vocab_size=128,
            dtype="float32",
            weight_dtype="float32",
        )
        # setup_mesh sets global mesh + rules
        mesh = setup_mesh(config)
        model = create_sharded_model(config, mesh)
        assert isinstance(model, Transformer)

        # Verify forward pass works
        tokens = jnp.zeros((2, 8), dtype=jnp.int32)
        positions = jnp.broadcast_to(jnp.arange(8), (2, 8))
        from mintext.models import make_causal_mask

        mask = make_causal_mask(8)
        logits, _ = model(tokens, positions, mask)
        assert logits.shape == (2, 8, 128)


class TestGetModelParamShardings:
    def test_extracts_shardings(self):
        config = MinTextConfig(
            num_hidden_layers=1,
            hidden_size=64,
            num_attention_heads=4,
            head_dim=16,
            intermediate_size=128,
            vocab_size=128,
            dtype="float32",
            weight_dtype="float32",
        )
        mesh = setup_mesh(config)
        model = create_sharded_model(config, mesh)
        shardings = get_model_param_shardings(model, mesh)
        # Should be a pytree of NamedSharding
        leaves = jax.tree.leaves(shardings, is_leaf=lambda x: isinstance(x, NamedSharding))
        assert len(leaves) > 0
        assert all(isinstance(s, NamedSharding) for s in leaves)


class TestInitializeDistributed:
    def test_skips_single_node(self):
        """With no env vars and non-TPU platform, distributed init is skipped."""
        cpu_device = mock.MagicMock()
        cpu_device.platform = "cpu"
        with mock.patch.dict("os.environ", {"JAX_COORDINATOR_IP": ""}, clear=False):
            with mock.patch("jax.distributed.is_initialized", return_value=False):
                with mock.patch("jax.devices", return_value=[cpu_device]):
                    initialize_distributed(timeout=5)
                    # Should not raise

    def test_skips_if_already_initialized(self):
        """If JAX distributed is already initialized, skip."""
        with mock.patch("jax.distributed.is_initialized", return_value=True):
            # Should return immediately without error
            initialize_distributed(timeout=5)

    def test_gpu_env_vars_calls_initialize(self):
        """With GPU env vars set, should call jax.distributed.initialize."""
        env_vars = {
            "JAX_COORDINATOR_IP": "10.0.0.1",
            "JAX_COORDINATOR_PORT": "12345",
            "NNODES": "2",
            "NODE_RANK": "0",
        }
        with mock.patch.dict("os.environ", env_vars, clear=False):
            with mock.patch("jax.distributed.is_initialized", return_value=False):
                with mock.patch("jax.distributed.initialize") as mock_init:
                    initialize_distributed(timeout=60)
                    mock_init.assert_called_once_with(
                        coordinator_address="10.0.0.1:12345",
                        num_processes=2,
                        process_id=0,
                        local_device_ids=None,
                        initialization_timeout=60,
                    )

    def test_gpu_env_vars_with_cuda_visible(self):
        """CUDA_VISIBLE_DEVICES should be parsed into local_device_ids."""
        env_vars = {
            "JAX_COORDINATOR_IP": "10.0.0.1",
            "NNODES": "2",
            "NODE_RANK": "1",
            "CUDA_VISIBLE_DEVICES": "0,1,2,3",
        }
        with mock.patch.dict("os.environ", env_vars, clear=False):
            with mock.patch("jax.distributed.is_initialized", return_value=False):
                with mock.patch("jax.distributed.initialize") as mock_init:
                    initialize_distributed(timeout=30)
                    mock_init.assert_called_once()
                    call_kwargs = mock_init.call_args[1]
                    assert call_kwargs["local_device_ids"] == [0, 1, 2, 3]
                    assert call_kwargs["process_id"] == 1


class TestDetectNumSlices:
    def test_fallback_no_slice_index(self):
        """When devices don't have slice_index, returns 1."""
        # Use real CPU devices which lack slice_index
        devices = jax.devices()[:1]
        assert _detect_num_slices(devices) == 1

    def test_single_slice_attr(self):
        """Devices with slice_index=0 returns 1."""
        mock_dev = mock.MagicMock()
        mock_dev.slice_index = 0
        assert _detect_num_slices([mock_dev]) == 1

    def test_multi_slice(self):
        """Devices with different slice_index values."""
        devs = []
        for si in [0, 0, 0, 0, 1, 1, 1, 1]:
            d = mock.MagicMock()
            d.slice_index = si
            devs.append(d)
        assert _detect_num_slices(devs) == 2


class TestDCNMesh:
    def test_single_slice_same_as_before(self):
        """DCN with 1 slice produces same mesh as non-DCN."""
        config = MinTextConfig(
            ici_data_parallelism=1,
            ici_fsdp_parallelism=1,
            ici_tensor_parallelism=1,
            dcn_data_parallelism=1,
            dcn_fsdp_parallelism=1,
            dcn_tensor_parallelism=1,
            num_slices=1,
        )
        devices = jax.devices()[:1]
        mesh = create_mesh(config, devices)
        assert mesh.shape == {"data": 1, "fsdp": 1, "tensor": 1}
        assert mesh.axis_names == ("data", "fsdp", "tensor")


class TestZeRO1Sharding:
    """Tests for ZeRO-1 optimizer sharding utility."""

    @pytest.fixture
    def mesh(self):
        devices = np.array(jax.devices()[:1])
        return Mesh(devices.reshape(1, 1, 1), ("data", "fsdp", "tensor"))

    def test_add_data_axis_to_unsharded_dim(self, mesh):
        """Adds 'data' to the first None dimension."""
        sharding = NamedSharding(mesh, P(None, "fsdp"))
        result = add_data_axis_to_sharding(sharding)
        assert result.spec == P("data", "fsdp")

    def test_add_data_axis_skips_sharded(self, mesh):
        """Skips already-sharded dims, adds to first None."""
        sharding = NamedSharding(mesh, P("fsdp", None))
        result = add_data_axis_to_sharding(sharding)
        assert result.spec == P("fsdp", "data")

    def test_add_data_axis_compound_sharding(self, mesh):
        """When all dims are sharded, compound-shards the first."""
        sharding = NamedSharding(mesh, P("fsdp", "tensor"))
        result = add_data_axis_to_sharding(sharding)
        assert result.spec == P(("data", "fsdp"), "tensor")

    def test_add_data_axis_scalar(self, mesh):
        """Works with fully replicated (no-dim) partitions."""
        sharding = NamedSharding(mesh, P(None,))
        result = add_data_axis_to_sharding(sharding)
        assert result.spec == P("data",)


class TestCanScanLayers:
    def test_homogeneous_layers(self):
        """All full_attention layers should be scannable."""
        config = MinTextConfig(
            num_hidden_layers=4,
            scan_layers=True,
            layer_types=["full_attention"] * 4,
        )
        assert _all_layers_scannable(config) is True

    def test_scan_disabled(self):
        """scan_layers=False disables scan."""
        config = MinTextConfig(
            num_hidden_layers=4,
            scan_layers=False,
            layer_types=["full_attention"] * 4,
        )
        assert _all_layers_scannable(config) is False

    def test_single_layer(self):
        """Single layer is not worth scanning."""
        config = MinTextConfig(num_hidden_layers=1, scan_layers=True)
        assert _all_layers_scannable(config) is False

    def test_heterogeneous_layers(self):
        """Mixed layer types cannot be scanned."""
        config = MinTextConfig(
            num_hidden_layers=4,
            scan_layers=True,
            layer_types=["full_attention", "full_attention", "linear_attention", "full_attention"],
        )
        assert _all_layers_scannable(config) is False

    def test_moe_not_scannable(self):
        """MoE layers cannot be scanned (aux output varies)."""
        config = MinTextConfig(
            num_hidden_layers=4,
            scan_layers=True,
            num_experts=8,
            first_k_dense_replace=1,
        )
        assert _all_layers_scannable(config) is False

    def test_default_llama_scannable(self):
        """Default Llama config (all full_attention, no MoE) is scannable."""
        config = MinTextConfig(num_hidden_layers=4, scan_layers=True, model_type="llama3")
        assert _all_layers_scannable(config) is True


class TestScanForward:
    @pytest.fixture
    def tiny_config(self):
        return dict(
            num_hidden_layers=3,
            hidden_size=64,
            num_attention_heads=4,
            head_dim=16,
            intermediate_size=128,
            vocab_size=128,
            seq_length=16,
            dtype="float32",
            weight_dtype="float32",
        )

    def test_scan_forward_shape(self, tiny_config):
        """Scan forward produces correct output shape."""
        config = MinTextConfig(scan_layers=True, **tiny_config)
        mesh = setup_mesh(config)
        model = Transformer(config, rngs=nnx.Rngs(0))

        tokens = jnp.zeros((2, 8), dtype=jnp.int32)
        positions = jnp.broadcast_to(jnp.arange(8), (2, 8))
        mask = make_causal_mask(8)

        logits, all_aux = model(tokens, positions, mask)
        assert logits.shape == (2, 8, 128)
        assert len(all_aux) == 3
        assert all(a is None for a in all_aux)

    def test_scan_forward_not_nan(self, tiny_config):
        """Scan forward output is finite."""
        config = MinTextConfig(scan_layers=True, **tiny_config)
        setup_mesh(config)
        model = Transformer(config, rngs=nnx.Rngs(42))

        tokens = jax.random.randint(jax.random.key(0), (2, 8), 0, 128)
        positions = jnp.broadcast_to(jnp.arange(8), (2, 8))
        mask = make_causal_mask(8)

        logits, _ = model(tokens, positions, mask)
        assert jnp.all(jnp.isfinite(logits))

    def test_scan_grad_works(self, tiny_config):
        """Gradients can be computed through scanned layers."""
        config = MinTextConfig(scan_layers=True, **tiny_config)
        setup_mesh(config)
        model = Transformer(config, rngs=nnx.Rngs(0))

        tokens = jnp.ones((2, 8), dtype=jnp.int32)
        positions = jnp.broadcast_to(jnp.arange(8), (2, 8))
        mask = make_causal_mask(8)

        def loss_fn(model):
            logits, _ = model(tokens, positions, mask)
            return jnp.mean(logits)

        grad_fn = nnx.grad(loss_fn)
        grads = grad_fn(model)
        # Should produce non-zero gradients
        grad_leaves = jax.tree.leaves(grads)
        assert len(grad_leaves) > 0

    def test_loop_forward_shape(self, tiny_config):
        """Loop (non-scan) forward still works correctly."""
        config = MinTextConfig(scan_layers=False, **tiny_config)
        setup_mesh(config)
        model = Transformer(config, rngs=nnx.Rngs(0))

        tokens = jnp.zeros((2, 8), dtype=jnp.int32)
        positions = jnp.broadcast_to(jnp.arange(8), (2, 8))
        mask = make_causal_mask(8)

        logits, all_aux = model(tokens, positions, mask)
        assert logits.shape == (2, 8, 128)
        assert len(all_aux) == 3
