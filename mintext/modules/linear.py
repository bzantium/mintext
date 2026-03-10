"""Dense layers and MLP."""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from flax import nnx
from jax.ad_checkpoint import checkpoint_name

from mintext.config import MinTextConfig


# --- Initializers ---


def variance_scaling_init(scale: float = 1.0):
    """Variance-scaling initializer for multi-axis dense kernels."""

    def init(key, shape, dtype, in_axis, out_axis):
        # Compute fan_in/fan_out for multi-axis shapes
        total = 1
        for i, s in enumerate(shape):
            total *= s
        fan_in = total // shape[out_axis] if isinstance(out_axis, int) else total // (shape[out_axis[0]] * shape[out_axis[1]] if len(out_axis) == 2 else shape[out_axis[0]])
        std = scale / jnp.sqrt(fan_in)
        return jax.random.truncated_normal(key, -2.0, 2.0, shape, dtype) * std

    return init


def _default_kernel_init():
    return variance_scaling_init(1.0)


def _fp8_matmul(x: jax.Array, kernel: jax.Array, dtype: jnp.dtype) -> jax.Array:
    """FP8 matmul with dynamic scaling. Falls back to standard matmul if FP8 not available."""
    try:
        fp8_e4m3 = jnp.float8_e4m3fn
        fp8_e5m2 = jnp.float8_e5m2
    except AttributeError:
        return jnp.dot(jnp.asarray(x, dtype), jnp.asarray(kernel, dtype))

    x_f32 = jnp.asarray(x, jnp.float32)
    x_amax = jnp.max(jnp.abs(x_f32))
    x_scale = jnp.where(x_amax > 0, 448.0 / x_amax, 1.0)
    x_fp8 = (x_f32 * x_scale).astype(fp8_e4m3)

    k_f32 = jnp.asarray(kernel, jnp.float32)
    k_amax = jnp.max(jnp.abs(k_f32))
    k_scale = jnp.where(k_amax > 0, 57344.0 / k_amax, 1.0)
    k_fp8 = (k_f32 * k_scale).astype(fp8_e5m2)

    result = jnp.dot(x_fp8, k_fp8, preferred_element_type=jnp.float32)
    result = result / (x_scale * k_scale)
    return jnp.asarray(result, dtype)


# --- Linear ---


class Linear(nnx.Module):
    """General linear layer supporting expand, contract, and standard matmul.

    - ``in_features: int, out_features: int`` — standard 2D matmul
    - ``in_features: int, out_features: tuple`` — expand: ``[B,S,in] → [B,S,*out]``
    - ``in_features: tuple, out_features: int`` — contract: ``[B,S,*in] → [B,S,out]``
    """

    def __init__(
        self,
        in_features: int | tuple[int, ...],
        out_features: int | tuple[int, ...],
        dtype: jnp.dtype = jnp.bfloat16,
        weight_dtype: jnp.dtype = jnp.float32,
        kernel_axes: tuple[str, ...] = (),
        use_bias: bool = False,
        kernel_init_fn=None,
        use_fp8: bool = False,
        *,
        rngs: nnx.Rngs,
    ):
        self.dtype = dtype
        self.use_fp8 = use_fp8

        # Determine kernel shape and in/out axes for initialization
        if isinstance(in_features, tuple) and isinstance(out_features, int):
            # Contract mode: kernel [*in_features, out_features]
            kernel_shape = (*in_features, out_features)
            in_axis = tuple(range(len(in_features)))
            out_axis = len(in_features)
            self._mode = "contract"
        elif isinstance(in_features, int) and isinstance(out_features, tuple):
            # Expand mode: kernel [in_features, *out_features]
            kernel_shape = (in_features, *out_features)
            in_axis = 0
            out_axis = tuple(range(1, len(kernel_shape)))
            if len(out_axis) == 1:
                out_axis = out_axis[0]
            self._mode = "expand"
        else:
            # Standard 2D: kernel [in_features, out_features]
            kernel_shape = (in_features, out_features)
            in_axis = 0
            out_axis = 1
            self._mode = "standard"

        if kernel_init_fn is not None:
            kernel = kernel_init_fn(rngs.params(), kernel_shape, weight_dtype)
        else:
            init = _default_kernel_init()
            kernel = init(rngs.params(), kernel_shape, weight_dtype, in_axis=in_axis, out_axis=out_axis)

        self.kernel = nnx.Param(kernel, sharding=kernel_axes)

        if use_bias:
            bias_shape = out_features if isinstance(out_features, tuple) else (out_features,)
            self.bias = nnx.Param(jnp.zeros(bias_shape, dtype=weight_dtype))
        else:
            self.bias = None

    def __call__(self, x: jax.Array) -> jax.Array:
        x = jnp.asarray(x, self.dtype)
        kernel = jnp.asarray(self.kernel[...], self.dtype)

        if self._mode == "contract":
            # [B, S, heads, dim] @ [heads, dim, out] -> [B, S, out]
            y = jnp.einsum("bshd,hdo->bso", x, kernel)
        elif self._mode == "expand":
            if self.use_fp8 and kernel.ndim == 2:
                y = _fp8_matmul(x, kernel, self.dtype)
            elif kernel.ndim == 2:
                y = jnp.dot(x, kernel)
            else:
                # [B, S, in] @ [in, heads, dim] -> [B, S, heads, dim]
                y = jnp.einsum("bsi,ihd->bshd", x, kernel)
        else:
            # Standard 2D matmul
            if self.use_fp8:
                y = _fp8_matmul(x, kernel, self.dtype)
            else:
                y = jnp.dot(x, kernel)

        if self.bias is not None:
            y = y + jnp.asarray(self.bias[...], self.dtype)
        return y



# --- Activation function registry ---

ACT2FN = {
    "silu": jax.nn.silu,
    "gelu": jax.nn.gelu,
    "gelu_pytorch_tanh": functools.partial(jax.nn.gelu, approximate=True),
    "relu": jax.nn.relu,
}


# --- MLP (SwiGLU / gated MLP) ---


class MLP(nnx.Module):
    """Gated MLP block.

    Computes: output = down_proj(act_fn(gate_proj(x)) * up_proj(x))
    Default activation is SiLU (SwiGLU); configurable via config.hidden_activation.
    """

    def __init__(self, config: MinTextConfig, *, rngs: nnx.Rngs, mesh: jax.sharding.Mesh | None = None):
        dtype = config.jnp_dtype
        weight_dtype = config.jnp_weight_dtype
        use_fp8 = config.use_fp8
        self._mesh = mesh

        act_name = config.hidden_activation
        if act_name not in ACT2FN:
            raise ValueError(f"Unknown activation: {act_name!r}. Available: {sorted(ACT2FN.keys())}")
        self.act_fn = ACT2FN[act_name]

        self.gate = Linear(
            config.hidden_size,
            config.intermediate_size,
            dtype=dtype,
            weight_dtype=weight_dtype,
            kernel_axes=("embed", "mlp"),
            use_fp8=use_fp8,
            rngs=rngs,
        )
        self.up = Linear(
            config.hidden_size,
            config.intermediate_size,
            dtype=dtype,
            weight_dtype=weight_dtype,
            kernel_axes=("embed", "mlp"),
            use_fp8=use_fp8,
            rngs=rngs,
        )
        self.down = Linear(
            config.intermediate_size,
            config.hidden_size,
            dtype=dtype,
            weight_dtype=weight_dtype,
            kernel_axes=("mlp", "embed"),
            use_fp8=use_fp8,
            rngs=rngs,
        )

    def _constrain(self, x: jax.Array, spec: jax.sharding.PartitionSpec) -> jax.Array:
        """Apply sharding constraint if mesh is available."""
        if self._mesh is not None:
            return jax.lax.with_sharding_constraint(
                x, jax.sharding.NamedSharding(self._mesh, spec)
            )
        return x

    def __call__(self, x: jax.Array) -> jax.Array:
        gate_out = self.act_fn(self.gate(x))
        gate_out = self._constrain(gate_out, jax.sharding.PartitionSpec("data", None, "tensor"))
        gate_out = checkpoint_name(gate_out, "mlp_gate")
        up_out = self.up(x)
        up_out = self._constrain(up_out, jax.sharding.PartitionSpec("data", None, "tensor"))
        up_out = checkpoint_name(up_out, "mlp_up")
        result = self.down(gate_out * up_out)
        result = self._constrain(result, jax.sharding.PartitionSpec("data", None, "fsdp"))
        result = checkpoint_name(result, "mlp_down")
        return result
