# Activation

This folder contains the rational activation implementation. The optimizer policy does not live here, but the activation is designed to expose structure that an optimizer can use.

## RLB Layer

RLB is a no-GLU rational FFN:

```text
v = x W_in
s_g = sqrt(mean(v_g^2) + eps)
u_g = v_g / s_g
h_g = s_g R_g(u_g)
y = h W_out
```

There is no gate branch and no hidden SiLU path inside the RLB layer. The rational function `R_g` is evaluated on normalized group inputs, then scaled back by the group RMS.

## Optimizer-Visible Structure

| component | optimizer handle |
| --- | --- |
| `W_in` | Sets the normalized domain seen by each rational group. |
| rational coefficients | Shape local odd/bump rational features. |
| `W_out` | Selects and mixes rational features into the residual stream. |
| group RMS | Provides a scale signal for activity and saturation. |
| positive gauge | Allows function-preserving rebalance between input and output matrices. |

The positive gauge is central to the current research plan. If one group of `W_in` is multiplied by `a > 0`, the normalized input `u_g` is unchanged and the group output scales by `a`. Dividing the matching `W_out` columns by `a` preserves the layer function. This makes gauge stress a direct test of whether an optimizer understands rational structure or is sensitive to arbitrary matrix scaling.

## Implementation Boundaries

```text
activation/rational_opt/  Python package and PyTorch fallback paths
activation/csrc/          CUDA/C++ extension sources
```

A6000 launchers currently set `RATIONAL_OPT_TORCH_FALLBACK=1` so runs can use the PyTorch implementation of the same RLB math when the local compiled extension is not the right path for the node. CPU forward tests can still hit CUDA-only rational extension paths; use GPU training runs for end-to-end activation validation.
