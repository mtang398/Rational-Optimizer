# Activation

This folder contains the rational activation implementation. The optimizer policy does not live here, but RLB is designed to expose structure that an optimizer can use.

## Method Context

RLB computes:

```text
v = x W_in
s_g = sqrt(mean(v_g^2) + eps)
u_g = v_g / s_g
h_g = s_g R_g(u_g)
y = h W_out
```

RLB is not a GLU. There is no gate branch and no hidden SiLU path inside the RLB layer.

## Optimizer Handles

| part | optimizer meaning |
| --- | --- |
| `W_in` | chooses the input domain seen by each rational group. |
| rational basis | supplies learnable nonlinear shape inside that domain. |
| `W_out` | recombines rational features back into the residual stream. |
| group scale | creates a positive gauge that can be balanced after updates. |
| live stats | expose activity and derivative pressure for diagnostics or future policies. |

The current MatrixPolicy optimizer only uses part of this structure. It has a verified WikiText lead, but it does not meaningfully win the completed saturated synthetic transfer tasks. A better optimizer should use rational activity, derivative pressure, group health, and layer role without changing the shared global LR schedule.

## Implementation

```text
activation/rational_opt/  Python package
activation/csrc/          CUDA/C++ extension sources
```

A6000 launchers currently set `RATIONAL_OPT_TORCH_FALLBACK=1`, using the PyTorch implementation of the same RLB math when the local compiled extension does not provide a usable A6000 image.
