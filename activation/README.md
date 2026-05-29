# Activation

This folder contains the rational activation implementation and CUDA extension. Optimizer policy does not live here; RLB modules expose the structure and statistics that the optimizer can use.

## Package

```text
activation/rational_opt/  Python package
activation/csrc/          CUDA/C++ extension sources
```

Build from the repo root:

```bash
.venv-cu128/bin/python setup.py build_ext --inplace
```

Import check:

```bash
PYTHONPATH=activation .venv-cu128/bin/python -c "import rational_opt; print(rational_opt.__all__)"
```

## RLB FFN Target

The optimizer work targets the fused no-GLU Rational Local Basis FFN:

```text
rlb_fused_fixed_strong_ffn        h = 3072
rlb_fused_fixed_strong_h2880_ffn  h = 2880
```

RLB computes grouped rational features:

```text
v = x W_in
s_g = sqrt(mean(v_g^2) + eps)
u_g = v_g / s_g
h_g = s_g R_g(u_g)
y = h W_out
```

This is not a GLU. There is no gate projection, no up branch, and no SiLU path inside the RLB layer. The SiLU/SwiGLU baseline is a separate activation in the training harness.

## Interface Used By Optimizers

| RLB item | optimizer use |
| --- | --- |
| `W_in` | matrix group for rational input-domain formation |
| `W_out` | matrix group for rational feature composition |
| group/layer metadata | layer-depth and matrix-role policy |
| live stats | diagnostics and optional group-stat policy |
| exact gauge | function-preserving post-step matrix rebalance |

The exact positive group gauge is:

```text
W_in,g  <- c W_in,g
W_out,g <- W_out,g / c
```

The current best optimizer uses MatrixPolicy AdamW plus short early Muon on RLB `W_in/W_out` matrices. Non-RLB weights and rational coefficients stay on AdamW by default.

## A6000 Fallback

Current A6000 launchers set:

```text
RATIONAL_OPT_TORCH_FALLBACK=1
```

That fallback uses the same RLB math in PyTorch and avoids relying on a CUDA image that may not be built for A6000.
