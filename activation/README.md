# Activation

This folder contains the rational activation implementation and CUDA extension. Optimizer logic does not live here; RLB modules only expose the structure and statistics that the optimizer uses.

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

The current optimizer work targets the fused no-GLU Rational Local Basis FFN:

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

This is a rational FFN, not a GLU variant. There is no gate projection, no up branch, and no SiLU path inside the RLB layer.

## Interface Used By The Optimizer

The current optimizer is `rational_matrix_policy_onpolicy`, implemented outside this folder. It uses RLB's structure in three ways:

| RLB item | optimizer use |
| --- | --- |
| `W_in` | matrix group for rational input-domain formation |
| `W_out` | matrix group for rational feature composition |
| group/layer metadata | layer-depth and matrix-role policy |
| live stats | optional on-policy damping and diagnostics |
| exact gauge | function-preserving post-step matrix rebalance |

RLB modules expose optimizer statistics such as:

```text
abs_moments
raw_moments
num_gram
den_gram
atom_gram
atom_rms
output_rms
derivative_rms
```

The exact positive group gauge is:

```text
W_in,g  <- c W_in,g
W_out,g <- W_out,g / c
```

`rational_matrix_policy_onpolicy` applies that gauge after optimizer steps. The activation implementation should remain optimizer-independent: activation code exposes structure and stats; optimizer policy belongs in `optimizer_design/`; training wiring belongs in `training/`.

## Current Optimizer Context

The current best uses RLB `W_in/W_out` matrices with MatrixPolicy AdamW plus a short early Muon phase. The non-RLB backbone and rational coefficients remain on AdamW. The global LR schedule is unchanged.
