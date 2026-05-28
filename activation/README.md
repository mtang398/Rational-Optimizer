# Activation

This folder contains the rational activation implementation and CUDA extension. Optimizer policy does not live here; RLB modules expose the structure and statistics that optimizers use.

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

## A6000 Fallback

The local compiled extension did not provide a usable A6000 kernel image during these runs. Training launchers therefore set:

```text
RATIONAL_OPT_TORCH_FALLBACK=1
```

That fallback keeps the same RLB math in PyTorch and is slower, so A6000 runs use `--batch-size 16 --grad-accum 2` to preserve the same global tokens per step without OOM.

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

This is not a GLU. There is no gate projection, no up branch, and no SiLU path inside the RLB layer.

## Interface Used By The Optimizer

| RLB item | optimizer use |
| --- | --- |
| `W_in` | matrix group for rational input-domain formation |
| `W_out` | matrix group for rational feature composition |
| group/layer metadata | layer-depth and matrix-role policy |
| live stats | optional on-policy damping and diagnostics |
| exact gauge | function-preserving post-step matrix rebalance |

The exact positive group gauge is:

```text
W_in,g  <- c W_in,g
W_out,g <- W_out,g / c
```

The current best optimizer uses MatrixPolicy AdamW plus short early Muon on RLB `W_in/W_out` matrices. Non-RLB weights and rational coefficients remain on AdamW.
