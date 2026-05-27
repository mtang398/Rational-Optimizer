# Activation

This folder contains the rational activation implementation and CUDA extension.

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

## Version A 5/4

The reusable scalar rational family is:

```text
R_A(x) = (a0 + a1 x + a2 x^2 + a3 x^3 + a4 x^4 + a5 x^5)
         / (1 + |b1 x| + |b2 x^2| + |b3 x^3| + |b4 x^4|)
```

The CUDA extension provides fused forward/backward kernels for the rational paths used by the training code.

## RLB FFN

The active optimizer work targets the fused no-GLU Rational Local Basis FFN:

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

## Optimizer Interface

RLB modules expose group structure and on-policy statistics used by the optimizer code:

```text
abs_moments
raw_moments
num_gram
den_gram
atom_gram
atom_rms
```

The optimizer side also uses the exact positive group gauge:

```text
W_in,g  <- c W_in,g
W_out,g <- W_out,g / c
```

The activation implementation should remain optimizer-independent. Optimizer logic belongs in `optimizer_design/` and wiring belongs in `training/`.
