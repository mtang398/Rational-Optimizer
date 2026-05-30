# Activation

This folder implements rational activations used by the language-model experiments. The research activation is the Rational Local Basis FFN (RLB), a single-branch no-GLU FFN designed to expose optimizer-visible structure.

## RLB Definition

For a hidden vector split into `G` groups of width `m`:

```text
z = W_in x
z_g = group_g(z)
r_g = sqrt((1/m) ||z_g||_2^2 + eps)
u_g = z_g / r_g
h_g = r_g R_g(u_g)
y = W_out concat_g(h_g)
```

`R_g` is a learned rational function. In the local-basis variants it is the sum of a base rational curve and trainable local odd/bump atoms around fixed centers.

This is not a GLU:

```text
no multiplicative gate branch
no hidden SiLU path
no SwiGLU-style value/gate split
```

## Homogeneity And Gauge

The normalization makes RLB positively homogeneous at the group level. For any `a_g > 0`:

```text
z_g' = a_g z_g
r_g' = a_g r_g
u_g' = u_g
h_g' = a_g h_g
```

Therefore the matrix transform:

```text
W_in[g]  <- a_g W_in[g]
W_out[g] <- W_out[g] / a_g
```

preserves the represented function. This is the mathematical reason an RLB-specific optimizer can do something unavailable to a generic FFN optimizer: it can choose a better gauge representative without changing the function.

## Optimizer Handles

RLB exposes these handles to the optimizer:

| component | mathematical role |
| --- | --- |
| `W_in` group rows | choose the distribution of normalized inputs `u_g`. |
| rational numerator/denominator | set the base curve and derivative profile. |
| local basis coefficients | add local odd/bump corrections to the curve. |
| `W_out` group columns | select rational features for the residual stream. |
| group RMS and derivative statistics | reveal active, saturated, or underused groups. |

The activation code therefore supports not only forward computation, but also optimizer diagnostics such as output RMS, derivative RMS, coefficient activity, and group pressure.

## Evidence Boundary

RLB is not treated as a standalone activation win. The current real-corpus evidence says:

```text
FineWeb:     RLB+MatrixPolicy (group-stat) beats SiLU+AdamW by 0.160467 loss / 13.40 PPL.
FineWeb-Edu: RLB+MatrixPolicy (group-stat) beats SiLU+AdamW by 0.152964 loss / 9.70 PPL.
```

Plain `RLB+AdamW` is close to `SiLU+AdamW` on FineWeb but diverges on FineWeb-Edu. Therefore activation changes should be evaluated with the optimizer controls in [README.md](../README.md), not by comparing RLB against SiLU in isolation.

## Implementation Layout

```text
activation/rational_opt/  Python package and PyTorch fallback paths
activation/csrc/          CUDA/C++ extension sources
```

A6000 launchers set `RATIONAL_OPT_TORCH_FALLBACK=1` when needed so the PyTorch implementation of the same RLB math is used on nodes where the compiled extension is not the desired path.
