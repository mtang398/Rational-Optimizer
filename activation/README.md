# Activation

This directory implements the rational activation blocks used by RationalOPT. The research activation is the Rational Local Basis FFN, abbreviated RLB.

RLB is not presented as a standalone activation improvement. Its purpose is to expose structure that an optimizer can use: group normalization, rational curve coefficients, per-group activity statistics, and an exact positive gauge between `W_in` and `W_out`.

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

`R_g` is a learned rational function. The local-basis variants use a base rational curve plus trainable local odd/bump atoms around fixed centers.

RLB is a single-branch FFN:

```text
no GLU gate branch
no hidden SiLU value path
no SwiGLU-style value/gate split
```

## Positive Homogeneity And Gauge

The group normalization makes RLB positively homogeneous. If `a_g > 0`:

```text
z_g' = a_g z_g
r_g' = a_g r_g
u_g' = u_g
h_g' = a_g h_g
```

Therefore this matrix transform preserves the represented function:

```text
W_in[g]  <- a_g W_in[g]
W_out[g] <- W_out[g] / a_g
```

Generic optimizers still see different parameter norms, update scales, and conditioning. MatrixPolicy uses this gauge explicitly; AdamW and Muon do not.

## Optimizer-Visible Handles

RLB exposes these handles to the optimizer:

| component | role |
| --- | --- |
| `W_in` group rows | choose the normalized input domains seen by rational groups. |
| rational numerator/denominator parameters | set the nonlinear curve and derivative profile. |
| local basis coefficients | add local shape corrections. |
| `W_out` group columns | recombine rational features into the residual stream. |
| group RMS and derivative statistics | reveal active, saturated, and underused groups. |
| `W_in`/`W_out` gauge | can be rebalanced without changing the represented function. |

The activation code supports the forward path and the statistics needed by the optimizer wrapper.

The fused RLB path now exposes `_rlb_optimizer_stats` used by the training harness for paper telemetry:

```text
output RMS
derivative RMS
atom RMS
absolute moments
denominator probe margins
W_in/W_out gauge metrics
```

These are optimizer-diagnostic fields. They do not change the activation claim boundary: RLB is evaluated as activation plus optimizer-visible structure.

## Evidence Boundary

The current 3-seed evidence should be read as activation plus optimizer, not activation alone:

| task | RLB+MatrixPolicy gap vs SiLU+AdamW | RLB+AdamW readout |
| --- | ---: | --- |
| FineWeb | 0.159263 mean validation loss | slight mean gain, not enough to explain MatrixPolicy. |
| FineWeb-Edu | 0.154149 mean validation loss | one seed diverges; surviving seeds are near AdamW. |

This is why RLB changes should be evaluated with the full control set:

```text
SiLU+AdamW
RLB+AdamW
SiLU+Muon
RLB+Muon
RLB+MatrixPolicy (group-stat)
```

## Implementation Layout

```text
activation/rational_opt/  Python package and PyTorch fallback paths
activation/csrc/          CUDA/C++ extension sources
```

A6000 launchers set `RATIONAL_OPT_TORCH_FALLBACK=1` when the PyTorch implementation is the desired path. That fallback uses the same RLB math and is the path used by the current real-LM jobs.
