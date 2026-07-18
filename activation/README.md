# Activation

This directory implements rational activation blocks used inside Transformer MLP sublayers. The research activation is Rational Latent Basis, abbreviated RLB.

RLB is not presented as a standalone activation improvement. Its purpose is to expose structure that an optimizer can use: group normalization, rational curve coefficients, per-group activity statistics, and a positive `W_in`/`W_out` scale gauge that is exact in the homogeneous radius and approximate under the stabilizing RMS floor.

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

RLB is single-branch inside the Transformer MLP sublayer:

```text
no GLU gate branch
no hidden SiLU value path
no SwiGLU-style value/gate split
```

## Positive Homogeneity And Gauge

With the homogeneous radius, or when the stabilizing floor is inactive, the group normalization makes RLB positively homogeneous. If `a_g > 0`:

```text
z_g' = a_g z_g
r_g' = a_g r_g
u_g' = u_g
h_g' = a_g h_g
```

Therefore this matrix transform preserves the represented function in that homogeneous setting:

```text
W_in[g]  <- a_g W_in[g]
W_out[g] <- W_out[g] / a_g
```

With the stabilized radius used in training, the same transform is a controlled approximate gauge on nondegenerate activations. Generic optimizers still see different parameter norms, update scales, and conditioning. MatrixPolicy uses this structure explicitly; AdamW and Muon do not.

## Optimizer-Visible Handles

RLB exposes these handles to the optimizer:

| component | role |
| --- | --- |
| `W_in` group rows | choose the normalized input domains seen by rational groups. |
| rational numerator/denominator parameters | set the nonlinear curve and derivative profile. |
| local basis coefficients | add local shape corrections. |
| `W_out` group columns | recombine rational features into the residual stream. |
| group RMS and derivative statistics | reveal active, saturated, and underused groups. |
| `W_in`/`W_out` gauge | can be rebalanced exactly in the homogeneous setting and approximately under the RMS floor. |

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

The current paper-facing evidence is the manifest-based E1 M0/100M suite, E2 M0/300M suite, and completed E8 learning-rate/weight-decay sensitivity grid on DCLM, FineWeb-Edu, FineWeb, Dolma-sample, and C4. MatrixPolicy rows use the validated live-statistic-corrected campaign under `../experiments/corrections/matrixpolicy_live_stats_20260712/`; its main and E8 validators pass `30/30` and `80/80` rows, respectively. Non-MatrixPolicy RLB controls use the completed global-rational control sweep. Both use `rlb_fused_global_rational`, which keeps the single-branch RLB wrapper and grouped P5/Q4 rational with trainable rational parameters limited to the numerator and denominator. Full E1 curves, token-savings, and checkpoint tables are in `../experiments/results/iclr26_e1_figures/` and `../experiments/results/iclr26_e1_token_savings_2026_06_12/`; full E2 final/runtime/token/curve tables are in `../experiments/results/iclr26_e2_*` and `../experiments/results/iclr26_e2_figures/`; the E8 paper artifacts are under `../paper/iclr_method_draft/`. This remains evidence for RLB plus MatrixPolicy and matched RLB controls, not an activation-only claim; RLB+ADeMaMix is retained as a divergent/early-stop negative row.

## Paper Use

This README defines the RLB layer and its optimizer-visible handles. It should not be used to make a standalone activation claim. The paper plan lives in `../experiments/ICLR_EXACT_RUN_PLAN.md`; component ablations of activation or optimizer pieces are late-stage explanatory experiments, not the way to choose the main benchmark setting.

## Implementation Layout

```text
activation/rational_opt/  Python package, compiled kernels, and PyTorch reference path
activation/csrc/          CUDA/C++ extension sources
```

A6000 launchers use the compiled extension path by default (`RATIONAL_OPT_TORCH_FALLBACK=0`). The CUDA venv must expose `ninja` on `PATH` before running paper jobs. Set `RATIONAL_OPT_TORCH_FALLBACK=1` only for local implementation debugging, not paper runs.
