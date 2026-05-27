# Optimizer Design

This folder contains optimizer components for the no-GLU Rational Local Basis FFN, abbreviated RLB.

## Target Layer

```text
v = x W_in
s_g = sqrt(mean(v_g^2) + eps)
u_g = v_g / s_g
h_g = s_g R_g(u_g)
y = h W_out
```

RLB has no GLU gate and no SiLU inside the FFN. The optimizers in this folder are designed for this layer, not for SiLU.

## Accepted Optimizers

```text
rational_onpolicy_balance
rational_quotient_onpolicy
rational_jacobian_onpolicy
rational_quotient_jacobian_onpolicy
rational_adaptive_metric_onpolicy
```

These are used only with RLB activations. Standard comparison optimizers are `adamw` and `muon`.

## Files

```text
function_space_rational_optimizer.py       fixed-grid function-space coefficient optimizer
onpolicy_balance_optimizer.py              live RLB group-gauge optimizer wrapper
quotient_onpolicy_optimizer.py             gauge-gradient quotient wrapper for RLB matrices
jacobian_onpolicy_optimizer.py             curve-aware RLB matrix preconditioner
quotient_jacobian_onpolicy_optimizer.py    quotient projection plus Jacobian preconditioning
adaptive_metric_onpolicy_optimizer.py      on-policy empirical metric prototype
```

## RLB Gauge

RLB has an exact positive group gauge:

```text
W_in,g  <- c W_in,g
W_out,g <- W_out,g / c
```

The represented function is unchanged for positive `c`. This is the main structure the optimizer uses.

## Function-Space Coefficient Updates

Rational coefficients do not behave like ordinary dense matrix weights. A small coefficient change changes a function over the active scalar domain. `FunctionSpaceRationalOptimizer` normalizes numerator, denominator, atom, and center updates by their functional effect on a probe grid before applying trust clipping and curve decay.

## On-Policy Balance

`RationalOnPolicyBalanceOptimizer` wraps child optimizers. It updates ordinary weights with the child optimizer, then applies a function-preserving group-gauge transform. Its live signals are:

```text
rational output scale
rational derivative scale
input-side gradient pressure
output-side gradient pressure
rational coefficient gradient activity
```

The correction is layer-specific and time-varying.

## Quotient On-Policy Optimizer

`RationalQuotientOnPolicyOptimizer` adds a pre-step projection. The exact gauge direction for group `g` is:

```text
(W_in,g, -W_out,g)
```

Given gradients `(G_in,g, G_out,g)`, the vertical gauge coefficient is:

```text
a_g = (<G_in,g, W_in,g> - <G_out,g, W_out,g>)
      / (||W_in,g||^2 + ||W_out,g||^2 + eps)
```

It removes that component:

```text
G_in,g  <- G_in,g  - a_g W_in,g
G_out,g <- G_out,g + a_g W_out,g
```

Then the child optimizers step, and the on-policy balance transform selects the stable group-scale representative.

## Jacobian On-Policy Optimizer

`RationalJacobianOnPolicyOptimizer` is the current best measured optimizer row. It keeps the on-policy balance mechanism and adds an RLB-specific matrix gradient scaling step.

For each rational group `g`, the optimizer probes the current learned curve and estimates:

```text
D_g = mean |R'_g(t)|
O_g = sqrt(mean R_g(t)^2)
```

The input matrix group receives an inverse relative derivative scale:

```text
G_in,g <- G_in,g * clip((geomean(D) / D_g)^strength, min_scale, max_scale)
```

The output matrix group receives an inverse relative output-feature scale:

```text
G_out,g <- G_out,g * clip((geomean(O) / O_g)^strength, min_scale, max_scale)
```

This is specific to RLB because `W_in` gradients pass through `R'_g`, while `W_out` gradients see the feature amplitude `R_g`. The tested full-run setting used `strength = 0.5`, scale clip `[0.5, 2.0]`, and recomputation every 5 optimizer steps.

## Prototype Optimizers

`RationalQuotientJacobianOnPolicyOptimizer` projects gradients away from the exact gauge direction, then applies the Jacobian preconditioner. The seed-1337 h3072 probe finished at `3.615571`, which is better than AdamW on the same row but worse than the existing Jacobian row at `3.614862`.

`RationalAdaptiveMetricOnPolicyOptimizer` enables RLB modules to collect empirical output and derivative gains on the actual normalized training activations. It also contains an optional empirical Gram solve for rational coefficients, but this is off by default because probes showed it over-conditioned the small coefficient tensors.

## Current Result

Three-seed aggregate on the 100M-token WikiText-103 task:

```text
AdamW + SiLU/SwiGLU                         3.610129  PPL 36.973  sec/step 0.188997
AdamW + RLB h3072                           3.606629  PPL 36.845  sec/step 0.205268
RLB h3072 + rational_onpolicy_balance       3.606226  PPL 36.831  sec/step 0.209027
RLB h3072 + rational_quotient_onpolicy      3.606664  PPL 36.847  sec/step 0.205176
RLB h3072 + rational_jacobian_onpolicy      3.605394  PPL 36.800  sec/step 0.204885
```

The Jacobian on-policy row improves the mean loss by `-0.004736` versus AdamW + SiLU/SwiGLU and by `-0.001236` versus AdamW on the same RLB activation. The 2026-05-27 prototypes did not supersede this row, so `rational_jacobian_onpolicy` remains the recommended optimizer.
