# Optimizer Design

This folder contains optimizer components for the no-GLU Rational Local Basis FFN (RLB). The purpose is not to tune a generic training schedule. The purpose is to build an on-policy optimizer that uses rational-specific structure and beats both `SiLU/SwiGLU+AdamW` and `RLB+AdamW` under the same schedule.

## Target Layer

```text
v = x W_in
s_g = sqrt(mean(v_g^2) + eps)
u_g = v_g / s_g
h_g = s_g R_g(u_g)
y = h W_out
```

RLB has no GLU gate and no SiLU inside the FFN. Optimizers in this folder are for RLB rows only unless explicitly marked as an ablation.

## Implemented Components

| component | status | purpose |
| --- | --- | --- |
| `FunctionSpaceRationalOptimizer` | ablation utility | coefficient updates measured by function change on probe points |
| `RationalOnPolicyBalanceOptimizer` | tested | live group-gauge rebalance after a child optimizer step |
| `RationalQuotientOnPolicyOptimizer` | tested | remove exact matrix gauge-gradient direction before stepping |
| `RationalJacobianOnPolicyOptimizer` | best fixed-LR row | curve-aware matrix preconditioner plus gauge rebalance |
| `RationalQuotientJacobianOnPolicyOptimizer` | prototype | quotient projection plus Jacobian matrix scaling |
| `RationalAdaptiveMetricOnPolicyOptimizer` | prototype | collect on-policy activation/derivative metrics |
| `RationalTransportOnPolicyOptimizer` | tested prototype | rational amplitude transport and matrix/coefficient selectors |
| `FactoredAdamW` | negative probe | Adafactor-style matrix second moments; failed badly here |
| `SwitchingRationalOptimizer` | negative probe harness | layer/depth/time coefficient switching; scheduled switching failed |

Current optimizer names exposed by training:

```text
rational_onpolicy_balance
rational_quotient_onpolicy
rational_jacobian_onpolicy
rational_quotient_jacobian_onpolicy
rational_adaptive_metric_onpolicy
rational_transport_onpolicy
rational_jacobian_factored_onpolicy
rational_layerwise_switch_onpolicy
rational_layerwise_factored_switch_onpolicy
```

## Rational Structure

RLB has an exact positive group gauge:

```text
W_in,g  <- c W_in,g
W_out,g <- W_out,g / c
```

This preserves the represented function for positive `c`. Optimizers can use it to pick a numerically better representative of the same function.

RLB matrix gradients also see different local rational geometry:

```text
W_in gradient path:   R'_g(u)
W_out gradient path:  R_g(u)
```

Rational coefficients are not ordinary dense weights. A small coefficient update changes a scalar function over the active `u` distribution, and denominator/pole behavior can make nominally small parameter steps unsafe.

## Current Empirical State

Three-seed fixed-LR result on the 100M-token WikiText-103 benchmark:

| row | loss | PPL | gap vs SiLU+AdamW | gap vs RLB+AdamW |
| --- | ---: | ---: | ---: | ---: |
| AdamW + SiLU/SwiGLU | 3.610129 | 36.973 | +0.000000 | +0.003500 |
| AdamW + RLB h3072 | 3.606629 | 36.845 | -0.003500 | +0.000000 |
| RLB + `rational_onpolicy_balance` | 3.606226 | 36.831 | -0.003903 | -0.000403 |
| RLB + `rational_quotient_onpolicy` | 3.606664 | 36.847 | -0.003465 | +0.000035 |
| RLB + `rational_jacobian_onpolicy` | 3.605394 | 36.800 | -0.004736 | -0.001236 |

`rational_jacobian_onpolicy` is the best verified fixed-LR rational-specific optimizer, but the gain is too small for the project goal.

High-LR controls are diagnostic only. They showed that a large apparent loss drop can come from schedule correction: seed-1337 high-LR `RLB+AdamW` reached `3.455792`, high-LR `SiLU+AdamW` reached `3.456625`, and high-LR `RLB+rational_jacobian_onpolicy` reached `3.459508`. That means the large high-LR gap is not a rational optimizer win.

## Lessons From Failed Probes

Matrix-side rational geometry is consistently the least bad signal. Jacobian scaling and conservative gauge balancing helped slightly; transport matrix-only settings were close to the incumbent.

Scheduled coefficient motion is the wrong lever so far. Aggressive coefficient transport, late pullback, and layerwise switches changed the rational function path too much and did not recover later.

Factored matrix second moments are not suitable in this setup. `rational_jacobian_factored_onpolicy` was at `4.775638` loss by step 1000, far behind normal AdamW-style moments.

ASAM did not fix the small-gap problem. It made fixed-LR `SiLU+AdamW` and `RLB+AdamW` slightly worse in the probe.

## Next Design: On-Policy Functional Trust

The next optimizer should not be another step-index schedule. It should be a per-layer/per-group controller driven by live RLB signals.

Proposed update decomposition:

```text
1. collect on-policy u, R(u), R'(u), denominator margin, feature scale
2. split matrix gradients into horizontal function directions plus exact gauge direction
3. use gauge rebalance to choose a stable representative without changing the function
4. scale W_in groups by derivative metric and W_out groups by output-feature metric
5. build tiny per-group coefficient metrics from J_coeff R(u)
6. damp coefficient solves by denominator risk and clip by predicted function change
7. choose matrix-only, coefficient-natural, gauge-only, or freeze mode per group from live signals
```

The controller should prefer coefficient updates only when all of these are true:

```text
gradient agreement is positive
predicted function-space improvement is larger than matrix-only update
predicted change in R(u) is inside trust radius
denominator margin is safe
coefficient activity is not already dominating matrix activity
```

Otherwise the optimizer should spend the step on matrix preconditioning and gauge balancing. This keeps the optimizer rational-specific while avoiding the failure mode seen in scheduled coefficient switches.

## Design Rule

A new optimizer is interesting only if it beats both hard controls under the same schedule:

```text
SiLU/SwiGLU + AdamW
RLB + AdamW
```

Beating an undertrained fixed-LR baseline is not enough. The optimizer must create an RLB-specific advantage, not borrow improvement from a generic scheduler.
