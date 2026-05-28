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
rational_transport_onpolicy          tested prototype
rational_jacobian_factored_onpolicy  negative probe
rational_layerwise_switch_onpolicy   negative probe
rational_layerwise_factored_switch_onpolicy prototype
```

These are used only with RLB activations. Standard comparison optimizers are `adamw`, `muon`, and the negative-probe ablation `factored_adamw`.

## Files

```text
function_space_rational_optimizer.py       fixed-grid function-space coefficient optimizer
factored_adamw.py                          AdamW with factored second moments for matrix ablations
switching_rational_optimizer.py            layer/depth/time coefficient optimizer switch prototype
onpolicy_balance_optimizer.py              live RLB group-gauge optimizer wrapper
quotient_onpolicy_optimizer.py             gauge-gradient quotient wrapper for RLB matrices
jacobian_onpolicy_optimizer.py             curve-aware RLB matrix preconditioner
quotient_jacobian_onpolicy_optimizer.py    quotient projection plus Jacobian preconditioning
adaptive_metric_onpolicy_optimizer.py      on-policy empirical metric prototype
transport_onpolicy_optimizer.py            rational-curve transport and matrix selector prototype
```

## RLB Gauge

RLB has an exact positive group gauge:

```text
W_in,g  <- c W_in,g
W_out,g <- W_out,g / c
```

The represented function is unchanged for positive `c`. This is the main structure the optimizer uses.

The transport prototype also uses a second exact rational-only amplitude gauge:

```text
R_g     <- a R_g
W_out,g <- W_out,g / a
```

It is implemented by scaling each rational group numerator and local atom coefficients, then compensating the corresponding output-matrix group. Denominator parameters are left unchanged, so the rational curve is amplitude-transported without changing its poles or local length scale.

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

`RationalTransportOnPolicyOptimizer` is a tested prototype. It keeps the adaptive metric path and adds optional rational amplitude transport, optional pressure preconditioning, and a coefficient-mode selector that can switch from aggressive early rational coefficient updates to safer late updates by layer using live on-policy gradient activity. In the 2026-05-27 runs, aggressive coefficient switching and late coefficient pullback did not beat the incumbent. The best transport setting was the conservative matrix-only mode: baseline coefficient dynamics with `matrix_strength = 0.65`, no live matrix stats, and no amplitude transport.


## Stagewise And Factored Probes

`FactoredAdamW` keeps AdamW's first moment and decoupled weight decay, but stores row/column factored second moments for large matrices. It was added as a high-leverage matrix optimizer ablation. In the seed-1337 probe, `rational_jacobian_factored_onpolicy` was far behind by step 1000 (`4.775638` loss, `118.586` PPL), so factored second moments are not recommended for this benchmark.

`SwitchingRationalOptimizer` blends AdamW coefficient updates with the function-space rational optimizer by layer depth and training progress, with an optional on-policy selector from the outer RLB wrapper. This implements the many-switch idea directly: different layers can enter the function-space coefficient optimizer at different times, and live rational-vs-matrix activity can pull the switch forward. The aggressive seed-1337 probe was worse by step 1000 (`4.221224` loss, `68.117` PPL), so scheduled coefficient switching is also not recommended yet.

The implementation remains useful as an ablation harness, but the empirical rule is now clear: do not spend risk on rational coefficient motion unless an on-policy acceptance test proves it beats tuned `RLB+AdamW`. The successful part of the older rational optimizer family is conservative matrix/gauge handling, not forced coefficient phases.

## Current Result

Three-seed aggregate on the 100M-token WikiText-103 task:

```text
AdamW + SiLU/SwiGLU                         3.610129  PPL 36.973  sec/step 0.188997
AdamW + RLB h3072                           3.606629  PPL 36.845  sec/step 0.205268
RLB h3072 + rational_onpolicy_balance       3.606226  PPL 36.831  sec/step 0.209027
RLB h3072 + rational_quotient_onpolicy      3.606664  PPL 36.847  sec/step 0.205176
RLB h3072 + rational_jacobian_onpolicy      3.605394  PPL 36.800  sec/step 0.204885
```

The Jacobian on-policy row improves the mean loss by `-0.004736` versus AdamW + SiLU/SwiGLU and by `-0.001236` versus AdamW on the same RLB activation. The 2026-05-27 transport probes did not supersede this row, so `rational_jacobian_onpolicy` remains the best measured rational-specific optimizer in the fixed-LR full sweep. On seed 1337, the incumbent h3072 Jacobian row was `3.614862`; the best transport row found was baseline coefficient dynamics plus matrix strength `0.65` at `3.615149`. Aggressive coefficient schedules, selector-triggered cooldown, and coefficient pullback all landed between `3.6198` and `3.6217`, so they are implemented as ablation controls rather than recommended defaults.

The high-LR follow-up changes the practical recommendation. With `--lr 5e-4 --min-lr 5e-5`, seed-1337 `RLB+AdamW` reaches `3.455792` / `31.683`, `SiLU+AdamW` reaches `3.456625` / `31.710`, and `RLB+rational_jacobian_onpolicy` reaches `3.459508` / `31.801`. The large gap versus the original fixed-LR SiLU baseline is therefore a learning-rate schedule effect. Future optimizer work should use tuned high-LR `RLB+AdamW` and tuned high-LR `SiLU+AdamW` as the hard controls.

The loss/PPL plots and compact probe tables are stored in `experiments/results/transport_optimizer_analysis_2026_05_27/` and `experiments/results/high_lr_optimizer_followup_2026_05_27/`. The design lesson from both sets of plots is that rational-specific matrix geometry is the robust signal; coefficient movement should be treated as a reversible, function-space-bounded proposal rather than a scheduled phase switch.
