# matrixpolicyV7 Proposal: Secant-Trust MatrixPolicy

Status: proposal only. This is not a live optimizer alias, not in the Slurm allowlist, and not queued. The live MatrixPolicy optimizer remains `rational_matrix_policy_onpolicy`.

## Motivation

V6 uses instantaneous signed function-space alignment. That is the cleanest mathematical target, but it may be noisy or too expensive if implemented with detailed activation/gradient hooks.

V7 keeps the same high-level goal but uses a temporal secant test: trust a matrix channel only when recent function-space movement behaved like a stable local descent direction.

This is not an engineering tweak. It changes the optimizer rule from static role/depth scaling to a local quadratic trust estimate for each RLB matrix channel.

## Local Quadratic Model

Let `s_g,r,t` be the estimated function-space movement produced by the accepted matrix update for group `g` and role `r` at step `t`:

```text
s_g,in,t  ~= B_g J_g (delta A_g x)
s_g,out,t ~= delta B_g h_g
```

Let `g_y,t` be the residual-stream loss gradient. A local quadratic model says:

```text
L(y + s) ~= L(y) + <g_y, s> + 0.5 kappa ||s||^2
```

A secant curvature estimate is:

```text
kappa_g,r = <s_g,r,t-1, g_y,t - g_y,t-1> / (||s_g,r,t-1||^2 + eps)
```

The predicted descent of the current candidate channel is:

```text
pred_g,r = - <g_y,t, s_g,r,t>
```

A good channel has positive predicted descent and moderate positive curvature. A channel with negative predicted descent or very high curvature should be damped.

## Proposed Rule

Start from the original V1 MatrixPolicy update. Maintain low-frequency EMA estimates of:

```text
pred_g,r
descent_ratio_g,r = pred_g,r / (||s_g,r|| ||g_y|| + eps)
kappa_g,r
```

Use a trust multiplier:

```text
trust_g,r = sigmoid(a * descent_ratio_g,r) * sqrt(kappa_ref / clamp(kappa_g,r, kappa_min, kappa_max))
trust_g,r <- center by joint geomean over all (g,r)
trust_g,r <- clip(trust_g,r, 0.75, 1.20)
```

If `kappa_g,r <= 0`, do not boost the channel. Either keep it at the centered baseline or damp mildly until the EMA becomes stable. Negative curvature estimates are too noisy to exploit in this small pilot regime.

Initial conservative settings:

```text
secant_every = 8 optimizer steps
ema_decay = 0.95
a = 2.0
kappa_min = 0.05 * median_positive_kappa
kappa_max = 20.0 * median_positive_kappa
scale_clip = [0.75, 1.20]
start = 0.08 training progress
end = 0.55 training progress
```

## Why This Is A Priori Reasonable

For a local quadratic objective, the step size along a direction is inversely proportional to directional curvature. V7 estimates that curvature in the represented function movement of RLB matrix channels, not in raw parameter coordinates. This makes it closer to the geometry MatrixPolicy is meant to exploit.

The rule is also conservative: V1 supplies the base update; V7 only adjusts trust when observed recent movement supports it.

## Why This Is A Posteriori Motivated

The results suggest that constant reallocation is the wrong abstraction:

```text
V3: late Muon/projection did not improve any E1 dataset mean
V4: balance signal clipped and centered away
V5: real role scaling happened, but only FineWeb-Edu improved
```

So the next design should not be another fixed role multiplier. It should adapt only where observed training dynamics say V1 is locally over- or under-stepping a specific RLB matrix channel.

## Runtime Argument

V7 can be cheaper to pilot than V6 because it can be estimated at telemetry intervals and with EMA state. It should not require an additional forward/backward pass. If implementation requires dense per-step function probes, reject V7 and do not run a full E1 test.

## Required Telemetry

```text
matrix_policy_v7_trust_mean_by_role
matrix_policy_v7_trust_std_by_role
matrix_policy_v7_kappa_mean_by_role
matrix_policy_v7_kappa_positive_frac_by_role
matrix_policy_v7_predicted_descent_mean_by_role
matrix_policy_v7_scale_clip_frac
```

Immediate rejection conditions:

```text
scale_clip_frac > 0.25 for most logged steps
positive curvature fraction collapses below 0.50 after warmup
runtime exceeds paired V1 by more than 5% without clear early AUC gain
paired V1 wins validation AUC on both pilot datasets
```
