# matrixpolicyV5 Proposal: Joint Functional-Metric MatrixPolicy

Status: implemented as separate optimizer choice `matrixpolicyV5`; E1 manifest queued on 2026-06-21 as Slurm jobs `716298`-`716312`. V4 E1 is fully closed and documented as a neutral/rejected result, so V5 is the active E1 optimizer proposal. Submission job IDs are recorded in `experiments/ICLR_RUN_COMMANDS.md`.

## Why V5 Exists

V5 is not an engineering optimization and not an optimizer-family mixture. It is a correction to the metric used by MatrixPolicy for RLB matrix updates.

The evidence chain is:

1. Original MatrixPolicy remains the paper anchor: it wins every completed E1 and E2 dataset mean.
2. V3 added horizontal projection and a late confidence-gated Muon tail, but was worse than original MatrixPolicy on all E1 dataset means.
3. V4 tried to balance `delta B_g h_g` against `B_g J_g delta A_g x`, but the implemented proxy saturated: every recorded `matrix_policy_functional_balance_log_ratio_*` value was clipped at `+0.47`.
4. Because the V4 group scale is geometrically centered inside each role, a role-wise constant clipped signal is normalized away. The observed V4 near-tie is therefore evidence that the V4 mechanism mostly failed to act, not evidence that RLB functional metrics are useless.

The V5 hypothesis is that MatrixPolicy needs a joint function-space metric for the `(A_g, B_g)` pair. The metric must preserve role-level scaling between input selector and output recombiner instead of centering each role independently.

## A Priori Model

For one RLB group:

```text
z_g = A_g x
r_g = rms(z_g)
u_g = z_g / r_g
h_g = r_g R_g(u_g)
y_g = B_g h_g
```

A small matrix perturbation moves the represented function as:

```text
delta y_g ~= B_g delta h_g + delta B_g h_g
```

For the input selector, `delta h_g` has two first-order components:

```text
radial:   R_g(u_g) delta r_g
tangent:  r_g J_g delta u_g
```

Under RMSNorm-like inputs and the RLB normalization, a unit-RMS `A_g` step has functional sensitivity controlled by both the rational output and derivative, then amplified by `B_g`:

```text
k_in,g  = rms(B_g) sqrt(rms(R_g)^2 + rms(J_g)^2)
```

A unit-RMS `B_g` step recombines the current RLB feature. The activation radius is approximated by the input-selector matrix norm, so:

```text
k_out,g = rms(A_g) rms(R_g)
```

This is the missing scale in V4. V4 used a raw gradient/update proxy and omitted the output-side dependence on `A_g`; it then clipped to a constant and got centered away.

## V5 Update Rule

Let the joint layer center be:

```text
c_l = 0.5 mean_g(log k_in,g) + 0.5 mean_g(log k_out,g)
```

V5 applies inverse square-root functional-sensitivity scaling:

```text
s_in,g  = clip(exp(-0.5 alpha (log k_in,g  - c_l)), s_min, s_max)
s_out,g = clip(exp(-0.5 alpha (log k_out,g - c_l)), s_min, s_max)
```

with the initial E1 test using:

```text
alpha = 1.0
s_min = 0.70
s_max = 1.45
start = progress 0.02
full = progress 0.20
no late decay in the first E1 test
```

The centering is joint across both roles. Therefore a layer-level role imbalance between `A_g` and `B_g` survives. This directly fixes the V4 failure mode, where role-wise constants were centered away before changing the real step.

Implementation details:

- The role-mean component of `s_in`/`s_out` multiplies the AdamW and Muon matrix learning rates for that role/layer.
- The within-role residual component is applied as a centered per-group gradient scale.
- Original MatrixPolicy role/depth mechanics, early Muon window, group-stat scaling, and post-step gauge rebalance are left intact.
- V1, V3, and V4 optimizer choices remain available; V5 is additive and does not overwrite earlier results.

## Why This Is Not An Engineering Tweak

V5 changes the metric for matrix updates. It asks how much the represented function moves per unit matrix step, then rescales the update in that metric. It does not change kernels, batching, scheduling, compiler flags, launch shape, or logging frequency.

The design is also not `Muon + X`. Muon remains exactly the existing early MatrixPolicy component. V5 only changes the RLB matrix metric used to scale the same MatrixPolicy update.

## Expected Outcomes

V5 should produce telemetry that V4 did not:

```text
matrix_policy_functional_metric_role_scale_mean_by_role
matrix_policy_functional_metric_group_scale_mean/std/min/max
matrix_policy_functional_metric_log_sensitivity_mean_by_role
```

Acceptance requires:

1. V5 must beat or tie original MatrixPolicy on at least three of five E1 dataset means, with no catastrophic dataset regression.
2. V5 must show nontrivial functional-metric role scaling instead of a clipped constant signal.
3. If V5 only changes runtime or only changes telemetry without loss/token-to-target improvement, reject it.
4. E2 should not be queued until E1 proves that the joint metric improves over V1 or gives a clearly useful token-to-target advantage.

## Test Plan

```text
phase = E1_matrixpolicyV5_100m
method = rlb_matrixpolicyV5
activation = rlb_fused_fixed_strong_ffn
optimizer = matrixpolicyV5
rows = five datasets x three seeds
budget = 3050 steps, about 100M train tokens per row
```

Submission should use one manifest row per Slurm job in two dependency chains, matching the V4 replacement shape, to limit preemption loss to one row.
