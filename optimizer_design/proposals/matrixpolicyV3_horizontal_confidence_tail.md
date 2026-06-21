# matrixpolicyV3 Proposal: Horizontal Confidence-Tail MatrixPolicy

Status: rejected/superseded after the completed E1 rerun on 2026-06-20. The file is retained as negative evidence; it does not replace the paper anchor `rational_matrix_policy_onpolicy`, and it is not the active next proposal.

## Completed E1 Readout

V3 missed its acceptance gate. It was slightly worse than original MatrixPolicy on every E1 dataset mean:

| Dataset | V3 final val loss | Original MatrixPolicy | Delta | V3 optimizer-step s |
| --- | ---: | ---: | ---: | ---: |
| DCLM | 4.257245 +/- 0.003457 | 4.256224 | +0.001021 | 0.091257 |
| FineWeb-Edu | 4.089219 +/- 0.006443 | 4.088240 | +0.000979 | 0.091290 |
| FineWeb | 4.318981 +/- 0.009135 | 4.318581 | +0.000400 | 0.085624 |
| Dolma-sample | 4.324203 +/- 0.004118 | 4.323851 | +0.000352 | 0.087206 |
| C4 | 4.288422 +/- 0.015948 | 4.285119 | +0.003304 | 0.084667 |

Full-step timing is contaminated by node/restart variation in several rows, including row 7 with `Restarts=1` and slow-node rows 5, 9, 11, and 13, so V3 is judged mainly by final loss and log-step optimizer timing. The result rejects the late confidence-gated Muon tail and the default partial horizontal projection. The next proposal is `matrixpolicyV4_quotient_trust.md`, which keeps quotient geometry only as a scalar trust gate.

## Why V2 Is Rejected

The completed V2 E1/E2 reruns give a useful negative result:

| Phase | V2 vs original MatrixPolicy | V2 vs best non-MatrixPolicy |
| --- | ---: | ---: |
| E1 M0/100M | worse by `+0.012` to `+0.018` loss | still better by `0.027` to `0.038` loss |
| E2 M0/300M | worse by `+0.003` to `+0.006` loss | still better by `0.027` to `0.033` loss |

The inference is not that quotient geometry is useless. V2 still beats every non-MatrixPolicy baseline, so the RLB-specific geometry is real. The failure is that V2 removed the original early MatrixPolicy mechanics that are doing important early optimization work: the role/depth AdamW-Muon matrix policy and the group pressure/gain/activity scaling.

V3 therefore should not be another broad optimizer swap. It should keep the old winning policy and add one mathematically targeted correction where the current design is weakest.

## Model Geometry

For one RLB block, write

```text
y = B R(A x)
```

where `A = W_in`, `B = W_out`, and `R` is grouped. For group `g`, the positive scale action

```text
A_g -> a_g A_g
B_g -> B_g / a_g
```

is approximately function-preserving under the homogeneous RLB radius. The gauge tangent is

```text
v_g = (A_g, -B_g)
```

A matrix gradient is horizontal when its component along this tangent is removed:

```text
<g_A,g, A_g> - <g_B,g, B_g> = 0
```

The existing wrapper already has this horizontal projection. V2 used it but discarded the old policy. V3 keeps the old policy and applies only a modest horizontal projection.

## A Posteriori Constraints From E1/E2

The result pattern says:

1. Original MatrixPolicy is still the best observed optimizer.
2. V2 loses most at E1 and much less at E2, so V2 mainly damaged early training, not the long-run RLB advantage.
3. The old early Muon component should not be deleted. It should be retained where the RLB group state is confident and retired where live on-policy pressure says it is unsafe.
4. Runtime must be tracked from JSONL summary fields, not Slurm elapsed time, because requeues and dependencies contaminate scheduler elapsed.

## V3 Update Rule

V3 keeps the original MatrixPolicy step:

```text
Delta M = Delta AdamW(role/depth/group-scaled) + Delta Muon(early role/depth window)
```

and adds two limited changes.

### 1. Horizontal Gauge Projection

Before child optimizer steps, V3 projects a partial amount of the RLB matrix gradient away from the gauge tangent:

```text
c_g = alpha_q (<g_A,g, A_g> - <g_B,g, B_g>) / (||A_g||^2 + ||B_g||^2 + eps)
g_A,g <- g_A,g - c_g A_g
g_B,g <- g_B,g + c_g B_g
```

Default:

```text
alpha_q = 0.35
```

This is deliberately not full projection. The RLB radius floor, weight decay, and finite step size make the gauge only locally exact, so a partial projection is the conservative choice.

### 2. Confidence-Gated Muon Tail

Original MatrixPolicy has a time-only Muon window that decays to zero by progress `0.36`. V3 adds a small later Muon tail, but only where live on-policy statistics indicate low pressure and low rational coefficient excess activity.

Let

```text
p_g = mean |log p_in,g - log p_out,g|
a_g = relu((log p_rat,g - 0.5(log p_in,g + log p_out,g) - target) / width)
c_g = exp(-lambda_p p_g - lambda_a a_g)
```

Then

```text
mu_V3(l,r,t) = max(mu_original(l,r,t), mu_tail * window(t) * rho(l,r) * c_g)
```

Defaults:

```text
mu_tail = 0.12
window on: 0.24 -> 0.48
window off: 0.90 -> 1.00
lambda_p = 0.35
lambda_a = 0.35
```

This is not `Muon + MatrixPolicy` as an unprincipled mixture. The tail is only active after the original Muon component begins decaying, and it is suppressed by exactly the same RLB pressure/activity variables that define whether the group representative is trustworthy.

## Same-Method Speedup

Two speedups are used without changing the mathematical method:

1. The gauge wrapper caches stable RLB matrix views instead of repeatedly rebuilding `view(...).permute(...)` objects.
2. MatrixPolicy now caches the Muon mixture fraction by `(layer, selector, role)` for each step. Adam and Muon child groups share the same value instead of recomputing the same live-pressure scalar.

Runtime will be judged from JSONL fields:

```text
summary.total_seconds
summary.mean_seconds_per_step
summary.tokens_per_second
train.optimizer_step_seconds
```

Scheduler elapsed is not used as method runtime because dependencies and requeues contaminate it.

## V3 E1 Test

Run a matched isolated E1 M0/100M suite:

```text
phase = E1_matrixpolicyV3_100m
method = rlb_matrixpolicyV3
activation = rlb_fused_fixed_strong_ffn
optimizer = matrixpolicyV3
rows = five datasets x three seeds
```

Extra args are the original winning MatrixPolicy group-stat settings plus the V3 quotient/tail settings:

```text
--rational-matrix-policy-backbone-optimizer adamw
--rational-matrix-policy-adam-lr-scale 3.0
--rational-matrix-policy-group-gain-strength 0.20
--rational-matrix-policy-group-pressure-strength 0.10
--rational-matrix-policy-group-activity-damping 0.20
--rational-matrix-policy-group-start 0.02
--rational-matrix-policy-group-end 0.30
--rational-matrix-policy-group-min-scale 0.75
--rational-matrix-policy-group-max-scale 1.35
--matrixpolicy-v3-quotient-strength 0.35
--matrixpolicy-v3-muon-tail-strength 0.12
--matrixpolicy-v3-muon-tail-start 0.24
--matrixpolicy-v3-muon-tail-end 0.48
--matrixpolicy-v3-muon-tail-decay-start 0.90
--matrixpolicy-v3-muon-tail-decay-end 1.00
--matrixpolicy-v3-muon-tail-pressure-weight 0.35
--matrixpolicy-v3-muon-tail-activity-weight 0.35
```

## Decision Criteria

V3 is useful only if at least one of these is true:

1. It beats original MatrixPolicy mean final validation loss on E1.
2. It ties original within seed noise but improves early token-to-target or runtime.
3. It gives clear telemetry support: lower gauge drift or better update/weight ratio at equal or better loss.

If V3 is worse than original with no runtime win, reject it and treat the negative result as evidence that the original time-local Muon decay is already near the best E1 tradeoff.
