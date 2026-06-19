# matrixpolicyV2 Proposal: Horizontal Pressure Adam

This proposal replaces the earlier broad quotient-transport plan with a smaller
optimizer that is easier to justify and easier to test. The implementation name
is `matrixpolicyV2`; the old optimizer remains `rational_matrix_policy_onpolicy`
and is not overridden.

The goal is not to add a generic optimizer on top of RLB. The goal is to make
the RLB matrix update closer to the local function-space update that the current
results suggest is doing the useful work.

## Evidence From E1 And E2

Completed E1 M0/100M final-loss gaps against the best non-MatrixPolicy method:

| Dataset | MatrixPolicy | Best non-MP | Gap |
| --- | ---: | ---: | ---: |
| DCLM | 4.256224 | rlb_lion 4.305728 | 0.049505 |
| FineWeb-Edu | 4.088240 | rlb_lion 4.142669 | 0.054429 |
| FineWeb | 4.318581 | rlb_lion 4.367062 | 0.048481 |
| Dolma-sample | 4.323851 | rlb_lion 4.369254 | 0.045403 |
| C4 | 4.285119 | rlb_lion 4.335663 | 0.050544 |

Completed E2 M0/300M final-loss gaps against the best non-MatrixPolicy method:

| Dataset | MatrixPolicy | Best non-MP | Gap |
| --- | ---: | ---: | ---: |
| DCLM | 3.957627 | silu_lion 3.993430 | 0.035803 |
| FineWeb-Edu | 3.706480 | rlb_muon 3.738164 | 0.031684 |
| FineWeb | 3.965590 | rlb_muon 4.001245 | 0.035655 |
| Dolma-sample | 3.809853 | rlb_lion 3.842503 | 0.032650 |
| C4 | 3.882593 | rlb_muon 3.915858 | 0.033265 |

The pattern is stable but important:

- MatrixPolicy wins on every completed E1 and E2 dataset.
- The gap shrinks from about 0.05 loss at E1 to about 0.032-0.036 at E2.
- The strongest non-MatrixPolicy frontier at E2 is Lion/Muon, not AdamW.
- The current MatrixPolicy Muon fraction is zero late in training, so the durable
  advantage is not simply generic Muon.
- RLB and SiLU versions of the same generic optimizer are close, but the
  RLB-specific matrix policy separates clearly.

The design implication is that V2 should keep the RLB-specific matrix geometry
and remove pieces that are hard to explain: the permanent Muon child, duplicated
role/group scaling, and broad transport machinery that is not active in the
paper row.

## Mathematical Target

For one RLB group:

```text
z_g = A_g x
r_g = sqrt(mean(z_g^2) + eps)
u_g = z_g / r_g
h_g = r_g R_g(u_g)
y_g = B_g h_g
```

The optimizer should move the represented map `x -> y_g`, not arbitrary raw
coordinates. Locally, a useful approximation is:

```text
min_Delta <grad L, Delta> + (1 / 2 eta) ||J_g Delta||_D^2
```

where `J_g` is the Jacobian of the represented group function under the current
token distribution. Exact `J_g^T J_g` is too expensive, but RLB exposes cheap
block information:

- derivative RMS of `R_g` estimates sensitivity of `A_g`;
- output RMS of `R_g` estimates sensitivity of `B_g`;
- relative gradient pressure estimates whether `A_g` or `B_g` is bottlenecked;
- the `A_g, B_g` gauge identifies a non-functional scale direction.

matrixpolicyV2 is therefore a cheap quotient-aware block-natural AdamW update
for the RLB matrices.

## Quotient Direction

Ignoring the RMS floor, RLB has a groupwise positive scale symmetry:

```text
A_g -> s A_g
B_g -> B_g / s
```

The vertical tangent is:

```text
v_g = (A_g, -B_g)
```

A raw matrix gradient can waste update budget moving in this non-identifiable
representative direction. V2 projects the matrix gradient onto the horizontal
space before AdamW moments are updated:

```text
c = (<G_A, A_g> - <G_B, B_g>) / (||A_g||^2 + ||B_g||^2 + eps)
G_A <- G_A - c A_g
G_B <- G_B + c B_g
```

This is the first a priori reason V2 can beat the current policy: AdamW moments
stop integrating gauge-direction gradients that the post-step rebalance would
mostly undo later.

## Functional Matrix Metric

V2 applies a centered per-group sensitivity scale before the matrix step:

```text
scale_A,g = (geomean(derivative_rms) / derivative_rms_g)^alpha
scale_B,g = (geomean(output_rms) / output_rms_g)^alpha
```

The default is `alpha = 0.45` with clip `[0.70, 1.45]`, refreshed every 8 steps.
This approximates the diagonal block of the local function metric: groups with
large rational derivative or output gain get smaller coordinate steps because a
small raw movement already produces a large function movement.

V2 intentionally uses analytic rational-curve gains by default, not live matrix
stats, so it does not add forward-pass statistic overhead.

## On-Policy Pressure Metric

The wrapper already records:

```text
p_in,g  = rms(grad A_g) / rms(A_g)
p_out,g = rms(grad B_g) / rms(B_g)
p_rat,g = rational coefficient gradient activity
```

V2 uses only a mild centered pressure preconditioner:

```text
pressure_scale_A,g = centered_inverse(p_in,g)^0.20
pressure_scale_B,g = centered_inverse(p_out,g)^0.20
clip = [0.80, 1.25]
```

This is deliberately smaller than the old group-policy stack. It lets the
optimizer react to bottlenecked groups without creating another independent
hand-tuned group LR policy.

## Actual V2 Recipe

`matrixpolicyV2` keeps one AdamW child for RLB matrices and removes the Muon
child completely.

| Component | V2 default |
| --- | --- |
| Matrix optimizer | AdamW only |
| Matrix beta2 | 0.999 |
| Matrix LR multiplier | 3.0 |
| Matrix role/depth strength | 0.75 |
| Muon child | disabled |
| Old MatrixPolicy group gain/pressure/activity stack | disabled |
| Quotient gradient projection | strength 1.0 |
| Functional matrix metric | strength 0.45, clip [0.70, 1.45], every 8 steps |
| Pressure preconditioner | strength 0.20, clip [0.80, 1.25] |
| Gauge rebalance | same bounded RLB rebalance as old MatrixPolicy |
| Curve-amplitude transport | disabled |
| Coefficient Gram metric | disabled |
| Live matrix stats | disabled by default |

The resulting step order is:

```text
1. update on-policy pressure/activity EMAs
2. project A/B gradients off the positive-scale gauge direction
3. apply rational derivative/output matrix metric
4. apply mild pressure metric
5. step backbone AdamW and matrix AdamW
6. apply bounded W_in/W_out gauge rebalance and covariant state scaling
```

## Why This Is Simpler Than The Current Optimizer

The old row combines an AdamW matrix child, a Muon matrix child, role/depth LR
scaling, optional group policy, on-policy stats, and post-step gauge rebalance.
Even when the Muon fraction becomes zero, the child optimizer remains part of
the object graph.

V2 removes the most difficult part to justify, the generic Muon mixture, and
moves the new work into three RLB-specific operations:

```text
horizontal projection + functional matrix metric + pressure metric
```

Those are all direct approximations to the quotient trust-region objective.
They are not an arbitrary optimizer ensemble.

## Expected Outcomes

A priori prediction:

- V2 should improve early loss-vs-token if the old method was wasting update
  budget in gauge directions.
- V2 should be less sensitive to late saturation because AdamW moments are
  accumulated in a cleaner horizontal direction.
- V2 should be at least as fast as old MatrixPolicy on optimizer overhead because
  it has no Muon child and uses cached RLB matrix views.

A posteriori interpretation:

- If V2 beats old MatrixPolicy on E1, the main paper story becomes quotient
  matrix optimization for RLB rather than a hand-tuned Adam/Muon schedule.
- If V2 ties old MatrixPolicy but runs faster, it is still the cleaner default
  because it removes a hard-to-explain child optimizer.
- If V2 loses while RLB+Lion remains second, the missing ingredient is probably
  not generic Muon; it is the old per-group role policy, and the next ablation
  should reintroduce only that piece after the quotient projection.

## Test Plan

Run only `activation=rlb_fused_fixed_strong_ffn` and
`optimizer=matrixpolicyV2` for the matched E1 and E2 settings:

```text
E1: 5 datasets x 3 seeds = 15 jobs
phase = E1_matrixpolicyV2_100m
steps = 3050
train tokens = 100M

E2: 5 datasets x 3 seeds = 15 jobs
phase = E2_matrixpolicyV2_300m
steps = 9150
train tokens = 300M

method = rlb_matrixpolicyV2
```

The comparison targets are the completed E1 and E2 grids. The main readouts are:

- final validation loss by dataset and seed;
- token-to-target savings against old MatrixPolicy, RLB+Lion/Muon, and SiLU+AdamW;
- runtime from JSONL `summary.total_seconds`, not queue or requeue time;
- optimizer-step telemetry to verify lower overhead.
