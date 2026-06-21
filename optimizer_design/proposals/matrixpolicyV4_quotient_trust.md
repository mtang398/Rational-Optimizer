# matrixpolicyV4 Proposal: Quotient-Trust MatrixPolicy

Status: proposal only. V4 is the next candidate after the completed V3 E1 rerun rejected the horizontal confidence-tail variant.

## Postmortem That Motivates V4

The completed V3 E1 rerun is a negative result:

| Dataset | V3 final val loss | Original MatrixPolicy | Delta |
| --- | ---: | ---: | ---: |
| DCLM | 4.257245 +/- 0.003457 | 4.256224 | +0.001021 |
| FineWeb-Edu | 4.089219 +/- 0.006443 | 4.088240 | +0.000979 |
| FineWeb | 4.318981 +/- 0.009135 | 4.318581 | +0.000400 |
| Dolma-sample | 4.324203 +/- 0.004118 | 4.323851 | +0.000352 |
| C4 | 4.288422 +/- 0.015948 | 4.285119 | +0.003304 |

V3 kept the original MatrixPolicy recipe, then added partial horizontal gauge projection plus a late confidence-gated Muon tail. The loss did not improve on any dataset mean. That is enough to reject the late tail and to stop treating gradient projection as an obviously helpful default.

The useful signal is more specific:

1. Original MatrixPolicy remains the best observed RLB optimizer.
2. The role/depth early Muon window and group-stat policy should not be replaced.
3. Extra late matrix geometry is not helping; after the early window, the rational layer appears better served by stable AdamW-like consolidation.
4. Gauge information is still mathematically real, but V3 used it as a new direction/projection. V4 should use it only as a trust signal.

## A Priori Principle

For one RLB block:

```text
y = B R(A x)
```

and for group `g`, the positive scale action:

```text
A_g -> a_g A_g
B_g -> B_g / a_g
```

is approximately function-preserving under the homogeneous RLB radius. The gauge tangent is:

```text
v_g = (A_g, -B_g)
```

For matrix gradients `(G_A,g, G_B,g)`, the vertical gauge component is measured by:

```text
z_g = <G_A,g, A_g> - <G_B,g, B_g>
```

If `|z_g|` is large relative to total matrix-gradient energy, the raw update is spending step budget on parameterization scale rather than represented function. A quotient-aware optimizer should distrust that step. It does not need to invent a new update direction; it can simply reduce the risky matrix step and let the exact post-step gauge rebalance handle scale.

This is the core V4 change: use quotient geometry as a scalar trust gate, not as a projector and not as a late Muon source.

## V4 Update Rule

V4 starts from original `rational_matrix_policy_onpolicy`:

```text
Delta M = Delta AdamW(role/depth/group-scaled) + Delta Muon(early role/depth window)
```

Then, before applying the matrix-policy scale for group `g`, compute a dimensionless vertical ratio:

```text
num_g = |<G_A,g, A_g> - <G_B,g, B_g>|
den_g = sqrt((||G_A,g|| ||A_g||)^2 + (||G_B,g|| ||B_g||)^2) + eps
v_g = clamp(num_g / den_g, 0, 1)
```

Convert it to a conservative trust multiplier:

```text
tau_g = max(tau_min, exp(-lambda_v v_g^2))
```

Default proposal:

```text
lambda_v = 0.35
tau_min = 0.85
```

Apply the same paired trust multiplier to both input selector and output recombiner groups:

```text
a_mat(l,g,r,t) <- a_mat_original(l,g,r,t) * tau_g
mu(l,g,r,t)    <- mu_original(l,g,r,t) * sqrt(tau_g)
```

There is no late Muon tail. There is no gradient projection. There is no replacement of the old group pressure/gain/activity policy.

## Why This Is Simpler Than V3

V3 made two behavioral changes:

```text
gradient direction changed by partial horizontal projection
late-time update family changed by a confidence-gated Muon tail
```

V4 makes one conservative change:

```text
when the proposed matrix step is gauge-vertical, spend less matrix step budget
```

If gradients are already horizontal, `tau_g` is close to one and V4 reduces exactly to original MatrixPolicy. If gradients are mostly vertical, V4 cannot add a new harmful direction; it only damps the questionable update.

## A Posteriori Prediction

The E1/E2 pattern says MatrixPolicy's advantage is early and RLB-specific. Broader optimizers such as Muon catch up more at larger scale or longer runs, but the original MatrixPolicy still wins by reaching useful loss levels earlier. V4 therefore should strengthen early functional movement and avoid late perturbation.

Expected behavior:

1. final loss should tie or improve over original MatrixPolicy on E1;
2. gauge drift and `|log ||A_g|| - log ||B_g||||` variance should decrease;
3. update/weight RMS should become less spiky in groups with pressure imbalance;
4. optimizer-step time should be no slower than original MatrixPolicy after same-method speedups.

If these diagnostics do not move in the predicted direction, V4 should be rejected quickly.

## Same-Method Speed Plan

These speedups do not change the mathematical optimizer:

1. Cache RLB matrix views and selector-role metadata once per optimizer construction.
2. Cache per-step `(layer, group, role)` policy scalars so AdamW and Muon children share the same values.
3. Stop calling the Muon child optimizer after the last possible nonzero Muon window if no future Muon state can affect parameters.
4. Compute expensive telemetry only on log/eval capture steps; keep the scalar trust gate cheap and always on.
5. Use JSONL `train.optimizer_step_seconds` as the primary optimizer-speed metric; use full-step `summary.mean_seconds_per_step` only with node and restart notes.

The third item is only valid after the schedule proves every future Muon fraction is zero. Skipping Muon before a future nonzero window would change Muon state and is not allowed.

## E1 Acceptance Gate

V4 should be tested on E1 only before any E2 queueing:

```text
phase = E1_matrixpolicyV4_100m
method = rlb_matrixpolicyV4
activation = rlb_fused_fixed_strong_ffn
optimizer = matrixpolicyV4
rows = five datasets x three seeds
```

Acceptance requires:

1. mean final validation loss no worse than original MatrixPolicy on at least four of five datasets, with at least one clear win or clear token-to-target win;
2. no dataset mean worse by more than `0.0015` loss unless token-to-target improves materially;
3. optimizer-step time no worse than original MatrixPolicy by more than 2%;
4. clean accounting for Slurm restarts and slow-node full-step outliers.

If V4 misses the loss gate, do not queue E2.
