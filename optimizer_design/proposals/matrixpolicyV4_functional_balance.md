# matrixpolicyV4 Proposal: Functional-Balance MatrixPolicy

Status: completed and rejected/superseded after the E1-only replacement run on 2026-06-21. Replacement jobs `715054`-`715068` all completed with exit `0:0`; jobs `715054` and `715055` had `Restarts=1` before their clean final JSONL summaries, and jobs `715056`-`715068` had `Restarts=0`. V4 near-tied original MatrixPolicy, but its functional-balance signal clipped to a constant and mostly centered itself away. The next proposal is `matrixpolicyV5_joint_functional_metric.md`.

## Completed E1 Result

| Dataset | V4 final val loss | Original MatrixPolicy | Delta | Decision |
| --- | ---: | ---: | ---: | --- |
| DCLM | 4.255052 +/- 0.002431 | 4.256224 +/- 0.004972 | -0.001172 | near-tie |
| FineWeb-Edu | 4.088879 +/- 0.009448 | 4.088240 +/- 0.009434 | +0.000639 | near-tie |
| FineWeb | 4.317874 +/- 0.011026 | 4.318581 +/- 0.010914 | -0.000706 | near-tie |
| Dolma-sample | 4.323299 +/- 0.005749 | 4.323851 +/- 0.004565 | -0.000552 | near-tie |
| C4 | 4.287153 +/- 0.019124 | 4.285119 +/- 0.020677 | +0.002035 | near-tie/slightly worse |

The telemetry explains the neutral result. Across all 15 V4 rows, all `4590` recorded `matrix_policy_functional_balance_log_ratio_*` values were exactly clipped at `+0.47` (`clip_frac = 1.000`). Because the balance scale is then geometrically centered inside each role, this uniform clipped signal is normalized away rather than becoming a real A/B functional reallocation. V4 is therefore a useful negative result: the proposed proxy was underidentified and saturated, not a better MatrixPolicy.

## Evidence To Explain

The optimizer evidence currently available is original MatrixPolicy, rejected V2, and rejected V3.

Original MatrixPolicy is the strong anchor. It wins every completed E1 and E2 dataset mean. In E1 M0/100M, the gaps to the next current method are:

| Dataset | Original MatrixPolicy | Next current method | Gap |
| --- | ---: | ---: | ---: |
| DCLM | 4.256224 +/- 0.004972 | rlb_lion 4.305728 +/- 0.005836 | 0.049505 |
| FineWeb-Edu | 4.088240 +/- 0.009434 | rlb_lion 4.142669 +/- 0.006812 | 0.054429 |
| FineWeb | 4.318581 +/- 0.010914 | rlb_lion 4.367062 +/- 0.007532 | 0.048481 |
| Dolma-sample | 4.323851 +/- 0.004565 | rlb_lion 4.369254 +/- 0.005561 | 0.045403 |
| C4 | 4.285119 +/- 0.020677 | rlb_lion 4.335663 +/- 0.020917 | 0.050544 |

In E2 M0/300M, MatrixPolicy also wins every completed dataset and every seed, with mean gaps around `0.03` to `0.036` against the best non-MatrixPolicy aggregate method.

The negative evidence is just as important:

| Variant | Change | Result | Inference |
| --- | --- | --- | --- |
| V2 | removed original early role/depth policy and used broader quotient-style mechanics | worse than original by about `+0.012` to `+0.018` on E1 and `+0.003` to `+0.006` on E2 | the original early role/depth policy is doing real work |
| V3 | kept original policy but added partial horizontal projection and a late confidence-gated Muon tail | worse than original on all E1 dataset means, by `+0.000352` to `+0.003304` | extra late matrix geometry and projection are not the missing ingredient |

Therefore V4 should not be another generic optimizer mix, quotient projection, late Muon tail, damping gate, or runtime optimization. The next hypothesis must sharpen the part that is actually supported: RLB matrix roles need different functional step budgets.

## A Priori Model

For one RLB group, write the block as:

```text
y_g = B_g h_g
h_g = r_g R_g(u_g)
u_g = A_g x / r_g
r_g = rms(A_g x)
```

For a small optimizer step, the represented function moves as:

```text
delta y_g ~= delta B_g h_g + B_g J_g delta A_g x + B_g delta h_R,g
```

where `J_g` is the local derivative of the rational basis with respect to the normalized selector coordinate. The two matrix roles enter through different function-space channels:

```text
output recombiner movement:  delta B_g h_g
input selector movement:     B_g J_g delta A_g x
```

The original MatrixPolicy hard-codes a depth/role prior for these two channels. V4 keeps that prior but replaces the fixed per-group allocation with an on-policy functional-balance rule.

## V4 Design

Before the matrix step, estimate the functional movement that the current proposed input and output matrix updates would produce. For group `g`, define:

```text
m_in,g  ~= rms(B_g) rms(J_g) rms(x) rms(delta A_g)
m_out,g ~= rms(h_g) rms(delta B_g)
```

These are not generic gradient norms. They are approximations to the two terms in the RLB function linearization above. The required RLB statistics already exist conceptually in the model: derivative RMS for `J_g` and output/activation RMS for `h_g`. The proposed update RMS comes from the MatrixPolicy AdamW/Muon direction before applying the final group scalar.

Let the target role ratio be depth-dependent:

```text
target_g(l) = exp(beta_depth (d_l - 0.5))
```

where `target_g(l) = m_out,g / m_in,g`. Early layers should tolerate more input-selector movement; later layers should tolerate more output-recombiner movement. This keeps the original MatrixPolicy idea but makes it local and measured.

Define the imbalance:

```text
q_g = log((m_out,g + eps) / (m_in,g + eps)) - log target_g(l)
```

Then apply paired, centered role scalars:

```text
s_in,g  = exp(+kappa clip(q_g, -c, c))
s_out,g = exp(-kappa clip(q_g, -c, c))
```

with conservative bounds:

```text
s_in,g, s_out,g in [0.80, 1.25]
kappa = 0.25
c = log(1.6)
```

Interpretation:

- If output movement is too large relative to input movement, increase input-selector step and reduce output-recombiner step.
- If input movement is too large relative to output movement, reduce input-selector step and increase output-recombiner step.
- The paired product is approximately one, so V4 reallocates matrix step budget rather than simply damping or accelerating everything.

This is a real optimizer hypothesis: the RLB matrix update should equalize the predicted function-space contribution of `A_g` and `B_g`, subject to the original role/depth prior.

## Why This Is Not An Engineering Tweak

V4 changes the objective of the matrix policy. The scalar is not based on runtime, cache behavior, scheduler behavior, or a generic trust heuristic. It is based on the local functional derivative of the RLB block:

```text
delta y_g ~= delta B_g h_g + B_g J_g delta A_g x
```

The policy asks which role is actually moving the represented function too much or too little. That is the missing measurement in original MatrixPolicy: V1 knows roles and depth, but it does not measure per-group functional contribution balance.

## A Posteriori Rationale

The data points to role allocation, not optimizer-family mixing:

1. MatrixPolicy beats AdamW, Muon, Lion, SOAP-style, CAME-style, Schedule-Free, and ADeMaMix-style controls across E1 and E2. That argues the winning signal is RLB-specific structure.
2. V2 damaged performance when it removed the original early role/depth mechanics. That argues the early role/depth allocation must stay.
3. V3 damaged performance even while keeping the original policy, because it added projection and a late tail without improving the role allocation itself.
4. The E2 gaps are smaller than E1 but still consistent across all datasets and seeds, so the useful signal is early functional efficiency, not only final asymptotic loss.

V4 therefore targets early functional efficiency directly. It should improve token-to-target if the diagnosis is right, even if final loss only ties.

## What To Test

V4 should be tested on E1 first, and only after the implementation matches this proposal. No V4 E2 or V5 jobs should be queued before V4 E1 is analyzed.

```text
phase = E1_matrixpolicyV4_100m
method = rlb_matrixpolicyV4
activation = rlb_fused_fixed_strong_ffn
optimizer = matrixpolicyV4
rows = five datasets x three seeds
```

Acceptance requires:

1. V4 beats or ties original MatrixPolicy final validation loss on at least four of five E1 datasets.
2. V4 improves token-to-target or validation-loss AUC on at least three datasets.
3. The measured `m_out/m_in` imbalance moves toward the depth target during the early MatrixPolicy window.
4. If V4 only changes runtime and not the functional-balance telemetry, reject it.

## V5 Constraint

V5 should not be designed as an engineering optimization either. It should only be proposed after V4 E1 results reveal one of these failure modes:

| V4 observation | V5 direction |
| --- | --- |
| functional balance improves but loss does not | couple matrix balance to rational coefficient update timing |
| functional balance improves early token-to-target but hurts final loss | make the balance window shorter and decay into original MatrixPolicy |
| balance telemetry does not move | replace the proxy with a better function-space estimator, not a runtime tweak |
| V4 beats original on E1 | run E2 before designing V5 |

V5 must be posterior to V4 evidence, not pre-queued speculation.
