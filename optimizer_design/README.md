# Optimizer Design

This folder contains the RLB-specific optimizer implementation. The current research optimizer is `RationalMatrixPolicyOptimizer`, exposed through `rational_matrix_policy_onpolicy`.

## Thesis

RLB should not be optimized as an ordinary FFN with unusual coefficients. It exposes roles and symmetries that generic AdamW and Muon do not explicitly respect:

| RLB structure | optimizer implication |
| --- | --- |
| `W_in` | Controls which normalized domains the rational groups see. |
| rational coefficients | Move the shape of each local rational function. |
| `W_out` | Controls how rational features are used by the residual stream. |
| group activity | Shows whether a rational group is used, dead, or saturated. |
| derivative pressure | Shows whether updates are landing in responsive regions. |
| positive gauge | Separates useful function change from scale drift. |

The optimizer hypothesis is that using these signals can produce better training curves than generic optimizers under the same base LR schedule.

## Current Method

`RLB MatrixPolicy-Muon` is a role-aware matrix optimizer:

```text
ordinary Transformer weights -> AdamW
rational coefficients        -> AdamW
RLB W_in matrices            -> MatrixPolicy AdamW with input-role scaling
RLB W_out matrices           -> MatrixPolicy AdamW with output-role scaling
early RLB matrix window      -> Muon blend on W_in/W_out only
group scale drift            -> positive-gauge rebalance
```

Layer depth is part of the policy because shallow and deep rational FFNs use features differently. The early Muon phase is restricted to rational matrices; it is not a whole-model optimizer swap. The base warmup/cosine schedule remains shared with the controls.

## Why The Gauge Matters

For a group `g`, the transform

```text
W_in[g]  <- a_g W_in[g]
W_out[g] <- W_out[g] / a_g
```

preserves the represented RLB function for `a_g > 0`, but changes matrix norms and optimizer conditioning. This gives a direct mechanism test:

```text
A rational-specific optimizer should degrade less under gauge stress than generic RLB+AdamW or RLB+Muon.
```

If MatrixPolicy fails this test, the current optimizer is not exploiting the clearest symmetry in the model class and should be redesigned before adding more benchmark tasks.

## Verified Result

| row | final loss | final PPL | readout |
| --- | ---: | ---: | --- |
| RLB MatrixPolicy-Muon | 3.476232 | 32.34 | best verified row |
| RLB Smooth-MatrixPolicy | 3.493210 | 32.89 | older smooth policy |
| SiLU/SwiGLU+AdamW beta2=0.999 | 3.549346 | 34.79 | strongest AdamW control |
| RLB+AdamW beta2=0.999 | 3.550018 | 34.81 | generic AdamW on RLB |
| RLB+AdamW | 3.617501 | 37.24 | untuned generic AdamW |
| SiLU/SwiGLU+AdamW | 3.621982 | 37.41 | original AdamW control |
| SiLU/SwiGLU+Muon | 3.644921 | 38.28 | generic Muon control |
| RLB+Muon | 3.657877 | 38.78 | generic Muon on RLB |

This verifies that the optimizer family can beat strong same-LR controls, but the margin is still below the desired paper-level gap.

## Mechanistic Tests

The next optimizer work should be evaluated by these measurements, not only by final PPL:

| metric | reason |
| --- | --- |
| train and validation AUC | Measures full-curve improvement instead of one checkpoint. |
| time to loss threshold | Tests whether rational speedup appears early and persists. |
| gauge-stress degradation | Tests sensitivity to equivalent reparameterization. |
| group input/output RMS | Shows whether groups are active and used. |
| derivative pressure | Distinguishes responsive groups from saturated groups. |
| denominator margin | Tracks rational stability risk. |
| `W_in`/`W_out` norm product | Measures gauge drift. |
| function probe delta | Measures useful function movement per parameter movement. |

These diagnostics are the path toward a stronger optimizer. More global scheduling without a geometry signal should not count as progress.

## V2 Direction

The next non-trivial optimizer should make decisions per layer, role, and group:

| policy input | possible action |
| --- | --- |
| low group activity | revive or raise matrix scale only for that group. |
| high saturation pressure | damp coefficient updates and rebalance the input domain. |
| denominator risk | shrink coefficient trust radius. |
| `W_in`/`W_out` norm imbalance | apply stronger gauge rebalance. |
| stable gradient agreement | allow larger role-specific matrix steps. |
| late-layer instability | reduce output-role movement without changing global LR. |

The acceptance bar is strict: a new optimizer should beat `SiLU/SwiGLU+AdamW`, `RLB+AdamW`, `SiLU/SwiGLU+Muon`, and `RLB+Muon` under the same base LR schedule, then survive LR ablations after the large same-LR gap exists.
