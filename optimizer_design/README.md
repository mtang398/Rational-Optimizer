# Optimizer Design

This folder contains optimizer components for rational activations. The important file right now is `matrix_policy_optimizer.py`, which implements `RationalMatrixPolicyOptimizer` for RLB `W_in` and `W_out` matrices.

## Why RLB Needs A Different Optimizer

A normal SiLU/SwiGLU FFN hides most of its useful structure inside a smooth pointwise nonlinearity and dense matrices. RLB exposes more structure:

```text
v = x W_in
s_g = sqrt(mean(v_g^2) + eps)
u_g = v_g / s_g
h_g = s_g R_g(u_g)
y = h W_out
```

That gives the optimizer three handles:

| handle | meaning |
| --- | --- |
| rational domain | `W_in` decides where each group samples the rational basis. |
| feature recombination | `W_out` decides how rational features return to the model stream. |
| positive gauge | `W_in,g <- c W_in,g`, `W_out,g <- W_out,g / c` mostly preserves function while changing conditioning. |

MatrixPolicy exists because these handles are not present in the same way for SiLU/SwiGLU.

## Current MatrixPolicy

1. RLB creates explicit rational feature groups instead of a GLU gate.
2. `W_in` controls which input range each rational basis sees; `W_out` controls how those basis features are recombined.
3. Those two matrices have different jobs, so the optimizer should not treat them like ordinary dense FFN matrices.
4. MatrixPolicy uses a short early Muon phase only on RLB matrices, then returns those matrices to role/depth-aware AdamW.
5. Exact gauge balance keeps per-group basis scale from drifting while preserving the represented function as much as possible.

```text
activation:      rlb_fused_fixed_strong_ffn
optimizer:       rational_matrix_policy_onpolicy
base schedule:   same 3e-4 -> 3e-5 warmup/cosine schedule as controls
backbone:        AdamW
RLB matrices:    MatrixPolicy AdamW plus early matrix-only Muon
coefficients:    AdamW by default
gauge balance:   enabled
```

Exact flags live in `experiments/scripts/run_synthetic_fair_full_20260529.sh` and each run's JSONL `config` event.

The important point is not the exact scalar values. The important point is the structure of the update: matrix-local, role-aware, depth-aware, early-switching, and gauge-balanced.

## Why The Early Switch Exists

Early training benefits from a more geometry-aware matrix step. Muon helps there. Late training needs stable adaptive moments and should not keep the Muon pressure on. The current policy therefore uses Muon only as an early RLB-matrix component and then lets MatrixPolicy AdamW dominate.

This is different from testing `RLB+Muon`, which applies generic Muon as the optimizer. That generic control was worse on WikiText.

## Results That Matter

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

The verified gain is real but modest. A method that only improves by `0.0731` loss is not enough for the final research goal, but it shows that structure-aware optimization is not a dead end.

## Design Lessons

| helped | reason |
| --- | --- |
| short early Muon on RLB matrices | improves early matrix conditioning without harming late stability as much as full Muon. |
| different treatment for `W_in` and `W_out` | matches their different roles in rational domain selection and feature composition. |
| depth-dependent matrix pressure | later layers appear to prefer different matrix update pressure than early layers. |
| gauge balance | keeps group scales from absorbing optimizer effort. |

| did not help enough | readout |
| --- | --- |
| late Muon tails | worse probes. |
| role beta2 asymmetry | no material gain. |
| coefficient function-space updates | worse early signal. |
| larger or smaller role-depth strength | worse than current default. |
| global LR schedule tweaks | outside the claim unless the same change is applied to controls. |

## Open Direction

The next optimizer improvement should use live RLB information more selectively: group activity, derivative pressure, numerator/denominator risk, and layer role. It should first beat the same-LR controls on final loss, then be tested across LR only after the same-LR gap is large enough.
