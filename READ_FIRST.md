# Read First

This repository should be read as an optimizer research report, not as a chronological experiment log. The central question is whether rational FFNs expose structure that an optimizer can use to beat strong generic controls under the same base learning-rate schedule.

## What To Claim

Claim only this today:

```text
RLB MatrixPolicy-Muon has the best verified WikiText-103 row.
The verified lead is 0.0731 loss / 2.45 PPL over SiLU/SwiGLU+AdamW beta2=0.999.
The target 0.2-0.3 loss gap has not been reached.
```

Do not present Jacobian, quotient, or coefficient variants as baselines. The real controls are `SiLU/SwiGLU+AdamW`, `RLB+AdamW`, `SiLU/SwiGLU+Muon`, and `RLB+Muon`.

## Method In One Paragraph

RLB separates the FFN into `W_in`, grouped rational functions, and `W_out`. `W_in` chooses the rational input domains, the coefficients shape the nonlinear functions, and `W_out` recombines the features. Because each group has a positive scale gauge between `W_in` and `W_out`, two parameterizations can represent the same function while giving generic optimizers different conditioning. MatrixPolicy uses role, depth, and group structure to update rational matrices differently from ordinary Transformer weights without changing the shared global LR schedule.

## Evidence

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

The sparse synthetic curves suggest faster rational loss/PPL drops, especially on Code and Reasoning mix, but those runs are too sparse and too close to the final loss floor for a final claim. Use them as motivation for dense curves and harder tasks, not as the paper result.

## Current Research Plan

The immediate plan is:

1. Summarize the dense synthetic rerun with training sampled every 10 steps and validation every 25 steps.
2. Evaluate gauge stress at log-scale `2.0` against gauge `0.0` for generic RLB optimizers and MatrixPolicy variants.
3. Add diagnostics for group activity, derivative pressure, denominator margin, gauge drift, and function-space movement.
4. Move to harder non-saturated tasks only after the gauge result says whether the current optimizer is using RLB geometry.

Pass/fail rule: MatrixPolicy must degrade less than `RLB+AdamW` and `RLB+Muon` under equivalent-function gauge stress. If it does not, the optimizer needs redesign before more task chasing.

## Operating Rule

Use only A6000 GPUs and keep total active allocation at or below 8 A6000s. Raw runs under `experiments/runs/` are local artifacts; commit compact figures, summaries, and code only.
