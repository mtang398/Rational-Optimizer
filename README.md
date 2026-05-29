# RationalOPT

RationalOPT studies whether rational feed-forward layers can gain a real training advantage when the optimizer uses the structure that those layers expose. The target is not a better global learning-rate schedule. The target is an on-policy optimizer for rational activations that beats strong `SiLU/SwiGLU` and generic RLB controls under the same model size, token budget, seed, base LR schedule, and evaluation cadence.

## Research Claim

Current verified claim:

```text
RLB MatrixPolicy-Muon gives a modest same-LR WikiText-103 win.
The desired 0.2-0.3 loss gap is not reached yet.
```

The serious controls are:

```text
SiLU/SwiGLU+AdamW
RLB+AdamW
SiLU/SwiGLU+Muon
RLB+Muon
RLB MatrixPolicy variants
```

Jacobian, quotient, transport, and coefficient-only variants are ablations. They are not the baseline we are trying to beat. The benchmark target is the strongest generic optimizer on the strongest non-rational and rational models.

## RLB Geometry

RLB is a no-GLU rational FFN. For each rational group:

```text
v = x W_in
s_g = sqrt(mean(v_g^2) + eps)
u_g = v_g / s_g
h_g = s_g R_g(u_g)
y = h W_out
```

This creates optimizer-visible roles:

| object | role |
| --- | --- |
| `W_in` | chooses the input domain seen by each rational group. |
| rational basis | learns the nonlinear shape inside that normalized domain. |
| `W_out` | recombines rational features into the residual stream. |
| group scale | creates a positive gauge symmetry between `W_in` and `W_out`. |

The positive gauge is the key symmetry. Scaling one group's `W_in` rows by `a > 0` scales that group's RLB features by `a`; scaling the matching `W_out` columns by `1/a` preserves the represented function. Generic AdamW and Muon can still behave differently after this reparameterization because the matrices have different conditioning. A rational optimizer should be less sensitive to this gauge.

## Current Optimizer

`RLB MatrixPolicy-Muon` is the best verified optimizer family in this repo. It is not a global LR change.

```text
ordinary Transformer weights: AdamW
rational coefficients:        AdamW
RLB W_in / W_out matrices:    role- and depth-aware MatrixPolicy AdamW
early RLB matrix phase:       Muon blended only for W_in / W_out
after early phase:            return RLB matrices to MatrixPolicy AdamW
RLB groups:                   exact positive-gauge rebalance
```

The method uses the fact that `W_in`, rational coefficients, and `W_out` do different jobs. It also treats layer depth as a policy input because early and late layers do not use rational features in the same way. Exact scalar settings live in the Slurm launchers and JSONL `config` records; they are implementation details, not the research claim.

## Verified Evidence

The verified WikiText-103 result is a real same-LR lead, but it is still smaller than the final target.

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

Best verified gap versus the strongest `SiLU/SwiGLU+AdamW` row is `0.0731` loss and `2.45` PPL. That is promising but not enough for the intended paper claim.

## Figures

WikiText-103 validation loss:

![WikiText validation loss](experiments/results/rlb_matrix_policy_muon_switch_2026_05_28/same_lr_validation_loss.png)

WikiText-103 validation PPL:

![WikiText validation PPL](experiments/results/rlb_matrix_policy_muon_switch_2026_05_28/same_lr_validation_ppl.png)

WikiText-103 training loss from step 1:

![WikiText training loss from step 1](experiments/results/rlb_matrix_policy_muon_switch_2026_05_28/same_lr_training_loss_from_step1.png)

The sparse synthetic run is useful mainly as a curve-speed signal. Final bars are secondary because these tasks approach the loss floor.

Synthetic Code validation:

![Synthetic Code validation loss](experiments/results/synthetic_fair_full_2026_05_29/synthetic_code_validation_loss.png)

![Synthetic Code validation PPL](experiments/results/synthetic_fair_full_2026_05_29/synthetic_code_validation_ppl.png)

Synthetic Symbolic validation:

![Synthetic Symbolic validation loss](experiments/results/synthetic_fair_full_2026_05_29/synthetic_symbolic_validation_loss.png)

![Synthetic Symbolic validation PPL](experiments/results/synthetic_fair_full_2026_05_29/synthetic_symbolic_validation_ppl.png)

Reasoning mix validation:

![Reasoning mix validation loss](experiments/results/synthetic_fair_full_2026_05_29/synthetic_reasoning_mix_validation_loss.png)

![Reasoning mix validation PPL](experiments/results/synthetic_fair_full_2026_05_29/synthetic_reasoning_mix_validation_ppl.png)

Secondary final-state plots:

![Synthetic fair final loss](experiments/results/synthetic_fair_full_2026_05_29/final_loss_by_task.png)

![Synthetic fair final PPL](experiments/results/synthetic_fair_full_2026_05_29/final_ppl_by_task.png)

## Synthetic Readout

The completed sparse synthetic tasks show faster rational drops at sampled checkpoints, but they are sampled too sparsely and saturate too quickly for a final optimizer claim.

| task | curve signal | provisional early comparison |
| --- | --- | --- |
| Code | RLB drops faster, then the task saturates. | step 250 MatrixPolicy group-stat `0.1661` loss / `1.1807` PPL vs `SiLU/SwiGLU+AdamW` `0.4895` / `1.6314`. |
| Symbolic | MatrixPolicy is fastest early, but all rows nearly solve the task. | step 250 MatrixPolicy `0.0487` / `1.0499` vs `0.0609` / `1.0628`. |
| Reasoning mix | MatrixPolicy and group-stat lead the early/mid curve. | step 250 MatrixPolicy `0.3450` / `1.4120` vs `0.4127` / `1.5109`. |

The dense rerun exists to make this curve claim measurable: training every 10 steps, validation every 25 steps, starting at step 1. Until that run is summarized, the synthetic claim should be written as provisional.

## Validation Plan

The next research plan is mechanism-first:

1. Dense train/validation curves on the existing synthetic tasks, because the current sparse sampling misses early slope and crossover behavior.
2. Gauge-stress benchmark, because it directly tests whether the optimizer handles a real RLB symmetry better than generic AdamW and Muon.
3. Function-space diagnostics, because loss curves alone cannot tell whether updates change useful functions or mostly move along gauge directions.
4. Hard non-saturated synthetic tasks, because final loss below `0.1` compresses PPL and makes optimizer wins hard to interpret.
5. Real LM transfer at small scale, because synthetic-only evidence is not enough for a paper-level claim.

Falsification criteria:

```text
If MatrixPolicy is not more gauge-stable than generic RLB optimizers, the current optimizer is not exploiting the most obvious RLB geometry.
If dense curves remove the early rational speed signal, the synthetic result was a sampling artifact.
If hard non-saturated tasks do not preserve the early advantage into final loss, the optimizer is incomplete.
```

The current harsh self-review and task queue are in [TODO.md](TODO.md).

## Artifacts

```text
activation/         RLB activation implementation
training/           LLM harness, synthetic generators, gauge flags
optimizer_design/   RLB-specific optimizer implementation
experiments/        Slurm launchers and committed result figures
```

Raw JSONL runs and Slurm logs stay local under `experiments/runs/`. Committed figures and summaries live under `experiments/results/`.
