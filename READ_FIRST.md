# Read First

This repository is an optimizer research artifact. Read it as a method plus result package, not as a run diary.

## Core Question

Can a single-branch rational Transformer MLP sublayer outperform the standard SiLU/SwiGLU MLP sublayer because its optimizer uses rational structure?

Every comparison must keep the base protocol matched: model size, token budget, seed set, batch shape, sequence length, base LR schedule, weight decay, dataset slice, and evaluation cadence.

## Current Result Boundary

The paper-facing result boundary is complete for E1 M0/100M and E2 M0/300M across all five matched datasets: DCLM, FineWeb-Edu, FineWeb, Dolma-sample, and C4. E1 has three seeds per dataset with matched outer configs, dense validation every 50 steps, and full mean +/- sample std curves in `experiments/results/iclr26_e1_figures/`. E2 has three seeds per dataset, final eval at step `9150`, and full result packages under `experiments/results/iclr26_e2_*` plus curves in `experiments/results/iclr26_e2_figures/`.

MatrixPolicy values below use the 30 validated live-statistic-corrected `rlb_fused_global_rational` rows in `experiments/corrections/matrixpolicy_live_stats_20260712/`; non-MatrixPolicy RLB controls use `experiments/manifests/iclr26_global_rational_optimizer_controls_manifest.csv`. Token-to-target savings are tracked in the E1 package and in each E2 dataset package; RLB+ADeMaMix is retained as a divergent/early-stop negative row.

E1 M0/100M final validation-loss anchor:

| Dataset | MatrixPolicy final val loss | next best current method | gap |
| --- | ---: | ---: | ---: |
| DCLM | 4.252404 +/- 0.005837 | rlb_lion 4.294575 +/- 0.008320 | 0.042171 |
| FineWeb-Edu | 4.085588 +/- 0.009984 | rlb_lion 4.136091 +/- 0.008299 | 0.050503 |
| FineWeb | 4.317483 +/- 0.013287 | rlb_lion 4.362572 +/- 0.011154 | 0.045089 |
| Dolma-sample | 4.323237 +/- 0.003625 | rlb_lion 4.362160 +/- 0.006582 | 0.038923 |
| C4 | 4.286474 +/- 0.020047 | rlb_lion 4.327134 +/- 0.015977 | 0.040660 |

E2 M0/300M final validation-loss anchor:

| Dataset | MatrixPolicy final val loss | next best aggregate method | gap |
| --- | ---: | ---: | ---: |
| DCLM | 3.951745 +/- 0.029409 | rlb_lion 3.988719 +/- 0.029477 | 0.036973 |
| FineWeb-Edu | 3.702151 +/- 0.020357 | rlb_muon 3.737328 +/- 0.018698 | 0.035177 |
| FineWeb | 3.962942 +/- 0.008025 | rlb_lion 3.996049 +/- 0.010524 | 0.033107 |
| Dolma-sample | 3.805292 +/- 0.007943 | rlb_lion 3.841206 +/- 0.008478 | 0.035914 |
| C4 | 3.878322 +/- 0.015039 | rlb_lion 3.913219 +/- 0.013928 | 0.034897 |

This should be read as an optimizer result, not an RLB-only activation comparison. WikiText remains only a small demo anchor.

## Control Set

The completed E1 control set is fixed by `experiments/manifests/iclr26_main_manifest.csv`:

```text
SiLU + AdamW
RLB + AdamW
SiLU + Muon
RLB + Muon
SiLU + Lion
RLB + Lion
SiLU + SOAP/Shampoo-style AdamW
RLB + SOAP/Shampoo-style AdamW
SiLU + AdEMAMix
RLB + AdEMAMix
SiLU + Schedule-Free AdamW-style
RLB + Schedule-Free AdamW-style
SiLU + Adafactor/CAME-style
RLB + Adafactor/CAME-style
RLB + rational_matrix_policy_onpolicy
```

A rational optimizer result is meaningful only if the outer optimizer config is shared across rows. For paper runs, the manifest enforces the AdamW/MatrixPolicy `lr`, `min_lr`, and `weight_decay` parity rule.

## Method Summary

RLB maps each Transformer MLP hidden group through a normalized rational function:

```text
z_g = group_g(W_in x)
r_g = sqrt(mean(z_g^2) + eps)
h_g = r_g R_g(z_g / r_g)
y = W_out concat_g(h_g)
```

In the homogeneous radius, this creates a positive group gauge:

```text
W_in[g]  <- a_g W_in[g]
W_out[g] <- W_out[g] / a_g
```

for `a_g > 0`. The represented function is preserved in that setting; with the stabilizing RMS floor this remains the local scale structure used by the optimizer. Generic optimizers see different matrix norms and conditioning.

MatrixPolicy updates `W_in` and `W_out` with role-aware, depth-aware, time-aware matrix rules, applies group-stat preconditioning in the current recipe, and then rebalances the positive gauge. Ordinary Transformer weights stay on AdamW in the current recipe.

## What To Read

1. [README.md](README.md) for current results, curves, and the main rule.
2. [experiments/ICLR_EXACT_RUN_PLAN.md](experiments/ICLR_EXACT_RUN_PLAN.md) for the full experiment program and manifest workflow.
3. [experiments/README.md](experiments/README.md) for result packages and launch commands.
4. [optimizer_design/README.md](optimizer_design/README.md) for the optimizer definition.
5. [training/README.md](training/README.md) for the fair-comparison and logging contract.
6. [activation/README.md](activation/README.md) for the RLB layer definition.
7. [paper/iclr_method_draft/README.md](paper/iclr_method_draft/README.md) for the paper draft status.

## Resource Rules

```text
max 4 A6000 GPUs per task/job
max 8 A6000 GPUs active at the same time
repo size below 200G
```

## Evidence Standard

New paper runs follow `experiments/manifests/iclr26_main_manifest.csv` and `experiments/scripts/run_iclr26_manifest_job.sh`.

```text
dense curves: eval interval <= 50 for paper/protocol runs
fixed main comparisons before sensitivity maps
same outer AdamW/MatrixPolicy config inside each matched cell
final-budget comparisons at 100M and 300M tokens
speed-to-target in tokens, steps, GPU-hours, and wall-clock time
optimizer overhead, throughput, memory, clipping, and divergence accounting
model-scale and token-budget variation at academic scale
held-out corpus/task transfer
mechanism tests tied to RLB gauge/rational geometry
method ablations last
```

The exact order is in `experiments/ICLR_EXACT_RUN_PLAN.md`.
