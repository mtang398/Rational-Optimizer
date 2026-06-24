# Read First

This repository is an optimizer research artifact. Read it as a method plus result package, not as a run diary.

## Core Question

Can a single-branch rational Transformer MLP sublayer outperform the standard SiLU/SwiGLU MLP sublayer because its optimizer uses rational structure?

Every comparison must keep the base protocol matched: model size, token budget, seed set, batch shape, sequence length, base LR schedule, weight decay, dataset slice, and evaluation cadence.

## Current Result Boundary

The paper-facing result boundary is complete for E1 M0/100M and E2 M0/300M across all five matched datasets: DCLM, FineWeb-Edu, FineWeb, Dolma-sample, and C4. E1 has three seeds per dataset with matched outer configs, dense validation every 50 steps, and full mean +/- sample std curves in `experiments/results/iclr26_e1_figures/`. E2 has three seeds per dataset, final eval at step `9150`, and full result packages under `experiments/results/iclr26_e2_*` plus curves in `experiments/results/iclr26_e2_figures/`.

All MatrixPolicy values below use the accepted safe-speed replacement rows: `experiments/manifests/iclr26_matrixpolicy_safe_speed_e1_manifest.csv` for E1 and `experiments/manifests/iclr26_matrixpolicy_safe_speed_e2_manifest.csv` for E2. Token-to-target savings are tracked in the E1 package and in each E2 dataset package.

E1 M0/100M final validation-loss anchor:

| Dataset | MatrixPolicy final val loss | next best current method | gap |
| --- | ---: | ---: | ---: |
| DCLM | 4.256989 +/- 0.004197 | rlb_lion 4.305728 +/- 0.005836 | 0.048739 |
| FineWeb-Edu | 4.088287 +/- 0.009169 | rlb_lion 4.142669 +/- 0.006812 | 0.054382 |
| FineWeb | 4.319472 +/- 0.012370 | rlb_lion 4.367062 +/- 0.007532 | 0.047590 |
| Dolma-sample | 4.323933 +/- 0.005168 | rlb_lion 4.369254 +/- 0.005561 | 0.045321 |
| C4 | 4.286446 +/- 0.019324 | rlb_lion 4.335663 +/- 0.020917 | 0.049217 |

E2 M0/300M final validation-loss anchor:

| Dataset | MatrixPolicy final val loss | next best aggregate method | gap |
| --- | ---: | ---: | ---: |
| DCLM | 3.956069 +/- 0.030752 | silu_lion 3.993430 +/- 0.023038 | 0.037361 |
| FineWeb-Edu | 3.707768 +/- 0.018711 | rlb_muon 3.738164 +/- 0.021014 | 0.030396 |
| FineWeb | 3.964892 +/- 0.009459 | rlb_muon 4.001245 +/- 0.011375 | 0.036353 |
| Dolma-sample | 3.808954 +/- 0.006442 | rlb_lion 3.842503 +/- 0.009333 | 0.033549 |
| C4 | 3.883021 +/- 0.014134 | rlb_muon 3.915858 +/- 0.016066 | 0.032837 |

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
