# Read First

This repository is an optimizer research artifact. Read it as a method plus result package, not as a run diary.

## Core Question

Can a no-GLU rational FFN outperform SiLU/SwiGLU because its optimizer uses rational structure?

Every comparison must keep the base protocol matched: model size, token budget, seed set, batch shape, sequence length, base LR schedule, weight decay, dataset slice, and evaluation cadence.

## Current Result Boundary

The paper-facing result boundary now has two completed pieces: E1 M0/100M across all five matched datasets, and E2 M0/300M for the full DCLM cell. E1 has three seeds per dataset with matched outer configs, dense validation every 50 steps, and full mean +/- sample std curves in `experiments/ICLR_RUN_STATUS.md`. E2 DCLM rows `240-284` are complete and summarized in `experiments/results/iclr26_e2_dclm_2026_06_10/`.

| Dataset | MatrixPolicy final val loss | next best current method | gap |
| --- | ---: | ---: | ---: |
| DCLM | 4.256224 +/- 0.004972 | rlb_lion 4.305728 +/- 0.005836 | 0.049505 |
| FineWeb-Edu | 4.088240 +/- 0.009434 | rlb_lion 4.142669 +/- 0.006812 | 0.054429 |
| FineWeb | 4.318581 +/- 0.010914 | rlb_lion 4.367062 +/- 0.007532 | 0.048481 |
| Dolma-sample | 4.323851 +/- 0.004565 | rlb_lion 4.369254 +/- 0.005561 | 0.045403 |
| C4 | 4.285119 +/- 0.020677 | rlb_lion 4.335663 +/- 0.020917 | 0.050544 |

E2 DCLM M0/300M summary: MatrixPolicy final validation loss is `3.957627 +/- 0.030713`; the next-best aggregate method is `silu_lion` at `3.993430 +/- 0.023038`. MatrixPolicy is best on all three seeds, with the token-to-target savings below.

| Target loss | MP all-hit mean | Vs fastest non-MP: MP -> comparator (seeds) | Saved | Saved % | Vs SiLU+AdamW: MP -> AdamW (seeds) | Saved | Saved % |
| ---: | ---: | --- | ---: | ---: | --- | ---: | ---: |
| 4.40 | 74.3M | 74.3M -> 80.8M (3/3) | 6.6M | 8.1% | 74.3M -> 93.4M (3/3) | 19.1M | 20.5% |
| 4.30 | 101.0M | 101.0M -> 104.9M (3/3) | 3.8M | 3.6% | 101.0M -> 120.7M (3/3) | 19.7M | 16.3% |
| 4.20 | 133.3M | 133.3M -> 139.3M (3/3) | 6.0M | 4.3% | 133.3M -> 161.1M (3/3) | 27.9M | 17.3% |
| 4.10 | 176.4M | 176.4M -> 187.9M (3/3) | 11.5M | 6.1% | 176.4M -> 227.7M (3/3) | 51.3M | 22.5% |
| 4.05 | 205.3M | 205.3M -> 222.8M (3/3) | 17.5M | 7.8% | 185.1M -> 244.1M (1/3) | 59.0M | 24.2% |
| 4.00 | 244.7M | 232.7M -> 267.9M (2/3) | 35.2M | 13.1% | not reached (0/3) | not reached | n/a |

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

RLB maps each FFN hidden group through a normalized rational function:

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
