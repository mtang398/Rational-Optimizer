# Read First

This repository is an optimizer research artifact. Read it as a method plus result package, not as a run diary.

## Core Question

Can a no-GLU rational FFN outperform SiLU/SwiGLU because its optimizer uses rational structure?

Every comparison must keep the base protocol matched: model size, token budget, seed set, batch shape, sequence length, base LR schedule, weight decay, dataset slice, and evaluation cadence.

## Current Result Boundary

The current FineWeb/FineWeb-Edu signal is from 3-seed real-corpus language modeling:

```text
FineWeb:     RLB+MatrixPolicy (group-stat) beats SiLU+AdamW by 0.159263 mean loss.
FineWeb-Edu: RLB+MatrixPolicy (group-stat) beats SiLU+AdamW by 0.154149 mean loss.
```

It also beats the best non-MatrixPolicy control by about 0.15 mean validation loss on both datasets. FineWeb-Edu is important because plain `RLB+AdamW` has one divergent seed while MatrixPolicy completes all seeds.

This should be read as an optimizer result, not an RLB-only activation comparison.

## Control Set

The current completed control set is:

```text
SiLU/SwiGLU + AdamW
RLB + AdamW
SiLU/SwiGLU + Muon
RLB + Muon
RLB + rational_matrix_policy_onpolicy
```

A rational optimizer result is meaningful only if the outer optimizer config is shared across rows. For new paper runs, the manifest enforces the AdamW/MatrixPolicy `lr`, `min_lr`, and `weight_decay` parity rule.

The training harness includes broad modern optimizer-family baselines:

```text
SOAP/Shampoo-style AdamW
Lion
AdEMAMix
Schedule-Free AdamW-style
Adafactor/CAME-style
```

## Method Summary

RLB maps each FFN hidden group through a normalized rational function:

```text
z_g = group_g(W_in x)
r_g = sqrt(mean(z_g^2) + eps)
h_g = r_g R_g(z_g / r_g)
y = W_out concat_g(h_g)
```

This creates a positive group gauge:

```text
W_in[g]  <- a_g W_in[g]
W_out[g] <- W_out[g] / a_g
```

for `a_g > 0`. The represented function is preserved, but generic optimizers see different matrix norms and conditioning.

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
