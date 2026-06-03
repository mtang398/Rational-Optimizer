# Read First

This repository is an optimizer research artifact. Read it as a method plus evidence boundary, not as a run diary.

## Core Question

Can a no-GLU rational FFN outperform SiLU/SwiGLU because its optimizer uses rational structure?

A comparison is valid only under the same base protocol: model size, token budget, seed set, batch shape, sequence length, base LR schedule, weight decay, dataset slice, and evaluation cadence.

## Current Pilot Claim Boundary

The current pilot signal is from 3-seed real-corpus language modeling:

```text
FineWeb:     RLB+MatrixPolicy (group-stat) beats SiLU+AdamW by 0.159263 mean loss.
FineWeb-Edu: RLB+MatrixPolicy (group-stat) beats SiLU+AdamW by 0.154149 mean loss.
```

It also beats the best non-MatrixPolicy control by about 0.15 mean validation loss on both datasets. The FineWeb-Edu result is important because plain `RLB+AdamW` has one divergent seed while MatrixPolicy completes all seeds.

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

A rational optimizer result is meaningful only if the base LR schedule is shared across these rows.

The training harness now includes broad modern optimizer-family baselines, but they still need tuned paper-result runs before final claims:

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

MatrixPolicy updates `W_in` and `W_out` with role-aware, depth-aware, time-aware matrix rules, applies mild group-stat preconditioning, and then rebalances the exact positive gauge. Ordinary Transformer weights stay on AdamW in the current real-corpus run.

## What To Read

1. [README.md](README.md) for the current results and claim boundary.
2. [experiments/ICLR_OPTIMIZER_EXPERIMENT_BLUEPRINT.md](experiments/ICLR_OPTIMIZER_EXPERIMENT_BLUEPRINT.md) for the full paper experiment program.
3. [experiments/README.md](experiments/README.md) for result packages and regeneration commands.
4. [optimizer_design/README.md](optimizer_design/README.md) for the mathematical optimizer definition.
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

The current FineWeb/FineWeb-Edu result is a pilot signal. A paper-relevant claim requires the accepted optimizer-paper evidence stack:

```text
Dense curves are mandatory: eval interval <= 50 for paper/protocol runs, not every 200 steps
seriously tuned AdamW and accepted optimizer-family baselines
final-budget comparisons, not intermediate-checkpoint wins
speed-to-target in tokens, steps, GPU-hours, and wall-clock time
optimizer overhead, throughput, memory, clipping, and divergence accounting
model-scale and token-budget variation at academic scale
held-out corpus/task transfer without retuning
mechanism tests tied to RLB gauge/rational geometry
method ablations only after the main result is established
```

The paper plan is in `experiments/ICLR_OPTIMIZER_EXPERIMENT_BLUEPRINT.md`. It should be read before launching new jobs. The ordering is deliberate: decisive tuned benchmark first, mechanism and scale next, ablations last.
