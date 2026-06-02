# Read First

This repository is an optimizer research artifact. Read it as a method plus evidence boundary, not as a run diary.

## Core Question

Can a no-GLU rational FFN outperform SiLU/SwiGLU because its optimizer uses rational structure?

A comparison is valid only under the same base protocol: model size, token budget, seed set, batch shape, sequence length, base LR schedule, weight decay, dataset slice, and evaluation cadence.

## Current Claim Boundary

The main supported claim is now from 3-seed real-corpus language modeling:

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

A result should be treated as paper-relevant only if it has:

```text
same base LR schedule across controls
step-1 training and validation curves
strong tuned AdamW, Muon, SOAP/Shampoo-style, Lion, AdEMAMix, Schedule-Free AdamW, and Adafactor/CAME controls where stable
multi-seed real-corpus LM evidence
non-saturated task or real LM loss scale
reported divergent/nonfinite rows
mechanism readouts for gauge drift, function-space movement, role-specific updates, denominator margins, and matrix spectra
```

The current real-corpus screen now satisfies the preliminary multi-seed evidence requirement for two corpora. Telemetry and broad-baseline optimizer code are implemented; the main missing pieces are CUDA/DDP validation of the telemetry path, Phase A HPO over tuned broad baselines, scale/budget tests, mechanism-intervention experiments, downstream checks, and only then method-component ablations.
