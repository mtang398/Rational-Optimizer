# Read First

This repository is an optimizer research artifact. Read it as a method plus evidence boundary, not as a run diary.

## Core Question

Can a no-GLU rational FFN outperform SiLU/SwiGLU because its optimizer uses rational structure?

A comparison is valid only under the same base protocol: model size, token budget, seed, batch shape, sequence length, global LR schedule, weight decay, dataset slice, and evaluation cadence.

## Current Claim Boundary

The main supported claim is now from real-corpus language modeling:

```text
FineWeb:     RLB+MatrixPolicy (group-stat) beats SiLU+AdamW by 0.160467 loss / 13.40 PPL.
FineWeb-Edu: RLB+MatrixPolicy (group-stat) beats SiLU+AdamW by 0.152964 loss / 9.70 PPL.
WikiText-103 remains a useful anchor with a smaller 0.073114 loss / 2.45 PPL gap.
```

The FineWeb-Edu result is especially important because plain `RLB+AdamW` becomes nonfinite early, while MatrixPolicy completes and gives the best heldout loss. This means the current result should be read as an optimizer result, not as an RLB-only activation comparison.

The earlier saturated synthetic result packages were removed from the tracked public evidence. They were useful while debugging curve speed, but they are not strong enough for the current research story because the tasks reach a compressed loss floor. Real-corpus PPL plots omit divergent/nonfinite rows and include step-1000 zoomed versions so completed optimizers remain readable.

## Control Set

Do not use Jacobian, quotient, transport, coefficient-only, or scheduler ablations as the baseline. The real controls are:

```text
SiLU/SwiGLU + AdamW
RLB + AdamW
SiLU/SwiGLU + Muon
RLB + Muon
RLB + rational_matrix_policy_onpolicy
```

A rational optimizer result is meaningful only if the base LR schedule is shared across these rows.

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

for `a_g > 0`. The represented function is preserved, but the optimizer sees different matrix norms and conditioning.

MatrixPolicy is the RLB matrix optimizer. It updates `W_in` and `W_out` with role-aware, depth-aware, time-aware matrix rules, applies mild group-stat preconditioning, and then rebalances the exact positive gauge. Ordinary Transformer weights stay on AdamW in the current real-corpus run.

## What To Read

1. [README.md](README.md) for the current results, figures, and claim boundary.
2. [optimizer_design/README.md](optimizer_design/README.md) for the mathematical optimizer definition.
3. [experiments/README.md](experiments/README.md) for result packages and regeneration commands.
4. [training/README.md](training/README.md) for the fair-comparison and logging contract.
5. [activation/README.md](activation/README.md) for the RLB layer definition.
6. [paper/iclr_method_draft/README.md](paper/iclr_method_draft/README.md) for the method-only ICLR draft and build command.

## Evidence Standard

A result should be treated as paper-relevant only if it has:

```text
same base LR schedule across controls
step-1 training and validation curves
strong AdamW and Muon controls for both SiLU/SwiGLU and RLB
non-saturated task or real LM loss scale
at least one real-corpus LM setting
mechanism diagnostics for gauge drift and function-space movement
```

The current real-corpus screen satisfies the first five points for one seed. The main missing pieces are multi-seed confirmation and direct mechanism diagnostics.
