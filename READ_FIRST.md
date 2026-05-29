# Read First

This repository is an optimizer research artifact. Read it as a method plus evidence boundary, not as a run diary.

## Core Question

Can a no-GLU rational FFN outperform SiLU/SwiGLU because its optimizer uses rational structure?

The comparison is valid only under the same base protocol: model size, token budget, seed, batch shape, sequence length, global LR schedule, weight decay, dataset, and evaluation cadence.

## Claim Boundary

Current supported claim:

```text
RLB MatrixPolicy-Muon is the best verified WikiText-103 row in this repo.
It improves over SiLU/SwiGLU+AdamW beta2=0.999 by 0.0731 loss / 2.45 PPL.
This is not yet the desired 0.2-0.3 loss gap.
```

Do not use Jacobian, quotient, transport, or coefficient ablations as the baseline. The real control set is:

```text
SiLU/SwiGLU+AdamW
RLB+AdamW
SiLU/SwiGLU+Muon
RLB+Muon
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

for `a_g > 0`. The represented function is preserved, but the optimizer sees different matrix norms and conditioning. MatrixPolicy uses this structure by assigning separate update policies to `W_in`, rational coefficients, and `W_out`, and by applying a function-preserving gauge rebalance after optimizer steps.

The short definition is: MatrixPolicy is the RLB matrix optimizer. It updates only the RLB `W_in` and `W_out` matrices with role-aware, depth-aware, time-aware matrix rules. Ordinary Transformer weights remain on AdamW, rational coefficients use their coefficient optimizer, and the wrapper rebalances the positive gauge after the step without changing the represented function.

## What To Look At First

1. [optimizer_design/README.md](optimizer_design/README.md) for the mathematical optimizer definition.
2. [README.md](README.md) for the research claim, results table, figures, and falsification tests.
3. [experiments/README.md](experiments/README.md) for the experimental protocol and plotted artifacts.
4. [training/README.md](training/README.md) for the fair-comparison contract and logging standard.
5. [activation/README.md](activation/README.md) for the RLB layer definition.

## Evidence Standard

A result should be treated as paper-relevant only if it satisfies all of these:

```text
same base LR schedule across controls
step-1 training and validation curves
strong AdamW and Muon controls for both SiLU/SwiGLU and RLB
non-saturated task or real LM loss scale
mechanism test, especially positive gauge stress
```

Synthetic tasks that reach loss `<0.1` are useful for debugging curve speed, but not for claiming a large optimizer advantage.
