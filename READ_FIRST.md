# Read First

This repo is not trying to prove that a different LR schedule wins. It is trying to design an optimizer that uses the special structure of RLB and beats the real controls under the same base schedule.

## Claim Boundary

```text
verified current best: RLB MatrixPolicy-Muon on WikiText-103
verified gap:         +0.0731 loss / +2.45 PPL over SiLU/SwiGLU+AdamW beta2=0.999
requested target:     0.2-0.3 loss gap, not reached yet
synthetic status:     Code and Symbolic complete, Reasoning mix rerunning as job 951127
```

Do not present Jacobian, quotient, or transport optimizers as baselines. They are ablations. The baseline is `SiLU/SwiGLU+AdamW`, plus generic optimizer controls on both SiLU/SwiGLU and RLB.

## Method In One Paragraph

MatrixPolicy treats RLB as a structured layer, not just another dense FFN. RLB has an input matrix that chooses rational domains, an output matrix that recombines rational features, and a positive scale gauge between them. The optimizer uses an early matrix-only Muon window for fast conditioning, then returns to role/depth-aware AdamW while gauge balance controls scale drift.

## Results To Remember

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

Synthetic fair rerun so far:

| task | best finished row so far | result | interpretation |
| --- | --- | --- | --- |
| Code | SiLU/SwiGLU+AdamW | 0.088975 loss, 1.0931 PPL | RLB and MatrixPolicy lose final loss on this task. |
| Symbolic | SiLU/SwiGLU+Muon | 0.038782 loss, 1.0395 PPL | RLB generic optimizers and group-stat are close, but gains are tiny. |
| Reasoning mix | pending rerun | job 951127 | No claim until all six rows finish from scratch. |

The Code result is a warning: the current optimizer can learn fast early but still lose final loss. The Symbolic result is encouraging but tiny. Reasoning mix decides whether the pattern transfers across a more varied task.

## Running Jobs

Use A6000 only and keep total active allocation at or below 8 A6000s.

```bash
squeue -u mt872
```

Current continuation:

```text
job:      951127
name:     synth-reason
GPUs:     4x nvidia_rtx_a6000
requeue:  enabled
```

The launcher is restart-safe at row granularity: completed rows are skipped, incomplete run directories are archived before rerun, and the job asks Slurm to requeue on the pre-timeout signal.

## After Reasoning Mix Finishes

Run the summarizer, inspect the compact table and plots, then rewrite the docs around the actual result. Do not stack a new section on top of stale claims.

```bash
.venv-cu128/bin/python experiments/scripts/summarize_synthetic_fair_full_20260529.py \
  --run-root experiments/runs/synthetic_fair_reasoning_mix_20260529
```

Keep raw `experiments/runs/` logs out of git.
