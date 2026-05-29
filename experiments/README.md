# Experiments

This folder should tell the result story cleanly. Raw Slurm logs and JSONL runs stay under `experiments/runs/` and are not committed. Commit compact summaries, plots, scripts, and README updates.

## Current Status

```text
WikiText-103:      MatrixPolicy-Muon verified as best current row
synthetic/code:    complete, negative for MatrixPolicy final loss
synthetic/symbolic: complete, tiny favorable signals for some RLB rows
reasoning_mix:     rerunning from scratch as job 951127, Requeue=1
```

The original fair synthetic job `937608` completed Code and Symbolic, then was preempted during Reasoning mix. The continuation job `951127` reruns Reasoning mix under fresh run names and archives partial row directories before rerun.

## Main Result

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

The current verified result is a modest same-LR win, not a final large-gap win.

## Synthetic Fair Readout So Far

| task | best finished row so far | result | interpretation |
| --- | --- | --- | --- |
| Code | SiLU/SwiGLU+AdamW | 0.088975 loss, 1.0931 PPL | RLB and MatrixPolicy lose final loss on this task. |
| Symbolic | SiLU/SwiGLU+Muon | 0.038782 loss, 1.0395 PPL | RLB generic optimizers and group-stat are close, but gains are tiny. |
| Reasoning mix | pending rerun | job 951127 | No claim until all six rows finish from scratch. |

Completed rows from the fair synthetic run:

| task | method | loss | PPL | delta loss vs SiLU/SwiGLU+AdamW |
| --- | --- | ---: | ---: | ---: |
| Code | SiLU/SwiGLU+AdamW | 0.088975 | 1.0931 | +0.000000 |
| Code | RLB+AdamW | 0.089657 | 1.0938 | +0.000682 |
| Code | SiLU/SwiGLU+Muon | 0.092114 | 1.0965 | +0.003139 |
| Code | RLB+Muon | 0.092613 | 1.0970 | +0.003638 |
| Code | RLB MatrixPolicy | 0.097335 | 1.1022 | +0.008359 |
| Code | RLB MatrixPolicy group-stat | 0.098191 | 1.1032 | +0.009216 |
| Symbolic | SiLU/SwiGLU+AdamW | 0.040289 | 1.0411 | +0.000000 |
| Symbolic | RLB+AdamW | 0.039067 | 1.0398 | -0.001222 |
| Symbolic | SiLU/SwiGLU+Muon | 0.038782 | 1.0395 | -0.001507 |
| Symbolic | RLB+Muon | 0.038812 | 1.0396 | -0.001477 |
| Symbolic | RLB MatrixPolicy | 0.040503 | 1.0413 | +0.000214 |
| Symbolic | RLB MatrixPolicy group-stat | 0.039030 | 1.0398 | -0.001259 |

Interpretation: the current MatrixPolicy is not broadly winning on the finished synthetic tasks. That matters. The next design should explain why Code loses while Symbolic gets small gains.

## Active Job

```text
job:       951127
name:      synth-reason
script:    experiments/scripts/run_synthetic_fair_full_20260529.sh
run root:  experiments/runs/synthetic_fair_reasoning_mix_20260529/
requeue:   enabled
```

Summarize after completion:

```bash
.venv-cu128/bin/python experiments/scripts/summarize_synthetic_fair_full_20260529.py \
  --run-root experiments/runs/synthetic_fair_reasoning_mix_20260529
```

## Artifact Policy

Do not stack new results on stale text. Rewrite the result story around the current best evidence, and keep exact configs in scripts or JSONL `config` records rather than large README dumps.
