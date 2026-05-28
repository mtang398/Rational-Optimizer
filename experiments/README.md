# Experiments

Generated experiment artifacts live here. Raw Slurm JSONL runs are local artifacts. Compact result summaries and plots live under `experiments/results/` and are the files meant to be committed.

## Current Result Artifact

```text
experiments/results/rlb_smooth_matrix_policy_2026_05_28/
```

Important files:

```text
summary.md
summary.csv
summary.json
same_lr_validation_loss.png
same_lr_validation_ppl.png
same_lr_training_loss_from_step1.png
```

## Main Same-LR Result

| row | final loss | final PPL |
| --- | ---: | ---: |
| RLB Smooth-MatrixPolicy | 3.493210 | 32.89 |
| SiLU/SwiGLU + AdamW beta2=0.999 | 3.549346 | 34.79 |
| RLB + AdamW beta2=0.999 | 3.550018 | 34.81 |
| RLB + AdamW | 3.617501 | 37.24 |
| SiLU/SwiGLU + AdamW | 3.621982 | 37.41 |

## Artifact Policy

Keep compact summaries and plots under:

```text
experiments/results/
```

Do not commit raw probe directories under:

```text
experiments/runs/wikitext103/
```
