# Experiments

Generated experiment artifacts live here. Raw Slurm JSONL runs are local artifacts; compact result summaries and plots live under `experiments/results/` and are the files meant to be committed.

## Current Result Artifact

```text
experiments/results/rlb_matrix_policy_2026_05_28/
```

Important files:

```text
summary.md
summary.csv
same_lr_validation_loss.png
same_lr_validation_ppl.png
same_lr_training_loss_from_step1.png
matrix_policy_ablation_loss.png
matrix_policy_ablation_ppl.png
same_lr_1250_loss.png
same_lr_1250_ppl.png
```

The active best run is `RLB + rational_matrix_policy_onpolicy` with the same global LR schedule as the controls.

## Main Same-LR Result

| row | final loss | final PPL |
| --- | ---: | ---: |
| RLB MatrixPolicy-Y | 3.548665 | 34.77 |
| RLB + Jacobian | 3.614862 | 37.15 |
| RLB + AdamW | 3.617501 | 37.24 |
| SiLU/SwiGLU + AdamW | 3.621982 | 37.41 |

## Artifact Policy

Keep:

```text
experiments/cache/                         local dataset/token cache
experiments/results/                       compact committed summaries/plots
experiments/runs/wikitext103/...           local raw JSONL while actively analyzing
```

Do not commit raw probe directories. `.gitignore` ignores the current local `rlb_*probe*` run folders.
