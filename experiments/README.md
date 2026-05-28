# Experiments

Generated experiment artifacts live here. Raw Slurm JSONL runs are local artifacts. The committed result story is the compact current artifact below.

## Current Result Artifact

```text
experiments/results/rlb_matrix_policy_muon_switch_2026_05_28/
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
| RLB MatrixPolicy-Muon | 3.476232 | 32.34 |
| RLB Smooth-MatrixPolicy | 3.493210 | 32.89 |
| SiLU+AdamW beta2=0.999 | 3.549346 | 34.79 |
| RLB+AdamW beta2=0.999 | 3.550018 | 34.81 |
| RLB+AdamW | 3.617501 | 37.24 |
| SiLU+AdamW | 3.621982 | 37.41 |

## Artifact Policy

Keep compact summaries and plots under `experiments/results/`. Do not commit raw probe directories under `experiments/runs/wikitext103/`.
