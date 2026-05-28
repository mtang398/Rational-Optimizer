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
probe_summary.csv
synthetic_arithmetic_summary.csv
same_lr_validation_loss.png
same_lr_validation_ppl.png
same_lr_training_loss_from_step1.png
optimizer_probe_validation_loss.png
optimizer_probe_validation_ppl.png
synthetic_arithmetic_validation_loss.png
synthetic_arithmetic_validation_ppl.png
synthetic_arithmetic_training_loss_from_step1.png
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
| SiLU+Muon | 3.644921 | 38.28 |
| RLB+Muon | 3.657877 | 38.78 |

## Additional Tests

Muon controls:

```text
SiLU+Muon  final loss 3.644921, PPL 38.28
RLB+Muon   final loss 3.657877, PPL 38.78
```

Synthetic arithmetic final rows:

| row | final loss | final PPL |
| --- | ---: | ---: |
| SiLU+AdamW | 0.048182 | 1.04936 |
| RLB+AdamW | 0.048326 | 1.04951 |
| RLB MatrixPolicy-Muon | 0.048382 | 1.04957 |

A6000 optimizer probes:

| probe | last step | loss | readout |
| --- | ---: | ---: | --- |
| A6000 matched default | 1250 | 4.052293 | matched fallback screen |
| beta2 tail 0.995 | 1250 | 4.049556 | tiny +0.002738 vs matched default, not close to old best short curve |
| group policy 0.30 | 1000 | 4.141706 | neutral/worse vs matched default at 1000 |
| late Muon 0.05 | 500 | 4.673611 | worse than matched default at 500 |
| layer statgate | 250 | 5.369072 | tied with matched default |
| statgate+group 0.18 | 750 | 4.331103 | tiny +0.000628 vs matched default, noise-level |

## Interpretation

The current best is still `RLB MatrixPolicy-Muon`. Plain Muon is weaker than AdamW controls, and the new beta2-tail, group-policy, late-Muon, and stat-gated probes did not create a material gap. The synthetic task confirms early optimization transfer but not a final-loss win.

## Artifact Policy

Keep compact summaries and plots under `experiments/results/`. Do not commit raw probe directories or Slurm logs under `experiments/runs/`.
