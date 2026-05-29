# Experiments

This folder contains Slurm launchers and compact result artifacts. Raw Slurm logs and JSONL runs stay under `experiments/runs/` and are not committed.

## Graphs

Verified WikiText and arithmetic plots are already committed:

```text
experiments/results/rlb_matrix_policy_muon_switch_2026_05_28/
```

Key figures:

![WikiText validation loss](results/rlb_matrix_policy_muon_switch_2026_05_28/same_lr_validation_loss.png)

![WikiText validation PPL](results/rlb_matrix_policy_muon_switch_2026_05_28/same_lr_validation_ppl.png)

![WikiText training loss from step 1](results/rlb_matrix_policy_muon_switch_2026_05_28/same_lr_training_loss_from_step1.png)

![Synthetic Code validation loss](results/synthetic_fair_full_2026_05_29/synthetic_code_validation_loss.png)

![Synthetic Code validation PPL](results/synthetic_fair_full_2026_05_29/synthetic_code_validation_ppl.png)

![Synthetic Symbolic validation loss](results/synthetic_fair_full_2026_05_29/synthetic_symbolic_validation_loss.png)

![Synthetic Symbolic validation PPL](results/synthetic_fair_full_2026_05_29/synthetic_symbolic_validation_ppl.png)

![Reasoning mix validation loss](results/synthetic_fair_full_2026_05_29/synthetic_reasoning_mix_validation_loss.png)

![Reasoning mix validation PPL](results/synthetic_fair_full_2026_05_29/synthetic_reasoning_mix_validation_ppl.png)

Secondary final-state plots:

![Synthetic fair final loss](results/synthetic_fair_full_2026_05_29/final_loss_by_task.png)

![Synthetic fair final PPL](results/synthetic_fair_full_2026_05_29/final_ppl_by_task.png)

Synthetic Code/Symbolic/Reasoning mix plots are committed under `experiments/results/synthetic_fair_full_2026_05_29/`.

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

## Synthetic Fair Curve Result

| task | curve signal | strongest early gap vs `SiLU/SwiGLU+AdamW` | final readout |
| --- | --- | --- | --- |
| Code | RLB drops much faster, then the task saturates. | step 250 MatrixPolicy group-stat: `0.1661` loss / `1.1807` PPL vs `0.4895` / `1.6314`. | final SiLU+AdamW wins by a tiny floor-level margin. |
| Symbolic | MatrixPolicy is fastest early, but all rows are near solved. | step 250 MatrixPolicy: `0.0487` / `1.0499` vs `0.0609` / `1.0628`. | final differences are too small to claim broad superiority. |
| Reasoning mix | MatrixPolicy/group-stat lead the early and mid curve. | step 250 MatrixPolicy: `0.3450` / `1.4120` vs `0.4127` / `1.5109`. | final generic RLB+Muon is slightly best, but the curve signal is cleaner. |

The completed synthetic tasks are near saturation, so final-loss/PPL differences are not the main readout. The main synthetic evidence is that rational rows reach the low-loss regime faster. Full curve diagnostics are in `results/synthetic_fair_full_2026_05_29/curve_diagnostics.md`.

## Proposed Short Task Queue

These are not submitted yet. They are the next short tests that should replace saturated toy comparisons:

| priority | task | reason |
| ---: | --- | --- |
| 1 | `synthetic/rule_chain_hard` | direct test of rational piecewise composition. |
| 2 | `synthetic/key_value_recall` | tests context binding and delayed retrieval. |
| 3 | `synthetic/carry_arithmetic` | harder algorithmic arithmetic with non-smooth carry boundaries. |
| 4 | `synthetic/stack_brackets` | tests state tracking beyond shallow templates. |
| 5 | `synthetic/noisy_copy_transform` | tests robust span transforms under noise. |

Do not treat a task as a meaningful optimizer benchmark if the best controls reach loss `<0.1` at the target budget. The result may still be useful for debugging, but the PPL scale is too compressed for a research claim.

## Synthetic Artifact

```bash
.venv-cu128/bin/python experiments/scripts/summarize_synthetic_fair_full_20260529.py
```

The default summary combines Code/Symbolic from `experiments/runs/synthetic_fair_full_20260529/` with the clean Reasoning mix rerun from `experiments/runs/synthetic_fair_reasoning_mix_20260529/`. The committed artifact includes `summary.md`, `curve_diagnostics.md`, CSV curve diagnostics, raw curve CSVs, and loss/PPL PNGs for each task.
