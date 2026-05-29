# Experiments

This folder contains Slurm launchers and compact result artifacts. Raw run folders and Slurm logs under `experiments/runs/` are local artifacts and should not be committed.

## Active Fair Rerun

```text
completed job: 937608, preempted after Code and Symbolic finished
active job:    951127, Reasoning mix rerun with Requeue=1
script:        experiments/scripts/run_synthetic_fair_full_20260529.sh
job name:      synth-reason
GPUs:          4x nvidia_rtx_a6000
walltime:      24h
run root:      experiments/runs/synthetic_fair_reasoning_mix_20260529/
result root:   experiments/results/synthetic_fair_full_2026_05_29/
```

| task | compared rows |
| --- | --- |
| `synthetic/code` | SiLU/SwiGLU+AdamW, RLB+AdamW, SiLU/SwiGLU+Muon, RLB+Muon, RLB MatrixPolicy, RLB MatrixPolicy group-stat |
| `synthetic/symbolic` | same rows |
| `synthetic/reasoning_mix` | same rows |

The run is same-LR across all rows. The launcher now has `--requeue`, a `USR1` time-signal requeue trap, and restart-safe skip/archive handling for completed versus partial rows.

After completion:

```bash
.venv-cu128/bin/python experiments/scripts/summarize_synthetic_fair_full_20260529.py
```

Expected compact outputs:

```text
summary.md
summary.csv
eval_curves.csv
train_curves.csv
synthetic_code_validation_loss.png
synthetic_code_validation_ppl.png
synthetic_code_training_loss.png
synthetic_symbolic_validation_loss.png
synthetic_symbolic_validation_ppl.png
synthetic_symbolic_training_loss.png
synthetic_reasoning_mix_validation_loss.png
synthetic_reasoning_mix_validation_ppl.png
synthetic_reasoning_mix_training_loss.png
```

## Verified WikiText Artifact

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

## Verified WikiText Result

| row | final loss | final PPL |
| --- | ---: | ---: |
| RLB MatrixPolicy-Muon | 3.476232 | 32.34 |
| RLB Smooth-MatrixPolicy | 3.493210 | 32.89 |
| SiLU/SwiGLU+AdamW beta2=0.999 | 3.549346 | 34.79 |
| RLB+AdamW beta2=0.999 | 3.550018 | 34.81 |
| RLB+AdamW | 3.617501 | 37.24 |
| SiLU/SwiGLU+AdamW | 3.621982 | 37.41 |
| SiLU/SwiGLU+Muon | 3.644921 | 38.28 |
| RLB+Muon | 3.657877 | 38.78 |

## Interpretation

The current verified best is still `RLB MatrixPolicy-Muon`. Plain Muon is weaker than AdamW controls on the verified WikiText run, and the recent beta2-tail, coefficient, role-depth, and late-Muon probes did not create a material gap. The fair synthetic job is running to check transfer cleanly across multiple small LLM tasks.

## Artifact Policy

Commit compact summaries, plots, and scripts. Do not commit raw JSONL run directories or Slurm logs under `experiments/runs/`.
