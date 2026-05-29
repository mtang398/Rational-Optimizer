# Experiments

This folder contains launchers, summarizers, and committed result artifacts. The purpose is to make each experiment answer a specific research question.

Raw JSONL files and Slurm logs stay under `experiments/runs/` and are not committed. Research-facing artifacts live under `experiments/results/`.

## Experimental Questions

| experiment family | question |
| --- | --- |
| WikiText-103 same-LR comparison | Does MatrixPolicy beat strong generic controls on a real LM task? |
| Dense synthetic curves | Is the early rational speed visible from step 1 in train and validation curves? |
| Positive gauge stress | Does MatrixPolicy handle equivalent-function RLB reparameterizations better than generic optimizers? |
| Hard non-saturated tasks | Does early speed become a real final-loss gap when the task is not near the floor? |

## Verified WikiText-103 Result

| method | final loss | final PPL |
| --- | ---: | ---: |
| RLB MatrixPolicy-Muon | 3.476232 | 32.34 |
| RLB Smooth-MatrixPolicy | 3.493210 | 32.89 |
| SiLU/SwiGLU+AdamW beta2=0.999 | 3.549346 | 34.79 |
| RLB+AdamW beta2=0.999 | 3.550018 | 34.81 |
| RLB+AdamW | 3.617501 | 37.24 |
| SiLU/SwiGLU+AdamW | 3.621982 | 37.41 |
| SiLU/SwiGLU+Muon | 3.644921 | 38.28 |
| RLB+Muon | 3.657877 | 38.78 |

This is the current evidence boundary: the loss gap is `0.0731`, below the target `0.2-0.3`.

## Figures

WikiText-103 validation loss:

![WikiText validation loss](results/rlb_matrix_policy_muon_switch_2026_05_28/same_lr_validation_loss.png)

WikiText-103 validation PPL:

![WikiText validation PPL](results/rlb_matrix_policy_muon_switch_2026_05_28/same_lr_validation_ppl.png)

WikiText-103 training loss from step 1:

![WikiText training loss from step 1](results/rlb_matrix_policy_muon_switch_2026_05_28/same_lr_training_loss_from_step1.png)

Sparse synthetic curves:

![Synthetic Code validation loss](results/synthetic_fair_full_2026_05_29/synthetic_code_validation_loss.png)

![Synthetic Code validation PPL](results/synthetic_fair_full_2026_05_29/synthetic_code_validation_ppl.png)

![Synthetic Symbolic validation loss](results/synthetic_fair_full_2026_05_29/synthetic_symbolic_validation_loss.png)

![Synthetic Symbolic validation PPL](results/synthetic_fair_full_2026_05_29/synthetic_symbolic_validation_ppl.png)

![Reasoning mix validation loss](results/synthetic_fair_full_2026_05_29/synthetic_reasoning_mix_validation_loss.png)

![Reasoning mix validation PPL](results/synthetic_fair_full_2026_05_29/synthetic_reasoning_mix_validation_ppl.png)

## Synthetic Interpretation

The sparse synthetic curves suggest that rational rows can drop faster early, especially on Code and Reasoning mix. They do not yet prove a final optimizer advantage because the tasks become low-loss and the sampling is too sparse.

Use these metrics for dense synthetic analysis:

```text
train AUC
validation AUC
time to fixed loss threshold
final loss/PPL
curve crossing step
instability or late regression
```

Final bars should be secondary whenever loss is near the floor.

## Gauge-Stress Protocol

The gauge-stress benchmark applies the exact RLB gauge transform at initialization:

```text
W_in[g]  <- a_g W_in[g]
W_out[g] <- W_out[g] / a_g
log a_g ~ Uniform[-s, s]
```

The represented function is unchanged for `a_g > 0`. The metric is degradation under gauge stress:

```text
D(metric) = metric(gauge_log_scale = 2.0) - metric(gauge_log_scale = 0.0)
```

Report this for `RLB+AdamW`, `RLB+Muon`, `RLB MatrixPolicy`, and `RLB MatrixPolicy group-stat` using train AUC, validation AUC, time-to-threshold, and final loss.

## Regeneration

Sparse synthetic summary:

```bash
.venv-cu128/bin/python experiments/scripts/summarize_synthetic_fair_full_20260529.py
```

Dense synthetic summary after completion:

```bash
.venv-cu128/bin/python experiments/scripts/summarize_synthetic_fair_full_20260529.py \
  --run-root experiments/runs/synthetic_dense_curves_20260529 \
  --suffix 20260529_dense_curve \
  --result-dir experiments/results/synthetic_dense_curves_2026_05_29
```

New experiments should use A6000 GPUs only and keep total active allocation at or below 8 A6000s.
