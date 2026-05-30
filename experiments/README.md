# Experiments

This folder contains launchers, summarizers, and committed result artifacts. The current public evidence is intentionally compact: the May 30 real-corpus screen and the older WikiText-103 anchor.

Raw JSONL files and Slurm logs stay under `experiments/runs/` and are not committed. Research-facing tables and figures live under `experiments/results/`.

## Result Packages

| package | contents |
| --- | --- |
| `results/real_lm_screen_2026_05_30/` | FineWeb and FineWeb-Edu real-corpus screen, summary tables, train/eval curves, PPL/loss plots. |
| `results/rlb_matrix_policy_muon_switch_2026_05_28/` | WikiText-103 same-LR comparison and plots. |

The older saturated synthetic and single-seed synthetic gauge-stress bundles were removed from the tracked result set. They are not part of the current research claim.

## Real-Corpus Screen

Launcher: `experiments/scripts/run_real_lm_screen_20260530.sh`

Protocol:

```text
model: 12 layers, d_model 768, 12 heads, 123.6M parameters
train tokens: 100,000,000
validation tokens: 4,000,000
validation offset: 110,000,000 stream tokens
sequence length: 256
global tokens per step: 32,768
steps: 3,050
seed: 1337
logging: train every 10 steps, validation every 50 steps
base LR: optimizer_lr=3e-4, optimizer_min_lr=3e-5
```

Completed tasks:

| task | HF dataset/config | purpose |
| --- | --- | --- |
| FineWeb | `HuggingFaceFW/fineweb`, `sample-10BT` | noisier broad web pretraining slice. |
| FineWeb-Edu | `HuggingFaceFW/fineweb-edu`, `sample-10BT` | cleaner educational web pretraining slice. |

### FineWeb Results

| method | last finite validation loss | last finite PPL | val loss AUC <= 1000 | val loss AUC <= 2000 | note |
| --- | ---: | ---: | ---: | ---: | --- |
| SiLU+AdamW | 4.504617 | 90.43 | 5.993426 | 5.401559 | complete |
| RLB+AdamW | 4.493013 | 89.39 | 5.954484 | 5.373016 | complete |
| SiLU+Muon | 4.535766 | 93.29 | 6.664512 | 5.786310 | complete |
| RLB+Muon | 4.548868 | 94.53 | 6.585091 | 5.752002 | complete |
| RLB+MatrixPolicy (group-stat) | 4.344150 | 77.03 | 5.850945 | 5.262783 | complete |

### FineWeb-Edu Results

| method | last finite validation loss | last finite PPL | val loss AUC <= 1000 | val loss AUC <= 2000 | note |
| --- | ---: | ---: | ---: | ---: | --- |
| SiLU+AdamW | 4.225019 | 68.38 | 5.835354 | 5.186270 | complete |
| RLB+AdamW | 8.411884 | 4500.23 | 9.684973 | 9.684973 | train nonfinite at step 80; validation nonfinite at step 100 |
| SiLU+Muon | 4.252612 | 70.29 | 6.505154 | 5.563970 | complete |
| RLB+Muon | 4.271556 | 71.63 | 6.425483 | 5.529865 | complete |
| RLB+MatrixPolicy (group-stat) | 4.072055 | 58.68 | 5.670071 | 5.041694 | complete |

Summary files:

```text
results/real_lm_screen_2026_05_30/summary.md
results/real_lm_screen_2026_05_30/summary.csv
results/real_lm_screen_2026_05_30/eval_curves.csv
results/real_lm_screen_2026_05_30/train_curves.csv
```

## WikiText-103 Anchor

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

WikiText remains useful because it is a real LM comparison with strong controls, but the current real-corpus FineWeb/FineWeb-Edu gaps are larger and more important.

## Figures

FineWeb validation loss and PPL:

![FineWeb validation loss](results/real_lm_screen_2026_05_30/fineweb_validation_loss.png)

![FineWeb validation PPL](results/real_lm_screen_2026_05_30/fineweb_validation_ppl.png)

FineWeb training loss:

![FineWeb training loss](results/real_lm_screen_2026_05_30/fineweb_training_loss.png)

FineWeb-Edu validation loss and PPL:

![FineWeb-Edu validation loss](results/real_lm_screen_2026_05_30/fineweb_edu_validation_loss.png)

![FineWeb-Edu validation PPL](results/real_lm_screen_2026_05_30/fineweb_edu_validation_ppl.png)

FineWeb-Edu training loss:

![FineWeb-Edu training loss](results/real_lm_screen_2026_05_30/fineweb_edu_training_loss.png)

WikiText-103 validation and training curves:

![WikiText validation loss](results/rlb_matrix_policy_muon_switch_2026_05_28/same_lr_validation_loss.png)

![WikiText validation PPL](results/rlb_matrix_policy_muon_switch_2026_05_28/same_lr_validation_ppl.png)

![WikiText training loss from step 1](results/rlb_matrix_policy_muon_switch_2026_05_28/same_lr_training_loss_from_step1.png)

## Regeneration

Real-corpus summary and plots:

```bash
.venv-cu128/bin/python experiments/scripts/summarize_real_lm_screen_20260530.py
```

The launcher supports bounded streaming caches and requeue:

```text
#SBATCH --gres=gpu:nvidia_rtx_a6000:4
#SBATCH --time=72:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
```

The runtime rule remains: use A6000 GPUs only and keep total active allocation at or below 8 A6000s.
