# Experiments

This directory contains launchers, summarizers, compact result artifacts, and raw JSONL traces for the current real-LM evidence.

The current public evidence package is the 3-seed FineWeb/FineWeb-Edu pilot replication. Older one-seed plots and the WikiText anchor are retained as supporting artifacts. The final paper plan is not centered on this pilot; it must be driven by the locked headline benchmark in `ICLR_OPTIMIZER_EXPERIMENT_BLUEPRINT.md`.

## Result Packages

| package | role |
| --- | --- |
| `ICLR_OPTIMIZER_EXPERIMENT_BLUEPRINT.md` | Evidence blueprint: why the paper needs tuned baselines, final-budget comparisons, speed, scale, transfer, and mechanism. |
| `ICLR_EXACT_RUN_PLAN.md` | Exact runnable matrix: model sizes, datasets, token budgets, seeds, grids, final rows, mechanism runs, and first Slurm commands. |
| `results/real_lm_multiseed_2026_05_31/` | Primary current preliminary 3-seed tables, bootstrap gap CIs, curve CSVs, and multi-seed mean plots. |
| `runs/real_lm_multiseed_20260531/` | Compact raw JSONL traces for seed 2027 and seed 3407 runs. |
| `results/real_lm_screen_2026_05_30/` | Seed-1337 baseline summary, curves, and one-seed plot images. |
| `results/rlb_matrix_policy_muon_switch_2026_05_28/` | Older WikiText-103 same-LR anchor. |

Slurm `.out` files are local logs and are ignored. Compact JSONL traces are small enough to commit and are used by the multi-seed summarizer.

## Primary 3-Seed Summary

Regenerate the primary summary:

```bash
.venv-cu128/bin/python experiments/scripts/summarize_real_lm_multiseed.py \
  --run-root experiments/runs/real_lm_multiseed_20260531 \
  --baseline-summary-csv experiments/results/real_lm_screen_2026_05_30/summary.csv \
  --baseline-seed 1337
```

Outputs:

```text
results/real_lm_multiseed_2026_05_31/summary.md
results/real_lm_multiseed_2026_05_31/per_seed_summary.csv
results/real_lm_multiseed_2026_05_31/aggregate_summary.csv
results/real_lm_multiseed_2026_05_31/matrix_policy_gap_bootstrap_ci.csv
results/real_lm_multiseed_2026_05_31/eval_curves.csv
results/real_lm_multiseed_2026_05_31/train_curves.csv
results/real_lm_multiseed_2026_05_31/*_mean*.png
```

### FineWeb

| method | n | mean val loss | mean PPL | gap vs SiLU+AdamW | gap vs best control |
| --- | ---: | ---: | ---: | ---: | ---: |
| SiLU+AdamW | 3 | 4.528963 | 92.69 | 0.000000 | -0.006960 |
| RLB+AdamW | 3 | 4.522311 | 92.08 | 0.006653 | -0.000308 |
| SiLU+Muon | 3 | 4.566661 | 96.28 | -0.037698 | -0.044658 |
| RLB+Muon | 3 | 4.571341 | 96.70 | -0.042377 | -0.049337 |
| RLB+MatrixPolicy (group-stat) | 3 | 4.369701 | 79.04 | 0.159263 | 0.152302 |

### FineWeb-Edu

| method | n | div | mean val loss | mean PPL | gap vs SiLU+AdamW | gap vs best control |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SiLU+AdamW | 3 | 0 | 4.223572 | 68.28 | 0.000000 | -0.000748 |
| RLB+AdamW | 3 | 1 | 5.618928 | 1545.54 | -1.395356 | -1.396103 |
| SiLU+Muon | 3 | 0 | 4.258871 | 70.74 | -0.035300 | -0.036047 |
| RLB+Muon | 3 | 0 | 4.263744 | 71.08 | -0.040173 | -0.040920 |
| RLB+MatrixPolicy (group-stat) | 3 | 0 | 4.069422 | 58.52 | 0.154149 | 0.153402 |

Positive gaps mean lower validation loss than the comparison row.

Bootstrap CIs in `matrix_policy_gap_bootstrap_ci.csv` are paired over seeds. With only 3 seeds, treat them as a stability check rather than a definitive uncertainty estimate.

Generated multi-seed mean plots:

```text
fineweb_validation_loss_mean.png
fineweb_validation_loss_mean_zoom_step1000.png
fineweb_validation_ppl_mean.png
fineweb_validation_ppl_mean_zoom_step1000.png
fineweb_training_loss_mean.png
fineweb_edu_validation_loss_mean.png
fineweb_edu_validation_loss_mean_zoom_step1000.png
fineweb_edu_validation_ppl_mean.png
fineweb_edu_validation_ppl_mean_zoom_step1000.png
fineweb_edu_training_loss_mean.png
```


## Figures

The 3-seed tables are the primary result. These plots are kept in the README because curve shape matters for optimizer claims, especially early training and divergent controls.

### Multi-Seed Mean +/- Std Curves

Each line is the seed mean; shaded bands are +/- 1 std. PPL plots omit divergent/nonfinite seed-method rows.

![FineWeb mean validation loss](results/real_lm_multiseed_2026_05_31/fineweb_validation_loss_mean.png)

![FineWeb mean validation loss zoom](results/real_lm_multiseed_2026_05_31/fineweb_validation_loss_mean_zoom_step1000.png)

![FineWeb mean validation PPL](results/real_lm_multiseed_2026_05_31/fineweb_validation_ppl_mean.png)

![FineWeb mean validation PPL zoom](results/real_lm_multiseed_2026_05_31/fineweb_validation_ppl_mean_zoom_step1000.png)

![FineWeb mean training loss](results/real_lm_multiseed_2026_05_31/fineweb_training_loss_mean.png)

![FineWeb-Edu mean validation loss](results/real_lm_multiseed_2026_05_31/fineweb_edu_validation_loss_mean.png)

![FineWeb-Edu mean validation loss zoom](results/real_lm_multiseed_2026_05_31/fineweb_edu_validation_loss_mean_zoom_step1000.png)

![FineWeb-Edu mean validation PPL](results/real_lm_multiseed_2026_05_31/fineweb_edu_validation_ppl_mean.png)

![FineWeb-Edu mean validation PPL zoom](results/real_lm_multiseed_2026_05_31/fineweb_edu_validation_ppl_mean_zoom_step1000.png)

![FineWeb-Edu mean training loss](results/real_lm_multiseed_2026_05_31/fineweb_edu_training_loss_mean.png)

### Seed-1337 Detailed Curves

![FineWeb validation loss](results/real_lm_screen_2026_05_30/fineweb_validation_loss.png)

![FineWeb validation loss zoom](results/real_lm_screen_2026_05_30/fineweb_validation_loss_zoom_step1000.png)

![FineWeb validation PPL](results/real_lm_screen_2026_05_30/fineweb_validation_ppl.png)

![FineWeb validation PPL zoom](results/real_lm_screen_2026_05_30/fineweb_validation_ppl_zoom_step1000.png)

![FineWeb training loss](results/real_lm_screen_2026_05_30/fineweb_training_loss.png)

![FineWeb-Edu validation loss](results/real_lm_screen_2026_05_30/fineweb_edu_validation_loss.png)

![FineWeb-Edu validation loss zoom](results/real_lm_screen_2026_05_30/fineweb_edu_validation_loss_zoom_step1000.png)

![FineWeb-Edu validation PPL](results/real_lm_screen_2026_05_30/fineweb_edu_validation_ppl.png)

![FineWeb-Edu validation PPL zoom](results/real_lm_screen_2026_05_30/fineweb_edu_validation_ppl_zoom_step1000.png)

![FineWeb-Edu training loss](results/real_lm_screen_2026_05_30/fineweb_edu_training_loss.png)

![WikiText validation loss](results/rlb_matrix_policy_muon_switch_2026_05_28/same_lr_validation_loss.png)

![WikiText validation PPL](results/rlb_matrix_policy_muon_switch_2026_05_28/same_lr_validation_ppl.png)

![WikiText training loss from step 1](results/rlb_matrix_policy_muon_switch_2026_05_28/same_lr_training_loss_from_step1.png)

## One-Seed Curve Package

The May 30 package contains the seed-1337 curves and plot images:

```text
results/real_lm_screen_2026_05_30/summary.md
results/real_lm_screen_2026_05_30/summary.csv
results/real_lm_screen_2026_05_30/eval_curves.csv
results/real_lm_screen_2026_05_30/train_curves.csv
```

Regenerate it:

```bash
.venv-cu128/bin/python experiments/scripts/summarize_real_lm_screen_20260530.py
```

The plotted figures in this package are illustrative single-seed curves. The current aggregate claim should cite the multi-seed package above.

## WikiText-103 Anchor

WikiText-103 remains useful as an older same-LR LM anchor:

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

It is not the main result because the current real-corpus gaps are larger and more directly relevant to pretraining.

## Launching More Runs

Read `ICLR_EXACT_RUN_PLAN.md` before launching. The next launches should be new accepted-paper-style experiments:

```text
1. tiny DCLM/Dolma loader validation
2. M1 memory smoke
3. M0 speed-to-target runs on FineWeb-Edu and DCLM at 100M/300M
4. batch-size/overhead and memory profiling
5. scale, transfer, long-horizon, post-training, diagnostics
6. LR/WD landscapes as reviewer defense
7. ablations last
```

Main historical real-LM launcher:

```bash
sbatch experiments/scripts/run_real_lm_screen_20260530.sh
```

Resource rule: each submitted GPU job uses 4 A6000 GPUs; keep at most two active jobs. Use dependencies or manual sequencing so active usage never exceeds 8 A6000s.

Example M0 speed-to-target launch pattern:

```bash
REAL_LM_TASKS="fineweb_edu" \
SEEDS="1337" \
RUN_SUFFIX="iclr_speed_m0_100m_seed1337" \
OUTPUT_ROOT="experiments/runs/iclr_speed_m0" \
TOKEN_CACHE_DIR="experiments/cache/tokens_iclr_speed_m0" \
MAX_TRAIN_TOKENS=100000000 \
MAX_VAL_TOKENS=8000000 \
VAL_SKIP_TOKENS=610000000 \
STEPS=3050 \
EVAL_INTERVAL=50 \
LOG_INTERVAL=10 \
INCLUDE_MUON=1 \
sbatch experiments/scripts/run_real_lm_screen_20260530.sh
```

## Resource And Artifact Policy

Hard limits:

```text
max 4 A6000 GPUs per job
max 8 A6000 GPUs active total
repo size below 200G
```

Artifact policy:

```text
commit compact summaries and JSONL traces needed for table reproduction
do not commit Slurm .out logs
do not commit cache directories, checkpoints, or model weights
keep raw large datasets in cache paths ignored by git
```
