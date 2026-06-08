# Experiments

This directory keeps E1 matched main-suite results and the WikiText demo anchor. New paper experiments use the manifest workflow below.

## Result Pointers

```text
ICLR_RUN_STATUS.md
results/iclr26_e1_figures/
runs/iclr26_main/        # local raw JSONL, ignored
results/rlb_matrix_policy_muon_switch_2026_05_28/  # WikiText demo anchor
```

Paper runs follow the manifest rule below. Exact submitted commands are recorded in `ICLR_RUN_COMMANDS.md`.

## Current E1 Main-Suite Results

Current E1 M0/100M results are tracked in `ICLR_RUN_STATUS.md` and figures are under `results/iclr26_e1_figures/`. E1 uses five corpora, three seeds, 15 matched methods per dataset/seed cell, 4 A6000 GPUs per job, and dense validation every 50 steps. E1 M0/100M is complete.

| Dataset | MatrixPolicy final val loss | next best current method | gap |
| --- | ---: | ---: | ---: |
| DCLM | 4.256224 +/- 0.004972 | rlb_lion 4.305728 +/- 0.005836 | 0.049505 |
| FineWeb-Edu | 4.088240 +/- 0.009434 | rlb_lion 4.142669 +/- 0.006812 | 0.054429 |
| FineWeb | 4.318581 +/- 0.010914 | rlb_lion 4.367062 +/- 0.007532 | 0.048481 |
| Dolma-sample | 4.323851 +/- 0.004565 | rlb_lion 4.369254 +/- 0.005561 | 0.045403 |
| C4 | 4.285119 +/- 0.020677 | rlb_lion 4.335663 +/- 0.020917 | 0.050544 |

### E1 Dense Curve Figures

These are the same E1 figure panels embedded in `ICLR_RUN_STATUS.md`. The curves use completed E1 runs at native logging cadence: validation every 50 steps and training loss every 10 steps. Shaded bands are mean +/- 1 sample std over seeds. The all-method view includes MatrixPolicy, AdamW, Lion, SOAP, Muon, ScheduleFree, and CAME rows; the clean view omits SOAP from the plotted comparison.

#### DCLM

All-method view:

![DCLM E1 validation loss mean +/- std, all methods](results/iclr26_e1_figures/dclm_core_validation_loss_mean_std.svg)

![DCLM E1 validation PPL mean +/- std, all methods](results/iclr26_e1_figures/dclm_core_validation_ppl_mean_std.svg)

![DCLM E1 training loss mean +/- std, all methods](results/iclr26_e1_figures/dclm_core_training_loss_mean_std.svg)

Clean comparison view:

![DCLM E1 validation loss mean +/- std, clean comparison](results/iclr26_e1_figures/dclm_clean_validation_loss_mean_std.svg)

![DCLM E1 validation PPL mean +/- std, clean comparison](results/iclr26_e1_figures/dclm_clean_validation_ppl_mean_std.svg)

![DCLM E1 training loss mean +/- std, clean comparison](results/iclr26_e1_figures/dclm_clean_training_loss_mean_std.svg)

#### FineWeb-Edu

All-method view:

![FineWeb-Edu E1 validation loss mean +/- std, all methods](results/iclr26_e1_figures/fineweb_edu_core_validation_loss_mean_std.svg)

![FineWeb-Edu E1 validation PPL mean +/- std, all methods](results/iclr26_e1_figures/fineweb_edu_core_validation_ppl_mean_std.svg)

![FineWeb-Edu E1 training loss mean +/- std, all methods](results/iclr26_e1_figures/fineweb_edu_core_training_loss_mean_std.svg)

Clean comparison view:

![FineWeb-Edu E1 validation loss mean +/- std, clean comparison](results/iclr26_e1_figures/fineweb_edu_clean_validation_loss_mean_std.svg)

![FineWeb-Edu E1 validation PPL mean +/- std, clean comparison](results/iclr26_e1_figures/fineweb_edu_clean_validation_ppl_mean_std.svg)

![FineWeb-Edu E1 training loss mean +/- std, clean comparison](results/iclr26_e1_figures/fineweb_edu_clean_training_loss_mean_std.svg)

#### FineWeb

All-method view:

![FineWeb E1 validation loss mean +/- std, all methods](results/iclr26_e1_figures/fineweb_core_validation_loss_mean_std.svg)

![FineWeb E1 validation PPL mean +/- std, all methods](results/iclr26_e1_figures/fineweb_core_validation_ppl_mean_std.svg)

![FineWeb E1 training loss mean +/- std, all methods](results/iclr26_e1_figures/fineweb_core_training_loss_mean_std.svg)

Clean comparison view:

![FineWeb E1 validation loss mean +/- std, clean comparison](results/iclr26_e1_figures/fineweb_clean_validation_loss_mean_std.svg)

![FineWeb E1 validation PPL mean +/- std, clean comparison](results/iclr26_e1_figures/fineweb_clean_validation_ppl_mean_std.svg)

![FineWeb E1 training loss mean +/- std, clean comparison](results/iclr26_e1_figures/fineweb_clean_training_loss_mean_std.svg)

#### Dolma-sample

All-method view:

![Dolma-sample E1 validation loss mean +/- std, all methods](results/iclr26_e1_figures/dolma_sample_core_validation_loss_mean_std.svg)

![Dolma-sample E1 validation PPL mean +/- std, all methods](results/iclr26_e1_figures/dolma_sample_core_validation_ppl_mean_std.svg)

![Dolma-sample E1 training loss mean +/- std, all methods](results/iclr26_e1_figures/dolma_sample_core_training_loss_mean_std.svg)

Clean comparison view:

![Dolma-sample E1 validation loss mean +/- std, clean comparison](results/iclr26_e1_figures/dolma_sample_clean_validation_loss_mean_std.svg)

![Dolma-sample E1 validation PPL mean +/- std, clean comparison](results/iclr26_e1_figures/dolma_sample_clean_validation_ppl_mean_std.svg)

![Dolma-sample E1 training loss mean +/- std, clean comparison](results/iclr26_e1_figures/dolma_sample_clean_training_loss_mean_std.svg)

#### C4

All-method view:

![C4 E1 validation loss mean +/- std, all methods](results/iclr26_e1_figures/c4_en_core_validation_loss_mean_std.svg)

![C4 E1 validation PPL mean +/- std, all methods](results/iclr26_e1_figures/c4_en_core_validation_ppl_mean_std.svg)

![C4 E1 training loss mean +/- std, all methods](results/iclr26_e1_figures/c4_en_core_training_loss_mean_std.svg)

Clean comparison view:

![C4 E1 validation loss mean +/- std, clean comparison](results/iclr26_e1_figures/c4_en_clean_validation_loss_mean_std.svg)

![C4 E1 validation PPL mean +/- std, clean comparison](results/iclr26_e1_figures/c4_en_clean_validation_ppl_mean_std.svg)

![C4 E1 training loss mean +/- std, clean comparison](results/iclr26_e1_figures/c4_en_clean_training_loss_mean_std.svg)

Full E1 command history is recorded in `ICLR_RUN_COMMANDS.md`.

## WikiText Result

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

![WikiText validation loss](results/rlb_matrix_policy_muon_switch_2026_05_28/same_lr_validation_loss.png)

![WikiText validation PPL](results/rlb_matrix_policy_muon_switch_2026_05_28/same_lr_validation_ppl.png)

![WikiText training loss](results/rlb_matrix_policy_muon_switch_2026_05_28/same_lr_training_loss_from_step1.png)

## Forward Contract

Use `ICLR_EXACT_RUN_PLAN.md` and the generated manifest. Each matched cell must contain AdamW and MatrixPolicy rows with the same outer optimizer config:

```text
same dataset
same model
same train-token budget
same seed
same validation slice
same sequence length
same global tokens per step
same eval interval
same lr
same min_lr
same weight_decay
```

Do not launch one-sided AdamW grids. Do not launch MatrixPolicy sensitivity rows unless the corresponding AdamW/RLB control rows with the same outer config are in the same manifest cell. Sensitivity maps and method ablations come after main evidence.

## Reproducibility Map

Current paper runs are reproduced from these files:

```text
experiments/scripts/build_iclr26_main_manifest.py
experiments/manifests/iclr26_main_manifest.csv
experiments/scripts/run_iclr26_manifest_job.sh
training/run_lm_optimizer_sweep.sbatch
training/transformer_lm_compare.py
optimizer_design/matrix_policy_optimizer.py
optimizer_design/transport_onpolicy_optimizer.py
optimizer_design/baseline_optimizers.py
activation/rational_opt/rational.py
```

Regenerate paper figures and tables from raw JSONL with:

```bash
python3 experiments/scripts/plot_iclr26_e1_curves.py --status-md experiments/ICLR_RUN_STATUS.md
```

The curated WikiText anchor is tracked under `results/rlb_matrix_policy_muon_switch_2026_05_28/`; raw WikiText launcher output is local run data and is ignored.

## Commands

Generate the main manifest:

```bash
python3 experiments/scripts/build_iclr26_main_manifest.py \
  --output experiments/manifests/iclr26_main_manifest.csv \
  --print-summary
```

Run one manifest row:

```bash
CONFIRM_ICLR26_MANIFEST=1 \
MANIFEST=experiments/manifests/iclr26_main_manifest.csv \
ROW_START=0 \
ROW_LIMIT=1 \
sbatch experiments/scripts/run_iclr26_manifest_job.sh
```

Run a bounded shard:

```bash
CONFIRM_ICLR26_MANIFEST=1 \
MANIFEST=experiments/manifests/iclr26_main_manifest.csv \
ROW_START=40 \
ROW_LIMIT=4 \
sbatch experiments/scripts/run_iclr26_manifest_job.sh
```

Do not run more than two 4-GPU jobs at once.

## Output Policy

Raw new outputs belong under ignored `experiments/runs/`. Token caches belong under ignored `experiments/cache/`. New tracked summaries should be created only after matched cells complete and must include dense curves, AUC, timing, divergence markers, and exact manifest row IDs.
