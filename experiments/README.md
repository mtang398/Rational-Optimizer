# Experiments

This directory keeps E1 matched main-suite results, completed E2 DCLM and FineWeb-Edu M0/300M result packages, and the WikiText demo anchor. New paper experiments use the manifest workflow below.

Paper runs follow the manifest rule below. Exact submitted commands are recorded in `ICLR_RUN_COMMANDS.md`.

## Result Pointers

```text
ICLR_RUN_STATUS.md
results/iclr26_runtime_summary_2026_06_11/
results/iclr26_e2_dclm_2026_06_10/
results/iclr26_e2_fineweb_edu_2026_06_12/
results/iclr26_e1_figures/
runs/iclr26_main/        # local raw JSONL, ignored
results/rlb_matrix_policy_muon_switch_2026_05_28/  # WikiText demo anchor
```

## Completed Runtime Summary

Per optimizer/activation-combo runtimes for completed paper cells are tracked in `results/iclr26_runtime_summary_2026_06_11/`. The package covers all completed E1 M0/100M rows plus completed E2 M0/300M DCLM and FineWeb-Edu rows. It excludes E2 FineWeb rows `330-374` until that queued cell completes, and excludes E2 rows `375+` because they have not been queued/completed yet.

Default headline runtime tables use the clean aggregate: E1 FineWeb-Edu seed `2027` rows `75-89` are excluded from clean E1 runtime because Slurm job `158117` had `Restarts=6` and produced restart/node-contaminated throughput outliers. Raw all-completed CSVs are still kept for provenance.

The runtime metric is the JSONL `summary.total_seconds` training-harness wall time per manifest row. This excludes Slurm queue wait, dependency wait, token-cache construction, extension compilation, and launcher overhead.

## Current E2 300M Results

E2 M0/300M is complete for DCLM rows `240-284` and FineWeb-Edu rows `285-329`: each completed cell has three seeds, 15 fixed methods per seed, final eval at step `9150`, `32768` global tokens/step, and about `299.8M` train tokens per run.

E2 FineWeb rows `330-374` are now queued as the next dataset only, split into two one-row dependency chains. E2 rows `375+` are not queued.

### DCLM

Tracked package: `results/iclr26_e2_dclm_2026_06_10/`.

MatrixPolicy is best on all three DCLM E2 seeds. Mean final val loss is `3.957627 +/- 0.030713`; the next-best aggregate methods are `silu_lion` at `3.993430 +/- 0.023038`, `rlb_muon` at `3.993489 +/- 0.029634`, `rlb_lion` at `3.994293 +/- 0.030088`.

| Target loss | MP all-hit mean | Vs fastest non-MP: MP -> comparator (seeds) | Saved | Saved % | Vs SiLU+AdamW: MP -> AdamW (seeds) | Saved | Saved % |
| ---: | ---: | --- | ---: | ---: | --- | ---: | ---: |
| 4.40 | 74.3M | 74.3M -> 80.8M (3/3) | 6.6M | 8.1% | 74.3M -> 93.4M (3/3) | 19.1M | 20.5% |
| 4.30 | 101.0M | 101.0M -> 104.9M (3/3) | 3.8M | 3.6% | 101.0M -> 120.7M (3/3) | 19.7M | 16.3% |
| 4.20 | 133.3M | 133.3M -> 139.3M (3/3) | 6.0M | 4.3% | 133.3M -> 161.1M (3/3) | 27.9M | 17.3% |
| 4.10 | 176.4M | 176.4M -> 187.9M (3/3) | 11.5M | 6.1% | 176.4M -> 227.7M (3/3) | 51.3M | 22.5% |
| 4.05 | 205.3M | 205.3M -> 222.8M (3/3) | 17.5M | 7.8% | 185.1M -> 244.1M (1/3) | 59.0M | 24.2% |
| 4.00 | 244.7M | 232.7M -> 267.9M (2/3) | 35.2M | 13.1% | not reached (0/3) | not reached | n/a |

### FineWeb-Edu

Tracked package: `results/iclr26_e2_fineweb_edu_2026_06_12/`.

MatrixPolicy is best on all three FineWeb-Edu E2 seeds. Mean final val loss is `3.706480 +/- 0.020263`; the next-best aggregate methods are `rlb_muon` at `3.738164 +/- 0.021014`, `silu_lion` at `3.744017 +/- 0.020802`, `rlb_lion` at `3.745142 +/- 0.021429`.

| Target loss | MP all-hit mean | Vs fastest non-MP: MP -> comparator (seeds) | Saved | Saved % | Vs SiLU+AdamW: MP -> AdamW (seeds) | Saved | Saved % |
| ---: | ---: | --- | ---: | ---: | --- | ---: | ---: |
| 4.20 | 74.3M | 74.3M -> 81.4M (3/3) | 7.1M | 8.7% | 74.3M -> 93.4M (3/3) | 19.1M | 20.5% |
| 4.10 | 100.5M | 100.5M -> 102.7M (3/3) | 2.2M | 2.1% | 100.5M -> 118.0M (3/3) | 17.5M | 14.8% |
| 4.00 | 127.8M | 127.8M -> 130.5M (3/3) | 2.7M | 2.1% | 127.8M -> 151.8M (3/3) | 24.0M | 15.8% |
| 3.90 | 163.3M | 163.3M -> 167.7M (3/3) | 4.4M | 2.6% | 163.3M -> 200.4M (3/3) | 37.1M | 18.5% |
| 3.85 | 185.1M | 185.1M -> 191.1M (3/3) | 6.0M | 3.1% | 185.1M -> 237.0M (3/3) | 51.9M | 21.9% |
| 3.80 | 211.9M | 211.9M -> 224.5M (3/3) | 12.6M | 5.6% | 205.6M -> 287.5M (2/3) | 81.9M | 28.5% |
| 3.75 | 247.4M | 237.6M -> 262.1M (2/3) | 24.6M | 9.4% | not reached (0/3) | not reached | n/a |

Full per-method and per-seed tables are in the tracked E2 result package READMEs.

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


Regenerate completed-cell runtime tables with:

```bash
python3 experiments/scripts/summarize_iclr26_runtimes.py
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
