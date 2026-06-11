# RationalOPT

RationalOPT studies Rational Local Basis (RLB) variants inside causal Transformer language models and the `rational_matrix_policy_onpolicy` optimizer for pretraining.

Paper-facing results in this README include the completed E1 matched main suite and the completed E2 DCLM M0/300M cell. WikiText is kept as a small demo anchor.

## Result Pointers

```text
experiments/ICLR_RUN_STATUS.md
experiments/results/iclr26_runtime_summary_2026_06_11/
experiments/results/iclr26_e2_dclm_2026_06_10/
experiments/results/iclr26_e1_figures/
experiments/runs/iclr26_main/        # local raw JSONL, ignored
experiments/results/rlb_matrix_policy_muon_switch_2026_05_28/  # WikiText demo anchor
```

## Completed Runtime Summary

Per optimizer/activation-combo runtimes for completed paper cells are tracked in `experiments/results/iclr26_runtime_summary_2026_06_11/`. The package covers all completed E1 M0/100M rows and the completed E2 DCLM M0/300M rows. It deliberately excludes E2 FineWeb-Edu rows `285-329` because that dataset cell is still in progress.

Default headline runtime tables now use the clean aggregate: E1 FineWeb-Edu seed `2027` rows `75-89` are excluded from clean E1 runtime because Slurm job `158117` had `Restarts=6` and produced restart/node-contaminated throughput outliers. Raw all-completed CSVs are still kept for provenance.

The runtime metric is the JSONL `summary.total_seconds` training-harness wall time per manifest row. This is the comparable per-combo number for E1 because each E1 Slurm job ran a whole 15-row matched cell.

## Current E2 DCLM 300M Result

E2 M0/300M DCLM is complete for manifest rows `240-284`: three seeds, 15 fixed methods per seed, final eval at step `9150`, `32768` global tokens/step, and about `299.8M` train tokens per run. The tracked result package is `experiments/results/iclr26_e2_dclm_2026_06_10/`.

MatrixPolicy is best on all three seeds. Mean final validation loss is `3.957627 +/- 0.030713`; the next-best aggregate methods are `silu_lion` at `3.993430 +/- 0.023038`, `rlb_muon` at `3.993489 +/- 0.029634`, and `rlb_lion` at `3.994293 +/- 0.030088`. ADeMaMix diverged/non-finite for all E2 DCLM seeds.

| Target loss | MP all-hit mean | Vs fastest non-MP: MP -> comparator (seeds) | Saved | Saved % | Vs SiLU+AdamW: MP -> AdamW (seeds) | Saved | Saved % |
| ---: | ---: | --- | ---: | ---: | --- | ---: | ---: |
| 4.40 | 74.3M | 74.3M -> 80.8M (3/3) | 6.6M | 8.1% | 74.3M -> 93.4M (3/3) | 19.1M | 20.5% |
| 4.30 | 101.0M | 101.0M -> 104.9M (3/3) | 3.8M | 3.6% | 101.0M -> 120.7M (3/3) | 19.7M | 16.3% |
| 4.20 | 133.3M | 133.3M -> 139.3M (3/3) | 6.0M | 4.3% | 133.3M -> 161.1M (3/3) | 27.9M | 17.3% |
| 4.10 | 176.4M | 176.4M -> 187.9M (3/3) | 11.5M | 6.1% | 176.4M -> 227.7M (3/3) | 51.3M | 22.5% |
| 4.05 | 205.3M | 205.3M -> 222.8M (3/3) | 17.5M | 7.8% | 185.1M -> 244.1M (1/3) | 59.0M | 24.2% |
| 4.00 | 244.7M | 232.7M -> 267.9M (2/3) | 35.2M | 13.1% | not reached (0/3) | not reached | n/a |

Full per-method and per-seed tables are in `experiments/results/iclr26_e2_dclm_2026_06_10/README.md`.

## Current E1 Main-Suite Results

Current E1 M0/100M results are tracked in `experiments/ICLR_RUN_STATUS.md` and figures are under `experiments/results/iclr26_e1_figures/`. E1 uses five corpora, three seeds, 15 matched methods per dataset/seed cell, 4 A6000 GPUs per job, and dense validation every 50 steps. E1 M0/100M is complete.

| Dataset | MatrixPolicy final val loss | next best current method | gap |
| --- | ---: | ---: | ---: |
| DCLM | 4.256224 +/- 0.004972 | rlb_lion 4.305728 +/- 0.005836 | 0.049505 |
| FineWeb-Edu | 4.088240 +/- 0.009434 | rlb_lion 4.142669 +/- 0.006812 | 0.054429 |
| FineWeb | 4.318581 +/- 0.010914 | rlb_lion 4.367062 +/- 0.007532 | 0.048481 |
| Dolma-sample | 4.323851 +/- 0.004565 | rlb_lion 4.369254 +/- 0.005561 | 0.045403 |
| C4 | 4.285119 +/- 0.020677 | rlb_lion 4.335663 +/- 0.020917 | 0.050544 |

### E1 Dense Curve Figures

These are the same E1 figure panels embedded in `experiments/ICLR_RUN_STATUS.md`. The curves use completed E1 runs at native logging cadence: validation every 50 steps and training loss every 10 steps. Shaded bands are mean +/- 1 sample std over seeds. The all-method view includes MatrixPolicy, AdamW, Lion, SOAP, Muon, ScheduleFree, and CAME rows; the clean view omits SOAP from the plotted comparison.

#### DCLM

All-method view:

![DCLM E1 validation loss mean +/- std, all methods](experiments/results/iclr26_e1_figures/dclm_core_validation_loss_mean_std.svg)

![DCLM E1 validation PPL mean +/- std, all methods](experiments/results/iclr26_e1_figures/dclm_core_validation_ppl_mean_std.svg)

![DCLM E1 training loss mean +/- std, all methods](experiments/results/iclr26_e1_figures/dclm_core_training_loss_mean_std.svg)

Clean comparison view:

![DCLM E1 validation loss mean +/- std, clean comparison](experiments/results/iclr26_e1_figures/dclm_clean_validation_loss_mean_std.svg)

![DCLM E1 validation PPL mean +/- std, clean comparison](experiments/results/iclr26_e1_figures/dclm_clean_validation_ppl_mean_std.svg)

![DCLM E1 training loss mean +/- std, clean comparison](experiments/results/iclr26_e1_figures/dclm_clean_training_loss_mean_std.svg)

#### FineWeb-Edu

All-method view:

![FineWeb-Edu E1 validation loss mean +/- std, all methods](experiments/results/iclr26_e1_figures/fineweb_edu_core_validation_loss_mean_std.svg)

![FineWeb-Edu E1 validation PPL mean +/- std, all methods](experiments/results/iclr26_e1_figures/fineweb_edu_core_validation_ppl_mean_std.svg)

![FineWeb-Edu E1 training loss mean +/- std, all methods](experiments/results/iclr26_e1_figures/fineweb_edu_core_training_loss_mean_std.svg)

Clean comparison view:

![FineWeb-Edu E1 validation loss mean +/- std, clean comparison](experiments/results/iclr26_e1_figures/fineweb_edu_clean_validation_loss_mean_std.svg)

![FineWeb-Edu E1 validation PPL mean +/- std, clean comparison](experiments/results/iclr26_e1_figures/fineweb_edu_clean_validation_ppl_mean_std.svg)

![FineWeb-Edu E1 training loss mean +/- std, clean comparison](experiments/results/iclr26_e1_figures/fineweb_edu_clean_training_loss_mean_std.svg)

#### FineWeb

All-method view:

![FineWeb E1 validation loss mean +/- std, all methods](experiments/results/iclr26_e1_figures/fineweb_core_validation_loss_mean_std.svg)

![FineWeb E1 validation PPL mean +/- std, all methods](experiments/results/iclr26_e1_figures/fineweb_core_validation_ppl_mean_std.svg)

![FineWeb E1 training loss mean +/- std, all methods](experiments/results/iclr26_e1_figures/fineweb_core_training_loss_mean_std.svg)

Clean comparison view:

![FineWeb E1 validation loss mean +/- std, clean comparison](experiments/results/iclr26_e1_figures/fineweb_clean_validation_loss_mean_std.svg)

![FineWeb E1 validation PPL mean +/- std, clean comparison](experiments/results/iclr26_e1_figures/fineweb_clean_validation_ppl_mean_std.svg)

![FineWeb E1 training loss mean +/- std, clean comparison](experiments/results/iclr26_e1_figures/fineweb_clean_training_loss_mean_std.svg)

#### Dolma-sample

All-method view:

![Dolma-sample E1 validation loss mean +/- std, all methods](experiments/results/iclr26_e1_figures/dolma_sample_core_validation_loss_mean_std.svg)

![Dolma-sample E1 validation PPL mean +/- std, all methods](experiments/results/iclr26_e1_figures/dolma_sample_core_validation_ppl_mean_std.svg)

![Dolma-sample E1 training loss mean +/- std, all methods](experiments/results/iclr26_e1_figures/dolma_sample_core_training_loss_mean_std.svg)

Clean comparison view:

![Dolma-sample E1 validation loss mean +/- std, clean comparison](experiments/results/iclr26_e1_figures/dolma_sample_clean_validation_loss_mean_std.svg)

![Dolma-sample E1 validation PPL mean +/- std, clean comparison](experiments/results/iclr26_e1_figures/dolma_sample_clean_validation_ppl_mean_std.svg)

![Dolma-sample E1 training loss mean +/- std, clean comparison](experiments/results/iclr26_e1_figures/dolma_sample_clean_training_loss_mean_std.svg)

#### C4

All-method view:

![C4 E1 validation loss mean +/- std, all methods](experiments/results/iclr26_e1_figures/c4_en_core_validation_loss_mean_std.svg)

![C4 E1 validation PPL mean +/- std, all methods](experiments/results/iclr26_e1_figures/c4_en_core_validation_ppl_mean_std.svg)

![C4 E1 training loss mean +/- std, all methods](experiments/results/iclr26_e1_figures/c4_en_core_training_loss_mean_std.svg)

Clean comparison view:

![C4 E1 validation loss mean +/- std, clean comparison](experiments/results/iclr26_e1_figures/c4_en_clean_validation_loss_mean_std.svg)

![C4 E1 validation PPL mean +/- std, clean comparison](experiments/results/iclr26_e1_figures/c4_en_clean_validation_ppl_mean_std.svg)

![C4 E1 training loss mean +/- std, clean comparison](experiments/results/iclr26_e1_figures/c4_en_clean_training_loss_mean_std.svg)

Full E1 command history is recorded in `experiments/ICLR_RUN_COMMANDS.md`.

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

Validation loss:

![WikiText validation loss](experiments/results/rlb_matrix_policy_muon_switch_2026_05_28/same_lr_validation_loss.png)

Validation PPL:

![WikiText validation PPL](experiments/results/rlb_matrix_policy_muon_switch_2026_05_28/same_lr_validation_ppl.png)

Training loss from step 1:

![WikiText training loss](experiments/results/rlb_matrix_policy_muon_switch_2026_05_28/same_lr_training_loss_from_step1.png)

## Main Rule

Every comparison must match outer optimizer configs.

```text
If AdamW uses lr/min_lr/weight_decay = X in a matched cell, MatrixPolicy must also use lr/min_lr/weight_decay = X in that same cell.
```

A matched cell means the same dataset, model, token budget, seed, validation slice, sequence length, global batch, evaluation cadence, and run phase. Partial grids stay out of paper tables. A baseline grid cannot be compared against a non-grid MatrixPolicy row.

## Hard Resource Rules

```text
max 4 A6000 GPUs per job
max 8 A6000 GPUs active total
repo must stay below 200G
eval interval <= 50 for paper/protocol curves
curves and AUC are primary; final validation loss is only one table column
```

## Current Forward Plan

The exact plan is in `experiments/ICLR_EXACT_RUN_PLAN.md`; executed commands are recorded in `experiments/ICLR_RUN_COMMANDS.md`. It is now ordered as:

```text
0. manifest/loader preflight only
1. fixed-config M0 100M main evidence
2. fixed-config M0 300M main evidence
3. M1 scale check
4. 600M long-horizon frontier
5. throughput, memory, and equal-GPU-hour accounting
6. cross-corpus evaluation
7. corpus-shift continued training
8. sensitivity maps only after main evidence
9. method ablations last
```

No sensitivity map or method ablation should start before fixed-config main curves exist.

## Manifest Workflow

Generate and inspect the manifest before any GPU launch:

```bash
python3 experiments/scripts/build_iclr26_main_manifest.py \
  --output experiments/manifests/iclr26_main_manifest.csv \
  --print-summary
```

Launch a bounded manifest chunk:

```bash
CONFIRM_ICLR26_MANIFEST=1 \
MANIFEST=experiments/manifests/iclr26_main_manifest.csv \
ROW_START=0 \
ROW_LIMIT=1 \
sbatch experiments/scripts/run_iclr26_manifest_job.sh
```

The manifest generator verifies that every main cell has the required method rows and that AdamW and MatrixPolicy share the same outer `lr`, `min_lr`, and `weight_decay` config set.

## Method Sketch

RLB changes the Transformer MLP sublayer nonlinearity to grouped normalized rational functions:

```text
z = W_in x
z_g = group_g(z)
r_g = sqrt(mean(z_g^2) + eps)
u_g = z_g / r_g
h_g = r_g R_g(u_g)
y = W_out concat_g(h_g)
```

MatrixPolicy partitions backbone weights, rational coefficients, `W_in`, and `W_out`; applies role-aware matrix updates; uses group statistics in the original group-stat recipe; and applies a gauge rebalance. Future claims use matched configs and complete curves.
