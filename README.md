# RationalOPT

RationalOPT studies Rational Local Basis (RLB) variants inside causal Transformer language models and the `rational_matrix_policy_onpolicy` optimizer for pretraining.

Paper-facing results in this README include the completed E1 matched main suite and the completed E2 DCLM, FineWeb-Edu, FineWeb, Dolma-sample, and C4 M0/300M cells. WikiText is kept as a small demo anchor.

## Optimizer Variant Status

The completed `matrixpolicyV3` E1 rerun is rejected: it was slightly worse than original MatrixPolicy on every E1 dataset mean. The completed `matrixpolicyV4` E1 rerun is also rejected as a neutral result: it near-tied V1, but its functional-balance telemetry clipped to `+0.47` for 100% of recorded balance values, so the proposed mechanism mostly centered itself away. The active next proposal is `optimizer_design/proposals/matrixpolicyV5_joint_functional_metric.md`, a joint A/B functional-metric MatrixPolicy that changes the matrix update metric rather than adding engineering tweaks.

## Result Pointers

```text
experiments/ICLR_RUN_STATUS.md
experiments/results/iclr26_runtime_summary_2026_06_11/
experiments/results/iclr26_e2_dclm_2026_06_10/
experiments/results/iclr26_e2_fineweb_edu_2026_06_12/
experiments/results/iclr26_e2_fineweb_2026_06_15/
experiments/results/iclr26_e2_dolma_sample_2026_06_17/
experiments/results/iclr26_e2_c4_2026_06_19/
experiments/results/iclr26_e2_figures/
experiments/results/iclr26_e1_token_savings_2026_06_12/
experiments/results/iclr26_e1_figures/
experiments/runs/iclr26_main/        # local raw JSONL, ignored
experiments/results/rlb_matrix_policy_muon_switch_2026_05_28/  # WikiText demo anchor
```

## Completed Runtime Summary

Per optimizer/activation-combo runtimes for completed paper cells are tracked in `experiments/results/iclr26_runtime_summary_2026_06_11/`. The package covers cleaned E1 M0/100M rows plus completed E2 M0/300M DCLM, FineWeb-Edu, FineWeb, Dolma-sample, and C4 rows. It excludes E1 FineWeb-Edu seed `2027` rows `75-89` because Slurm job `158117` had `Restarts=6` and produced restart/node-contaminated throughput outliers. Rows `465+` are outside E2.

No raw all-completed E1 timing aggregate is tracked. The contaminated E1 rows are omitted from aggregate CSVs and `runtime_per_row.csv`.

The runtime metric is the JSONL `summary.total_seconds` training-harness wall time per manifest row. This excludes Slurm queue wait, dependency wait, token-cache construction, extension compilation, and launcher overhead.

## Current E2 300M Results

E2 M0/300M is complete for DCLM rows `240-284`, FineWeb-Edu rows `285-329`, FineWeb rows `330-374`, Dolma-sample rows `375-419`, and C4 rows `420-464`: each completed cell has three seeds, 15 fixed methods per seed, final eval at step `9150`, `32768` global tokens/step, and about `299.8M` train tokens per run. Rows `465+` are E3 and were not queued.

### DCLM

Tracked package: `experiments/results/iclr26_e2_dclm_2026_06_10/`.

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

Tracked package: `experiments/results/iclr26_e2_fineweb_edu_2026_06_12/`.

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

### FineWeb

Tracked package: `experiments/results/iclr26_e2_fineweb_2026_06_15/`.

MatrixPolicy is best on all three FineWeb E2 seeds. Mean final val loss is `3.965590 +/- 0.008530`; the next-best aggregate methods are `rlb_muon` at `4.001245 +/- 0.011375`, `rlb_lion` at `4.001381 +/- 0.012800`, `silu_lion` at `4.001499 +/- 0.008463`.

| Target loss | MP all-hit mean | Vs fastest non-MP: MP -> comparator (seeds) | Saved | Saved % | Vs SiLU+AdamW: MP -> AdamW (seeds) | Saved | Saved % |
| ---: | ---: | --- | ---: | ---: | --- | ---: | ---: |
| 4.40 | 83.6M | 83.6M -> 87.9M (3/3) | 4.4M | 5.0% | 83.6M -> 102.1M (3/3) | 18.6M | 18.2% |
| 4.30 | 112.5M | 112.5M -> 113.0M (3/3) | 0.5M | 0.5% | 112.5M -> 132.2M (3/3) | 19.7M | 14.9% |
| 4.20 | 143.6M | 143.6M -> 148.5M (3/3) | 4.9M | 3.3% | 143.6M -> 173.1M (3/3) | 29.5M | 17.0% |
| 4.10 | 187.3M | 187.3M -> 196.6M (3/3) | 9.3M | 4.7% | 187.3M -> 241.4M (3/3) | 54.1M | 22.4% |
| 4.05 | 214.6M | 214.6M -> 232.1M (3/3) | 17.5M | 7.5% | not reached (0/3) | not reached | n/a |
| 4.00 | 252.9M | 248.2M -> 285.9M (2/3) | 37.7M | 13.2% | not reached (0/3) | not reached | n/a |

### Dolma-sample

Tracked package: `experiments/results/iclr26_e2_dolma_sample_2026_06_17/`.

MatrixPolicy is best on all three Dolma-sample E2 seeds. Mean final val loss is `3.809853 +/- 0.005709`; the next-best aggregate methods are `rlb_lion` at `3.842503 +/- 0.009333`, `silu_lion` at `3.847523 +/- 0.009363`, `rlb_muon` at `3.848206 +/- 0.008937`. Both ADeMaMix variants diverged/non-finite on all three Dolma-sample seeds and are marked as such in the package.

| Target loss | MP all-hit mean | Vs fastest non-MP: MP -> comparator (seeds) | Saved | Saved % | Vs SiLU+AdamW: MP -> AdamW (seeds) | Saved | Saved % |
| ---: | ---: | --- | ---: | ---: | --- | ---: | ---: |
| 4.20 | 93.9M | 93.9M -> 94.5M (3/3) | 0.5M | 0.6% | 93.9M -> 110.9M (3/3) | 16.9M | 15.3% |
| 4.10 | 123.4M | 123.4M -> 124.0M (3/3) | 0.5M | 0.4% | 123.4M -> 145.3M (3/3) | 21.8M | 15.0% |
| 4.00 | 158.4M | 158.4M -> 163.8M (3/3) | 5.5M | 3.3% | 158.4M -> 195.0M (3/3) | 36.6M | 18.8% |
| 3.95 | 181.9M | 181.9M -> 190.6M (3/3) | 8.7M | 4.6% | 181.9M -> 232.7M (3/3) | 50.8M | 21.8% |
| 3.90 | 209.7M | 209.7M -> 223.9M (3/3) | 14.2M | 6.3% | 204.8M -> 283.4M (1/3) | 78.6M | 27.7% |
| 3.85 | 246.3M | 244.9M -> 276.9M (2/3) | 31.9M | 11.5% | not reached (0/3) | not reached | n/a |
| 3.82 | 282.4M | not reached (0/3) | not reached | n/a | not reached (0/3) | not reached | n/a |

### C4

Tracked package: `experiments/results/iclr26_e2_c4_2026_06_19/`.

MatrixPolicy is best on all three C4 E2 seeds. Mean final val loss is `3.882593 +/- 0.013925`; the next-best aggregate methods are `rlb_muon` at `3.915858 +/- 0.016066`, `rlb_lion` at `3.919576 +/- 0.014201`, `silu_lion` at `3.921326 +/- 0.010538`. Both ADeMaMix variants diverged/non-finite on all three C4 seeds and are marked as such in the package.

| Target loss | MP all-hit mean | Vs fastest non-MP: MP -> comparator (seeds) | Saved | Saved % | Vs SiLU+AdamW: MP -> AdamW (seeds) | Saved | Saved % |
| ---: | ---: | --- | ---: | ---: | --- | ---: | ---: |
| 4.40 | 67.7M | 67.7M -> 72.6M (3/3) | 4.9M | 6.8% | 67.7M -> 86.8M (3/3) | 19.1M | 22.0% |
| 4.30 | 90.7M | 90.7M -> 93.4M (3/3) | 2.7M | 2.9% | 90.7M -> 108.1M (3/3) | 17.5M | 16.2% |
| 4.20 | 118.5M | 118.5M -> 119.1M (3/3) | 0.5M | 0.5% | 118.5M -> 139.8M (3/3) | 21.3M | 15.2% |
| 4.10 | 151.3M | 151.3M -> 156.2M (3/3) | 4.9M | 3.1% | 151.3M -> 185.1M (3/3) | 33.9M | 18.3% |
| 4.05 | 170.9M | 170.9M -> 179.1M (3/3) | 8.2M | 4.6% | 170.9M -> 216.3M (3/3) | 45.3M | 21.0% |
| 4.00 | 196.1M | 196.1M -> 207.5M (3/3) | 11.5M | 5.5% | 196.1M -> 264.9M (3/3) | 68.8M | 26.0% |

Runtime tables are included in each E2 dataset package and in `experiments/results/iclr26_runtime_summary_2026_06_11/`.

### E2 Dense Curve Figures

The completed E2 curve package is tracked under `experiments/results/iclr26_e2_figures/`. The figures use every native JSONL log point from step 500 through 9150; shaded bands are mean +/- 1 sample std over three seeds.

#### DCLM

![DCLM E2 validation loss mean +/- std, all methods](experiments/results/iclr26_e2_figures/dclm_core_validation_loss_mean_std.svg)

![DCLM E2 validation PPL mean +/- std, all methods](experiments/results/iclr26_e2_figures/dclm_core_validation_ppl_mean_std.svg)

![DCLM E2 training loss mean +/- std, all methods](experiments/results/iclr26_e2_figures/dclm_core_training_loss_mean_std.svg)

![DCLM E2 validation loss mean +/- std, clean comparison](experiments/results/iclr26_e2_figures/dclm_clean_validation_loss_mean_std.svg)

![DCLM E2 validation PPL mean +/- std, clean comparison](experiments/results/iclr26_e2_figures/dclm_clean_validation_ppl_mean_std.svg)

![DCLM E2 training loss mean +/- std, clean comparison](experiments/results/iclr26_e2_figures/dclm_clean_training_loss_mean_std.svg)

#### FineWeb-Edu

![FineWeb-Edu E2 validation loss mean +/- std, all methods](experiments/results/iclr26_e2_figures/fineweb_edu_core_validation_loss_mean_std.svg)

![FineWeb-Edu E2 validation PPL mean +/- std, all methods](experiments/results/iclr26_e2_figures/fineweb_edu_core_validation_ppl_mean_std.svg)

![FineWeb-Edu E2 training loss mean +/- std, all methods](experiments/results/iclr26_e2_figures/fineweb_edu_core_training_loss_mean_std.svg)

![FineWeb-Edu E2 validation loss mean +/- std, clean comparison](experiments/results/iclr26_e2_figures/fineweb_edu_clean_validation_loss_mean_std.svg)

![FineWeb-Edu E2 validation PPL mean +/- std, clean comparison](experiments/results/iclr26_e2_figures/fineweb_edu_clean_validation_ppl_mean_std.svg)

![FineWeb-Edu E2 training loss mean +/- std, clean comparison](experiments/results/iclr26_e2_figures/fineweb_edu_clean_training_loss_mean_std.svg)

#### FineWeb

![FineWeb E2 validation loss mean +/- std, all methods](experiments/results/iclr26_e2_figures/fineweb_core_validation_loss_mean_std.svg)

![FineWeb E2 validation PPL mean +/- std, all methods](experiments/results/iclr26_e2_figures/fineweb_core_validation_ppl_mean_std.svg)

![FineWeb E2 training loss mean +/- std, all methods](experiments/results/iclr26_e2_figures/fineweb_core_training_loss_mean_std.svg)

![FineWeb E2 validation loss mean +/- std, clean comparison](experiments/results/iclr26_e2_figures/fineweb_clean_validation_loss_mean_std.svg)

![FineWeb E2 validation PPL mean +/- std, clean comparison](experiments/results/iclr26_e2_figures/fineweb_clean_validation_ppl_mean_std.svg)

![FineWeb E2 training loss mean +/- std, clean comparison](experiments/results/iclr26_e2_figures/fineweb_clean_training_loss_mean_std.svg)

#### Dolma-sample

![Dolma-sample E2 validation loss mean +/- std, all methods](experiments/results/iclr26_e2_figures/dolma_sample_core_validation_loss_mean_std.svg)

![Dolma-sample E2 validation PPL mean +/- std, all methods](experiments/results/iclr26_e2_figures/dolma_sample_core_validation_ppl_mean_std.svg)

![Dolma-sample E2 training loss mean +/- std, all methods](experiments/results/iclr26_e2_figures/dolma_sample_core_training_loss_mean_std.svg)

![Dolma-sample E2 validation loss mean +/- std, clean comparison](experiments/results/iclr26_e2_figures/dolma_sample_clean_validation_loss_mean_std.svg)

![Dolma-sample E2 validation PPL mean +/- std, clean comparison](experiments/results/iclr26_e2_figures/dolma_sample_clean_validation_ppl_mean_std.svg)

![Dolma-sample E2 training loss mean +/- std, clean comparison](experiments/results/iclr26_e2_figures/dolma_sample_clean_training_loss_mean_std.svg)

#### C4

![C4 E2 validation loss mean +/- std, all methods](experiments/results/iclr26_e2_figures/c4_en_core_validation_loss_mean_std.svg)

![C4 E2 validation PPL mean +/- std, all methods](experiments/results/iclr26_e2_figures/c4_en_core_validation_ppl_mean_std.svg)

![C4 E2 training loss mean +/- std, all methods](experiments/results/iclr26_e2_figures/c4_en_core_training_loss_mean_std.svg)

![C4 E2 validation loss mean +/- std, clean comparison](experiments/results/iclr26_e2_figures/c4_en_clean_validation_loss_mean_std.svg)

![C4 E2 validation PPL mean +/- std, clean comparison](experiments/results/iclr26_e2_figures/c4_en_clean_validation_ppl_mean_std.svg)

![C4 E2 training loss mean +/- std, clean comparison](experiments/results/iclr26_e2_figures/c4_en_clean_training_loss_mean_std.svg)

Full E2 per-method, per-seed, runtime, checkpoint, and curve tables are in the tracked E2 result package READMEs.

## Current E1 Main-Suite Results

Current E1 M0/100M results are tracked in `experiments/ICLR_RUN_STATUS.md` and figures are under `experiments/results/iclr26_e1_figures/`. E1 uses five corpora, three seeds, 15 matched methods per dataset/seed cell, 4 A6000 GPUs per job, and dense validation every 50 steps. E1 M0/100M is complete.

| Dataset | MatrixPolicy final val loss | next best current method | gap |
| --- | ---: | ---: | ---: |
| DCLM | 4.256224 +/- 0.004972 | rlb_lion 4.305728 +/- 0.005836 | 0.049505 |
| FineWeb-Edu | 4.088240 +/- 0.009434 | rlb_lion 4.142669 +/- 0.006812 | 0.054429 |
| FineWeb | 4.318581 +/- 0.010914 | rlb_lion 4.367062 +/- 0.007532 | 0.048481 |
| Dolma-sample | 4.323851 +/- 0.004565 | rlb_lion 4.369254 +/- 0.005561 | 0.045403 |
| C4 | 4.285119 +/- 0.020677 | rlb_lion 4.335663 +/- 0.020917 | 0.050544 |

### E1 Token-To-Target Savings

Full package: `experiments/results/iclr26_e1_token_savings_2026_06_12/`.

#### DCLM

| Target loss | MP all-hit mean | Vs fastest non-MP: MP -> comparator (seeds) | Saved | Saved % | Vs SiLU+AdamW: MP -> AdamW (seeds) | Saved | Saved % |
| ---: | ---: | --- | ---: | ---: | --- | ---: | ---: |
| 4.90 | 29.5M | 29.5M -> 32.2M (3/3) | 2.7M | 8.5% | 29.5M -> 36.0M (3/3) | 6.6M | 18.2% |
| 4.70 | 39.9M | 39.9M -> 42.1M (3/3) | 2.2M | 5.2% | 39.9M -> 48.6M (3/3) | 8.7M | 18.0% |
| 4.55 | 50.8M | 50.8M -> 53.5M (3/3) | 2.7M | 5.1% | 50.8M -> 63.4M (3/3) | 12.6M | 19.8% |
| 4.45 | 60.1M | 60.1M -> 64.4M (3/3) | 4.4M | 6.8% | 60.1M -> 82.5M (3/3) | 22.4M | 27.2% |
| 4.35 | 73.7M | 73.7M -> 83.0M (3/3) | 9.3M | 11.2% | not reached (0/3) | not reached | n/a |
| 4.30 | 84.1M | 83.6M -> 99.9M (1/3) | 16.4M | 16.4% | not reached (0/3) | not reached | n/a |

#### FineWeb-Edu

| Target loss | MP all-hit mean | Vs fastest non-MP: MP -> comparator (seeds) | Saved | Saved % | Vs SiLU+AdamW: MP -> AdamW (seeds) | Saved | Saved % |
| ---: | ---: | --- | ---: | ---: | --- | ---: | ---: |
| 4.80 | 30.0M | 30.0M -> 32.2M (3/3) | 2.2M | 6.8% | 30.0M -> 34.4M (3/3) | 4.4M | 12.7% |
| 4.60 | 38.2M | 38.2M -> 39.3M (3/3) | 1.1M | 2.8% | 38.2M -> 45.3M (3/3) | 7.1M | 15.7% |
| 4.40 | 49.7M | 49.7M -> 52.4M (3/3) | 2.7M | 5.2% | 49.7M -> 61.7M (3/3) | 12.0M | 19.5% |
| 4.30 | 59.0M | 59.0M -> 63.4M (3/3) | 4.4M | 6.9% | 59.0M -> 78.1M (3/3) | 19.1M | 24.5% |
| 4.20 | 71.5M | 71.5M -> 80.3M (3/3) | 8.7M | 10.9% | not reached (0/3) | not reached | n/a |
| 4.10 | 95.0M | not reached (0/3) | not reached | n/a | not reached (0/3) | not reached | n/a |

#### FineWeb

| Target loss | MP all-hit mean | Vs fastest non-MP: MP -> comparator (seeds) | Saved | Saved % | Vs SiLU+AdamW: MP -> AdamW (seeds) | Saved | Saved % |
| ---: | ---: | --- | ---: | ---: | --- | ---: | ---: |
| 5.00 | 30.6M | 30.6M -> 32.2M (3/3) | 1.6M | 5.1% | 30.6M -> 35.0M (3/3) | 4.4M | 12.5% |
| 4.80 | 39.3M | 39.3M -> 40.4M (3/3) | 1.1M | 2.7% | 39.3M -> 47.0M (3/3) | 7.6M | 16.3% |
| 4.60 | 51.9M | 51.9M -> 55.2M (3/3) | 3.3M | 5.9% | 51.9M -> 67.2M (3/3) | 15.3M | 22.8% |
| 4.50 | 62.3M | 62.3M -> 66.6M (3/3) | 4.4M | 6.6% | 62.3M -> 89.0M (3/3) | 26.8M | 30.1% |
| 4.40 | 77.0M | 77.0M -> 86.3M (3/3) | 9.3M | 10.8% | not reached (0/3) | not reached | n/a |
| 4.35 | 88.5M | not reached (0/3) | not reached | n/a | not reached (0/3) | not reached | n/a |

#### Dolma-sample

| Target loss | MP all-hit mean | Vs fastest non-MP: MP -> comparator (seeds) | Saved | Saved % | Vs SiLU+AdamW: MP -> AdamW (seeds) | Saved | Saved % |
| ---: | ---: | --- | ---: | ---: | --- | ---: | ---: |
| 5.00 | 31.7M | 31.7M -> 33.3M (3/3) | 1.6M | 4.9% | 31.7M -> 36.6M (3/3) | 4.9M | 13.4% |
| 4.80 | 40.4M | 40.4M -> 41.5M (3/3) | 1.1M | 2.6% | 40.4M -> 48.1M (3/3) | 7.6M | 15.9% |
| 4.60 | 54.1M | 54.1M -> 56.3M (3/3) | 2.2M | 3.9% | 54.1M -> 69.4M (3/3) | 15.3M | 22.0% |
| 4.50 | 63.9M | 63.9M -> 67.2M (3/3) | 3.3M | 4.9% | 63.9M -> 92.8M (3/3) | 28.9M | 31.2% |
| 4.40 | 77.6M | 77.6M -> 87.4M (3/3) | 9.8M | 11.2% | not reached (0/3) | not reached | n/a |
| 4.35 | 90.1M | not reached (0/3) | not reached | n/a | not reached (0/3) | not reached | n/a |

#### C4

| Target loss | MP all-hit mean | Vs fastest non-MP: MP -> comparator (seeds) | Saved | Saved % | Vs SiLU+AdamW: MP -> AdamW (seeds) | Saved | Saved % |
| ---: | ---: | --- | ---: | ---: | --- | ---: | ---: |
| 5.00 | 29.5M | 29.5M -> 31.1M (3/3) | 1.6M | 5.3% | 29.5M -> 34.4M (3/3) | 4.9M | 14.3% |
| 4.80 | 37.7M | 37.7M -> 38.8M (3/3) | 1.1M | 2.8% | 37.7M -> 45.3M (3/3) | 7.6M | 16.9% |
| 4.60 | 49.7M | 49.7M -> 51.9M (3/3) | 2.2M | 4.2% | 49.7M -> 63.4M (3/3) | 13.7M | 21.6% |
| 4.50 | 58.4M | 58.4M -> 61.7M (3/3) | 3.3M | 5.3% | 58.4M -> 80.3M (3/3) | 21.8M | 27.2% |
| 4.40 | 71.0M | 71.0M -> 78.1M (3/3) | 7.1M | 9.1% | not reached (0/3) | not reached | n/a |
| 4.30 | not reached | not reached (0/3) | not reached | n/a | not reached (0/3) | not reached | n/a |

### E1 Dense Curve Figures

These are the same E1 figure panels embedded in `experiments/ICLR_RUN_STATUS.md`. The curves use completed E1 runs at native logging cadence: validation every 50 steps and training loss every 10 steps. Shaded bands are mean +/- 1 sample std over seeds. The all-method view includes MatrixPolicy, AdamW, Lion, SOAP, Muon, ScheduleFree, and CAME rows; the clean view omits SOAP from the plotted comparison.

#### DCLM

![DCLM E1 validation loss mean +/- std, all methods](experiments/results/iclr26_e1_figures/dclm_core_validation_loss_mean_std.svg)

![DCLM E1 validation PPL mean +/- std, all methods](experiments/results/iclr26_e1_figures/dclm_core_validation_ppl_mean_std.svg)

![DCLM E1 training loss mean +/- std, all methods](experiments/results/iclr26_e1_figures/dclm_core_training_loss_mean_std.svg)

![DCLM E1 validation loss mean +/- std, clean comparison](experiments/results/iclr26_e1_figures/dclm_clean_validation_loss_mean_std.svg)

![DCLM E1 validation PPL mean +/- std, clean comparison](experiments/results/iclr26_e1_figures/dclm_clean_validation_ppl_mean_std.svg)

![DCLM E1 training loss mean +/- std, clean comparison](experiments/results/iclr26_e1_figures/dclm_clean_training_loss_mean_std.svg)

#### FineWeb-Edu

![FineWeb-Edu E1 validation loss mean +/- std, all methods](experiments/results/iclr26_e1_figures/fineweb_edu_core_validation_loss_mean_std.svg)

![FineWeb-Edu E1 validation PPL mean +/- std, all methods](experiments/results/iclr26_e1_figures/fineweb_edu_core_validation_ppl_mean_std.svg)

![FineWeb-Edu E1 training loss mean +/- std, all methods](experiments/results/iclr26_e1_figures/fineweb_edu_core_training_loss_mean_std.svg)

![FineWeb-Edu E1 validation loss mean +/- std, clean comparison](experiments/results/iclr26_e1_figures/fineweb_edu_clean_validation_loss_mean_std.svg)

![FineWeb-Edu E1 validation PPL mean +/- std, clean comparison](experiments/results/iclr26_e1_figures/fineweb_edu_clean_validation_ppl_mean_std.svg)

![FineWeb-Edu E1 training loss mean +/- std, clean comparison](experiments/results/iclr26_e1_figures/fineweb_edu_clean_training_loss_mean_std.svg)

#### FineWeb

![FineWeb E1 validation loss mean +/- std, all methods](experiments/results/iclr26_e1_figures/fineweb_core_validation_loss_mean_std.svg)

![FineWeb E1 validation PPL mean +/- std, all methods](experiments/results/iclr26_e1_figures/fineweb_core_validation_ppl_mean_std.svg)

![FineWeb E1 training loss mean +/- std, all methods](experiments/results/iclr26_e1_figures/fineweb_core_training_loss_mean_std.svg)

![FineWeb E1 validation loss mean +/- std, clean comparison](experiments/results/iclr26_e1_figures/fineweb_clean_validation_loss_mean_std.svg)

![FineWeb E1 validation PPL mean +/- std, clean comparison](experiments/results/iclr26_e1_figures/fineweb_clean_validation_ppl_mean_std.svg)

![FineWeb E1 training loss mean +/- std, clean comparison](experiments/results/iclr26_e1_figures/fineweb_clean_training_loss_mean_std.svg)

#### Dolma-sample

![Dolma-sample E1 validation loss mean +/- std, all methods](experiments/results/iclr26_e1_figures/dolma_sample_core_validation_loss_mean_std.svg)

![Dolma-sample E1 validation PPL mean +/- std, all methods](experiments/results/iclr26_e1_figures/dolma_sample_core_validation_ppl_mean_std.svg)

![Dolma-sample E1 training loss mean +/- std, all methods](experiments/results/iclr26_e1_figures/dolma_sample_core_training_loss_mean_std.svg)

![Dolma-sample E1 validation loss mean +/- std, clean comparison](experiments/results/iclr26_e1_figures/dolma_sample_clean_validation_loss_mean_std.svg)

![Dolma-sample E1 validation PPL mean +/- std, clean comparison](experiments/results/iclr26_e1_figures/dolma_sample_clean_validation_ppl_mean_std.svg)

![Dolma-sample E1 training loss mean +/- std, clean comparison](experiments/results/iclr26_e1_figures/dolma_sample_clean_training_loss_mean_std.svg)

#### C4

![C4 E1 validation loss mean +/- std, all methods](experiments/results/iclr26_e1_figures/c4_en_core_validation_loss_mean_std.svg)

![C4 E1 validation PPL mean +/- std, all methods](experiments/results/iclr26_e1_figures/c4_en_core_validation_ppl_mean_std.svg)

![C4 E1 training loss mean +/- std, all methods](experiments/results/iclr26_e1_figures/c4_en_core_training_loss_mean_std.svg)

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

## Result Regeneration

Regenerate paper-facing completed-cell tables and figures from raw JSONL with:

```bash
python3 experiments/scripts/plot_iclr26_e1_curves.py --status-md experiments/ICLR_RUN_STATUS.md
python3 experiments/scripts/summarize_iclr26_e1_token_savings.py
python3 experiments/scripts/summarize_iclr26_e2_dataset.py --dataset dclm --output-dir experiments/results/iclr26_e2_dclm_2026_06_10 --completed-date 2026-06-10
python3 experiments/scripts/summarize_iclr26_e2_dataset.py --dataset fineweb_edu --output-dir experiments/results/iclr26_e2_fineweb_edu_2026_06_12 --completed-date 2026-06-12
python3 experiments/scripts/summarize_iclr26_e2_dataset.py --dataset fineweb --output-dir experiments/results/iclr26_e2_fineweb_2026_06_15 --completed-date 2026-06-15
python3 experiments/scripts/summarize_iclr26_e2_dataset.py --dataset dolma_sample --output-dir experiments/results/iclr26_e2_dolma_sample_2026_06_17 --completed-date 2026-06-17
python3 experiments/scripts/summarize_iclr26_e2_dataset.py --dataset c4_en --output-dir experiments/results/iclr26_e2_c4_2026_06_19 --completed-date 2026-06-19
python3 experiments/scripts/plot_iclr26_e2_curves.py
python3 experiments/scripts/summarize_iclr26_runtimes.py
```

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
