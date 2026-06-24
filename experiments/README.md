# Experiments

This directory keeps the completed E1 matched main-suite results, completed E2 DCLM, FineWeb-Edu, FineWeb, Dolma-sample, and C4 M0/300M result packages, and the WikiText demo anchor. Paper runs follow the manifest workflow below, and exact submitted commands are recorded in `ICLR_RUN_COMMANDS.md`.

All E1/E2 MatrixPolicy tables, curves, token-savings readouts, and runtime summaries use the accepted safe-speed MatrixPolicy replacement rows.

## Optimizer Variant Status

Rejected MatrixPolicy proposal artifacts have been pruned from the live repo and raw run tree; the single retained negative-result state is `../optimizer_design/proposals/matrixpolicy_variant_failures.md`. Original `rational_matrix_policy_onpolicy` remains the paper anchor and the only active MatrixPolicy optimizer. Its method-preserving safe Muon-off implementation is now the paper-facing MatrixPolicy implementation for E1 and E2: E1 has a complete 15-row safe-speed MatrixPolicy replacement, and every E2 dataset has a complete three-seed safe-speed MatrixPolicy replacement.

## Result Pointers

```text
ICLR_RUN_STATUS.md
results/iclr26_runtime_summary_2026_06_11/
results/iclr26_e1_token_savings_2026_06_12/
results/iclr26_e1_figures/
results/iclr26_e2_dclm_2026_06_10/
results/iclr26_e2_fineweb_edu_2026_06_12/
results/iclr26_e2_fineweb_2026_06_15/
results/iclr26_e2_dolma_sample_2026_06_17/
results/iclr26_e2_c4_2026_06_19/
results/iclr26_e2_figures/
runs/iclr26_main/        # local raw JSONL, ignored
results/rlb_matrix_policy_muon_switch_2026_05_28/  # WikiText demo anchor
```

## Runtime Accounting

Generated: 2026-06-24.

This package summarizes clean per optimizer/activation-combo runtime from completed JSONL `summary` records. The runtime field is `summary.total_seconds`, i.e. training-harness wall time for a manifest row. It excludes Slurm queue wait, dependency wait, token-cache construction, extension compilation, launcher overhead, and pre-restart partial attempts.

Included in tracked runtime aggregates:

- E1 M0/100M clean rows: `225` rows. E1 FineWeb-Edu seed `2027` job `158117` had `Restarts=6`; rows `75-80` are retained because their completed JSONL timings match adjacent seeds. Original rows `81-88` are skipped because the existing artifacts cannot reconstruct trusted per-row runtime after multiple preempted allocations and partial JSONLs. Completed clean repair overlay rows for E1 FineWeb-Edu seed `2027` rows `81-88`: `8/8`. Row `89` is replaced by the completed safe-speed MatrixPolicy rerun when available.
- E2 M0/300M DCLM completed cell: `45` rows, one dataset x three seeds x 15 methods.
- E2 M0/300M FineWeb-Edu completed cell: `45` rows, one dataset x three seeds x 15 methods.
- E2 M0/300M FineWeb completed cell: `45` rows, one dataset x three seeds x 15 methods.
- E2 M0/300M Dolma-sample completed cell: `45` rows, one dataset x three seeds x 15 methods.
- E2 M0/300M C4 completed cell: `45` rows, one dataset x three seeds x 15 methods.

Excluded from tracked runtime aggregates:

- Original E1 FineWeb-Edu seed `2027` rows `81-88`: `8` rows skipped from the main manifest runtime source. They are overlaid from the completed clean repair manifest `manifests/iclr26_e1_fineweb_edu_seed2027_runtime_repair_manifest.csv`.
- Rows `465+` are outside E2.

No raw Slurm-elapsed E1 aggregate is tracked in this package. Runtime aggregates use completed JSONL `summary.total_seconds` only for clean row attempts; original restart-contaminated rows `81-88` are not assigned inferred row times.

Clean rows summarized: `450`.

## Current E1 M0/100M Results

E1 M0/100M is complete across DCLM, FineWeb-Edu, FineWeb, Dolma-sample, and C4 with three seeds per dataset, 15 matched methods per dataset/seed cell, dense validation every 50 steps, and final eval at step `3050`. MatrixPolicy rows below use `manifests/iclr26_matrixpolicy_safe_speed_e1_manifest.csv` and raw JSONL from `runs/iclr26_main/E1_matrixpolicy_safe_speed_100m/`; comparator rows use the clean main-manifest rows plus the completed FineWeb-Edu seed-2027 runtime repair overlay where applicable.

### E1 Runtime Table

| Combo | Runs | Mean runtime | Std | Range | Mean s/step | Mean tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RLB+MatrixPolicy | 15 | 27.3 min | 6.0 min | 22.1 min-37.2 min | 0.5102 | 67078.3 |
| SiLU+AdamW | 15 | 27.6 min | 5.3 min | 18.2 min-33.7 min | 0.5248 | 65212.8 |
| RLB+AdamW | 15 | 32.2 min | 5.1 min | 23.4 min-38.5 min | 0.6032 | 55854.1 |
| SiLU+Muon | 15 | 29.4 min | 5.4 min | 20.1 min-36.3 min | 0.5604 | 60754.1 |
| RLB+Muon | 15 | 33.9 min | 5.0 min | 25.1 min-40.4 min | 0.6349 | 52890.2 |
| SiLU+Lion | 15 | 27.9 min | 6.2 min | 18.1 min-40.1 min | 0.5304 | 65164.9 |
| RLB+Lion | 15 | 32.5 min | 5.8 min | 23.3 min-44.3 min | 0.6091 | 55659.8 |
| SiLU+SOAP | 15 | 30.5 min | 5.9 min | 21.4 min-38.8 min | 0.5802 | 58784.6 |
| RLB+SOAP | 15 | 32.0 min | 5.3 min | 24.0 min-40.0 min | 0.5986 | 56363.9 |
| SiLU+ADeMaMix | 15 | 26.9 min | 5.8 min | 18.0 min-35.1 min | 0.5119 | 67431.3 |
| RLB+ADeMaMix | 15 | 33.4 min | 5.2 min | 26.2 min-40.3 min | 0.5958 | 56664.4 |
| SiLU+CAME | 15 | 28.3 min | 5.8 min | 19.3 min-36.3 min | 0.5382 | 63804.1 |
| RLB+CAME | 15 | 33.4 min | 5.1 min | 25.4 min-39.9 min | 0.6254 | 53724.4 |
| SiLU+ScheduleFree | 15 | 26.7 min | 5.8 min | 17.8 min-33.9 min | 0.5072 | 68064.2 |
| RLB+ScheduleFree | 15 | 31.6 min | 5.4 min | 23.0 min-38.3 min | 0.5921 | 57193.3 |

### E1 Token-To-Target Savings

Generated from completed E1 M0/100M JSONL eval records. All rows still trained to the fixed budget of about `99.9M` tokens; this is an early-stop/speed-to-target readout only.

Each row uses `32768` global tokens/step and the native E1 eval cadence of 50 steps, or `1.64M` tokens per readout interval.

`Second-best` means the fastest non-MatrixPolicy method to reach the target within the same seed. `AdamW` means the standard `silu_adamw` row. Savings and proportions are computed only on seeds where both MatrixPolicy and the comparator reached the target.

#### DCLM

| Target loss | MP all-hit mean | Vs fastest non-MP: MP -> comparator (seeds) | Saved | Saved % | Vs SiLU+AdamW: MP -> AdamW (seeds) | Saved | Saved % |
| ---: | ---: | --- | ---: | ---: | --- | ---: | ---: |
| 4.90 | 30.0M | 30.0M -> 32.2M (3/3) | 2.2M | 6.8% | 30.0M -> 36.0M (3/3) | 6.0M | 16.7% |
| 4.70 | 39.9M | 39.9M -> 42.1M (3/3) | 2.2M | 5.2% | 39.9M -> 48.6M (3/3) | 8.7M | 18.0% |
| 4.55 | 50.2M | 50.2M -> 53.5M (3/3) | 3.3M | 6.1% | 50.2M -> 63.4M (3/3) | 13.1M | 20.7% |
| 4.45 | 60.6M | 60.6M -> 64.4M (3/3) | 3.8M | 5.9% | 60.6M -> 82.5M (3/3) | 21.8M | 26.5% |
| 4.35 | 73.7M | 73.7M -> 83.0M (3/3) | 9.3M | 11.2% | not reached (0/3) | not reached | n/a |
| 4.30 | 84.7M | 85.2M -> 99.9M (1/3) | 14.7M | 14.8% | not reached (0/3) | not reached | n/a |

#### FineWeb-Edu

| Target loss | MP all-hit mean | Vs fastest non-MP: MP -> comparator (seeds) | Saved | Saved % | Vs SiLU+AdamW: MP -> AdamW (seeds) | Saved | Saved % |
| ---: | ---: | --- | ---: | ---: | --- | ---: | ---: |
| 4.80 | 30.0M | 30.0M -> 32.2M (3/3) | 2.2M | 6.8% | 30.0M -> 34.4M (3/3) | 4.4M | 12.7% |
| 4.60 | 38.2M | 38.2M -> 39.3M (3/3) | 1.1M | 2.8% | 38.2M -> 45.3M (3/3) | 7.1M | 15.7% |
| 4.40 | 49.7M | 49.7M -> 52.4M (3/3) | 2.7M | 5.2% | 49.7M -> 61.7M (3/3) | 12.0M | 19.5% |
| 4.30 | 58.4M | 58.4M -> 63.4M (3/3) | 4.9M | 7.8% | 58.4M -> 78.1M (3/3) | 19.7M | 25.2% |
| 4.20 | 71.5M | 71.5M -> 80.3M (3/3) | 8.7M | 10.9% | not reached (0/3) | not reached | n/a |
| 4.10 | 95.0M | not reached (0/3) | not reached | n/a | not reached (0/3) | not reached | n/a |

#### FineWeb

| Target loss | MP all-hit mean | Vs fastest non-MP: MP -> comparator (seeds) | Saved | Saved % | Vs SiLU+AdamW: MP -> AdamW (seeds) | Saved | Saved % |
| ---: | ---: | --- | ---: | ---: | --- | ---: | ---: |
| 5.00 | 30.6M | 30.6M -> 32.2M (3/3) | 1.6M | 5.1% | 30.6M -> 35.0M (3/3) | 4.4M | 12.5% |
| 4.80 | 38.8M | 38.8M -> 40.4M (3/3) | 1.6M | 4.1% | 38.8M -> 47.0M (3/3) | 8.2M | 17.4% |
| 4.60 | 51.9M | 51.9M -> 55.2M (3/3) | 3.3M | 5.9% | 51.9M -> 67.2M (3/3) | 15.3M | 22.8% |
| 4.50 | 62.3M | 62.3M -> 66.6M (3/3) | 4.4M | 6.6% | 62.3M -> 89.0M (3/3) | 26.8M | 30.1% |
| 4.40 | 77.0M | 77.0M -> 86.3M (3/3) | 9.3M | 10.8% | not reached (0/3) | not reached | n/a |
| 4.35 | 88.5M | not reached (0/3) | not reached | n/a | not reached (0/3) | not reached | n/a |

#### Dolma-sample

| Target loss | MP all-hit mean | Vs fastest non-MP: MP -> comparator (seeds) | Saved | Saved % | Vs SiLU+AdamW: MP -> AdamW (seeds) | Saved | Saved % |
| ---: | ---: | --- | ---: | ---: | --- | ---: | ---: |
| 5.00 | 31.7M | 31.7M -> 33.3M (3/3) | 1.6M | 4.9% | 31.7M -> 36.6M (3/3) | 4.9M | 13.4% |
| 4.80 | 39.9M | 39.9M -> 41.5M (3/3) | 1.6M | 3.9% | 39.9M -> 48.1M (3/3) | 8.2M | 17.0% |
| 4.60 | 53.5M | 53.5M -> 56.3M (3/3) | 2.7M | 4.9% | 53.5M -> 69.4M (3/3) | 15.8M | 22.8% |
| 4.50 | 63.4M | 63.4M -> 67.2M (3/3) | 3.8M | 5.7% | 63.4M -> 92.8M (3/3) | 29.5M | 31.8% |
| 4.40 | 77.6M | 77.6M -> 87.4M (3/3) | 9.8M | 11.2% | not reached (0/3) | not reached | n/a |
| 4.35 | 89.6M | not reached (0/3) | not reached | n/a | not reached (0/3) | not reached | n/a |

#### C4

| Target loss | MP all-hit mean | Vs fastest non-MP: MP -> comparator (seeds) | Saved | Saved % | Vs SiLU+AdamW: MP -> AdamW (seeds) | Saved | Saved % |
| ---: | ---: | --- | ---: | ---: | --- | ---: | ---: |
| 5.00 | 29.5M | 29.5M -> 31.1M (3/3) | 1.6M | 5.3% | 29.5M -> 34.4M (3/3) | 4.9M | 14.3% |
| 4.80 | 37.7M | 37.7M -> 38.8M (3/3) | 1.1M | 2.8% | 37.7M -> 45.3M (3/3) | 7.6M | 16.9% |
| 4.60 | 49.7M | 49.7M -> 51.9M (3/3) | 2.2M | 4.2% | 49.7M -> 63.4M (3/3) | 13.7M | 21.6% |
| 4.50 | 58.4M | 58.4M -> 61.7M (3/3) | 3.3M | 5.3% | 58.4M -> 80.3M (3/3) | 21.8M | 27.2% |
| 4.40 | 71.5M | 71.5M -> 78.1M (3/3) | 6.6M | 8.4% | not reached (0/3) | not reached | n/a |
| 4.30 | not reached | not reached (0/3) | not reached | n/a | not reached (0/3) | not reached | n/a |

#### Files

- `token_savings.csv`: aggregate token-to-target savings by dataset and target.
- `token_savings_per_seed.csv`: per-seed threshold hits and comparator identities.

### E1 Dense Curves And Checkpoint Tables

Completed E1 M0/100M datasets: DCLM, FineWeb-Edu, FineWeb, Dolma-sample, and C4. Figures use every native JSONL log point from step 500 through 3050. Validation curves use every 50-step eval; training-loss curves use every 10-step train log. Shaded bands are mean +/- 1 sample std over three seeds.

MatrixPolicy curves and tables use the accepted E1 safe-speed replacement JSONL rows when the generator is called with `--matrixpolicy-manifest`; all other methods use the clean main E1 rows plus the completed FineWeb-Edu seed-2027 runtime repair overlay where applicable.

Final validation-loss overview across completed E1 datasets. Lower is better; cells are mean +/- sample std over three seeds.

| Method | DCLM final | FineWeb-Edu final | FineWeb final | Dolma-sample final | C4 final |
| --- | ---: | ---: | ---: | ---: | ---: |
| MatrixPolicy | 4.2570 +/- 0.0042 | 4.0883 +/- 0.0092 | 4.3195 +/- 0.0124 | 4.3239 +/- 0.0052 | 4.2864 +/- 0.0193 |
| RLB+AdamW | 4.4047 +/- 0.0046 | 4.2380 +/- 0.0061 | 4.4705 +/- 0.0133 | 4.4881 +/- 0.0009 | 4.4426 +/- 0.0198 |
| SiLU+AdamW | 4.4056 +/- 0.0099 | 4.2375 +/- 0.0086 | 4.4758 +/- 0.0097 | 4.4862 +/- 0.0012 | 4.4469 +/- 0.0160 |
| RLB+Lion | 4.3057 +/- 0.0058 | 4.1427 +/- 0.0068 | 4.3671 +/- 0.0075 | 4.3693 +/- 0.0056 | 4.3357 +/- 0.0209 |
| SiLU+Lion | 4.3183 +/- 0.0069 | 4.1494 +/- 0.0092 | 4.3825 +/- 0.0083 | 4.3878 +/- 0.0046 | 4.3536 +/- 0.0156 |
| RLB+SOAP | 4.4351 +/- 0.0217 | 4.2623 +/- 0.0131 | 4.4850 +/- 0.0255 | 4.5029 +/- 0.0157 | 4.4572 +/- 0.0104 |
| SiLU+SOAP | 4.4160 +/- 0.0038 | 4.2630 +/- 0.0220 | 4.4840 +/- 0.0112 | 4.4987 +/- 0.0035 | 4.4582 +/- 0.0146 |
| RLB+Muon | 4.4742 +/- 0.0041 | 4.2877 +/- 0.0199 | 4.5216 +/- 0.0120 | 4.5586 +/- 0.0093 | 4.4945 +/- 0.0220 |
| SiLU+Muon | 4.4572 +/- 0.0126 | 4.2787 +/- 0.0243 | 4.5163 +/- 0.0264 | 4.5443 +/- 0.0108 | 4.4824 +/- 0.0233 |
| RLB+ScheduleFree | 4.8781 +/- 0.0055 | 4.7797 +/- 0.0113 | 4.9877 +/- 0.0196 | 5.0241 +/- 0.0136 | 4.9701 +/- 0.0149 |
| SiLU+ScheduleFree | 4.9023 +/- 0.0111 | 4.8258 +/- 0.0072 | 5.0142 +/- 0.0188 | 5.0494 +/- 0.0149 | 5.0064 +/- 0.0160 |
| RLB+CAME | 5.0074 +/- 0.0082 | 4.9043 +/- 0.0046 | 5.1251 +/- 0.0173 | 5.1425 +/- 0.0164 | 5.1000 +/- 0.0153 |
| SiLU+CAME | 5.0107 +/- 0.0143 | 4.9207 +/- 0.0123 | 5.1325 +/- 0.0126 | 5.1650 +/- 0.0189 | 5.1230 +/- 0.0178 |
| RLB+ADeMaMix | 246105152.0000 +/- 0.0000 (n=1) | 7880.5115 +/- 12045.6569 | 3022914304.0000 +/- 0.0000 (n=1) | 1775543041.7179 +/- 2510996321.2383 (n=2) | 449788003.0859 +/- 779025043.4338 |
| SiLU+ADeMaMix | 48.6455 +/- 13.7255 | 242.8537 +/- 242.0122 | 51.9965 +/- 15.1322 | 28427.1854 +/- 40987.7167 | 344.9990 +/- 372.6250 |

#### DCLM

All-method view:

![DCLM E1 validation loss mean +/- std, all methods](results/iclr26_e1_figures/dclm_core_validation_loss_mean_std.svg)

![DCLM E1 validation PPL mean +/- std, all methods](results/iclr26_e1_figures/dclm_core_validation_ppl_mean_std.svg)

![DCLM E1 training loss mean +/- std, all methods](results/iclr26_e1_figures/dclm_core_training_loss_mean_std.svg)

Clean comparison view:

![DCLM E1 validation loss mean +/- std, clean comparison](results/iclr26_e1_figures/dclm_clean_validation_loss_mean_std.svg)

![DCLM E1 validation PPL mean +/- std, clean comparison](results/iclr26_e1_figures/dclm_clean_validation_ppl_mean_std.svg)

![DCLM E1 training loss mean +/- std, clean comparison](results/iclr26_e1_figures/dclm_clean_training_loss_mean_std.svg)

DCLM validation-loss checkpoint table, mean +/- sample std:

| Method | 500 | 1000 | 1500 | 2000 | 2500 | 3050 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MatrixPolicy | 5.2562 +/- 0.0052 | 4.8267 +/- 0.0078 | 4.5532 +/- 0.0045 | 4.3985 +/- 0.0005 | 4.3073 +/- 0.0029 | 4.2570 +/- 0.0042 |
| RLB+AdamW | 5.3672 +/- 0.0076 | 4.9292 +/- 0.0045 | 4.6748 +/- 0.0011 | 4.5270 +/- 0.0048 | 4.4464 +/- 0.0037 | 4.4047 +/- 0.0046 |
| SiLU+AdamW | 5.3838 +/- 0.0115 | 4.9398 +/- 0.0091 | 4.6788 +/- 0.0083 | 4.5306 +/- 0.0101 | 4.4489 +/- 0.0086 | 4.4056 +/- 0.0099 |
| RLB+Lion | 5.4021 +/- 0.0175 | 4.8664 +/- 0.0063 | 4.5860 +/- 0.0085 | 4.4348 +/- 0.0049 | 4.3511 +/- 0.0078 | 4.3057 +/- 0.0058 |
| SiLU+Lion | 5.4593 +/- 0.0049 | 4.8989 +/- 0.0070 | 4.6081 +/- 0.0038 | 4.4516 +/- 0.0063 | 4.3649 +/- 0.0049 | 4.3183 +/- 0.0069 |
| RLB+SOAP | 5.4086 +/- 0.0216 | 4.9523 +/- 0.0162 | 4.7242 +/- 0.0457 | 4.5516 +/- 0.0158 | 4.4597 +/- 0.0155 | 4.4351 +/- 0.0217 |
| SiLU+SOAP | 5.7233 +/- 0.1468 | 5.0899 +/- 0.0598 | 4.8541 +/- 0.1178 | 4.5941 +/- 0.0321 | 4.4658 +/- 0.0017 | 4.4160 +/- 0.0038 |
| RLB+Muon | 5.7691 +/- 0.0133 | 5.0865 +/- 0.0118 | 4.7917 +/- 0.0057 | 4.6209 +/- 0.0014 | 4.5246 +/- 0.0042 | 4.4742 +/- 0.0041 |
| SiLU+Muon | 5.8027 +/- 0.0123 | 5.0927 +/- 0.0102 | 4.7831 +/- 0.0090 | 4.6036 +/- 0.0127 | 4.5065 +/- 0.0120 | 4.4572 +/- 0.0126 |
| RLB+ScheduleFree | 5.8580 +/- 0.0045 | 5.3702 +/- 0.0099 | 5.1395 +/- 0.0085 | 5.0064 +/- 0.0073 | 4.9281 +/- 0.0061 | 4.8781 +/- 0.0055 |
| SiLU+ScheduleFree | 5.8883 +/- 0.0112 | 5.3887 +/- 0.0131 | 5.1590 +/- 0.0136 | 5.0276 +/- 0.0134 | 4.9514 +/- 0.0116 | 4.9023 +/- 0.0111 |
| RLB+CAME | 5.8999 +/- 0.0064 | 5.4569 +/- 0.0106 | 5.2388 +/- 0.0114 | 5.1140 +/- 0.0101 | 5.0445 +/- 0.0084 | 5.0074 +/- 0.0082 |
| SiLU+CAME | 5.9441 +/- 0.0091 | 5.4677 +/- 0.0067 | 5.2453 +/- 0.0124 | 5.1177 +/- 0.0147 | 5.0477 +/- 0.0139 | 5.0107 +/- 0.0143 |
| RLB+ADeMaMix | 16.1630 +/- 5.4229 | 370.9679 +/- 490.7603 | 43330230.6667 +/- 59088752.2823 | 175540490.6667 +/- 74840716.6504 | 3782729864.0000 +/- 5062818831.9630 (n=2) | 246105152.0000 +/- 0.0000 (n=1) |
| SiLU+ADeMaMix | 6.9823 +/- 0.6147 | 247.6634 +/- 51.2342 | 78.1737 +/- 22.2689 | 62.7083 +/- 16.6845 | 53.5849 +/- 14.3896 | 48.6455 +/- 13.7255 |

#### FineWeb-Edu

All-method view:

![FineWeb-Edu E1 validation loss mean +/- std, all methods](results/iclr26_e1_figures/fineweb_edu_core_validation_loss_mean_std.svg)

![FineWeb-Edu E1 validation PPL mean +/- std, all methods](results/iclr26_e1_figures/fineweb_edu_core_validation_ppl_mean_std.svg)

![FineWeb-Edu E1 training loss mean +/- std, all methods](results/iclr26_e1_figures/fineweb_edu_core_training_loss_mean_std.svg)

Clean comparison view:

![FineWeb-Edu E1 validation loss mean +/- std, clean comparison](results/iclr26_e1_figures/fineweb_edu_clean_validation_loss_mean_std.svg)

![FineWeb-Edu E1 validation PPL mean +/- std, clean comparison](results/iclr26_e1_figures/fineweb_edu_clean_validation_ppl_mean_std.svg)

![FineWeb-Edu E1 training loss mean +/- std, clean comparison](results/iclr26_e1_figures/fineweb_edu_clean_training_loss_mean_std.svg)

FineWeb-Edu validation-loss checkpoint table, mean +/- sample std:

| Method | 500 | 1000 | 1500 | 2000 | 2500 | 3050 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MatrixPolicy | 5.2424 +/- 0.0162 | 4.7106 +/- 0.0106 | 4.3975 +/- 0.0112 | 4.2365 +/- 0.0100 | 4.1393 +/- 0.0089 | 4.0883 +/- 0.0092 |
| RLB+AdamW | 5.3781 +/- 0.0151 | 4.8164 +/- 0.0076 | 4.5191 +/- 0.0025 | 4.3651 +/- 0.0057 | 4.2795 +/- 0.0055 | 4.2380 +/- 0.0061 |
| SiLU+AdamW | 5.4084 +/- 0.0172 | 4.8348 +/- 0.0049 | 4.5281 +/- 0.0072 | 4.3683 +/- 0.0072 | 4.2811 +/- 0.0074 | 4.2375 +/- 0.0086 |
| RLB+Lion | 5.4528 +/- 0.0239 | 4.7506 +/- 0.0088 | 4.4331 +/- 0.0041 | 4.2754 +/- 0.0061 | 4.1875 +/- 0.0057 | 4.1427 +/- 0.0068 |
| SiLU+Lion | 5.5121 +/- 0.0186 | 4.7814 +/- 0.0167 | 4.4520 +/- 0.0124 | 4.2870 +/- 0.0101 | 4.1964 +/- 0.0086 | 4.1494 +/- 0.0092 |
| RLB+SOAP | 5.9371 +/- 0.8724 | 4.9025 +/- 0.0578 | 4.5635 +/- 0.0082 | 4.4069 +/- 0.0287 | 4.3073 +/- 0.0147 | 4.2623 +/- 0.0131 |
| SiLU+SOAP | 5.5541 +/- 0.0178 | 5.0548 +/- 0.1513 | 4.6308 +/- 0.0214 | 4.4105 +/- 0.0136 | 4.3051 +/- 0.0112 | 4.2630 +/- 0.0220 |
| RLB+Muon | 5.8189 +/- 0.0126 | 4.9679 +/- 0.0124 | 4.6289 +/- 0.0139 | 4.4407 +/- 0.0180 | 4.3387 +/- 0.0192 | 4.2877 +/- 0.0199 |
| SiLU+Muon | 5.8660 +/- 0.0098 | 4.9736 +/- 0.0094 | 4.6220 +/- 0.0117 | 4.4295 +/- 0.0202 | 4.3292 +/- 0.0230 | 4.2787 +/- 0.0243 |
| RLB+ScheduleFree | 5.9870 +/- 0.0127 | 5.4052 +/- 0.0161 | 5.1139 +/- 0.0151 | 4.9433 +/- 0.0138 | 4.8431 +/- 0.0123 | 4.7797 +/- 0.0113 |
| SiLU+ScheduleFree | 6.0279 +/- 0.0274 | 5.4487 +/- 0.0075 | 5.1579 +/- 0.0075 | 4.9894 +/- 0.0073 | 4.8896 +/- 0.0073 | 4.8258 +/- 0.0072 |
| RLB+CAME | 6.0101 +/- 0.0223 | 5.4786 +/- 0.0122 | 5.1973 +/- 0.0082 | 5.0381 +/- 0.0064 | 4.9489 +/- 0.0052 | 4.9043 +/- 0.0046 |
| SiLU+CAME | 6.0899 +/- 0.0329 | 5.5003 +/- 0.0147 | 5.2179 +/- 0.0104 | 5.0573 +/- 0.0103 | 4.9669 +/- 0.0110 | 4.9207 +/- 0.0123 |
| RLB+ADeMaMix | 16.3601 +/- 9.0794 | 772.2738 +/- 845.8176 | 2575.6854 +/- 3736.6125 | 2426.1191 +/- 2936.8239 | 5742.2397 +/- 8431.4467 | 7880.5115 +/- 12045.6569 |
| SiLU+ADeMaMix | 6.5917 +/- 0.0381 | 323.2475 +/- 232.2217 | 132.7501 +/- 91.4211 | 293.8233 +/- 355.2971 | 262.2238 +/- 285.4760 | 242.8537 +/- 242.0122 |

#### FineWeb

All-method view:

![FineWeb E1 validation loss mean +/- std, all methods](results/iclr26_e1_figures/fineweb_core_validation_loss_mean_std.svg)

![FineWeb E1 validation PPL mean +/- std, all methods](results/iclr26_e1_figures/fineweb_core_validation_ppl_mean_std.svg)

![FineWeb E1 training loss mean +/- std, all methods](results/iclr26_e1_figures/fineweb_core_training_loss_mean_std.svg)

Clean comparison view:

![FineWeb E1 validation loss mean +/- std, clean comparison](results/iclr26_e1_figures/fineweb_clean_validation_loss_mean_std.svg)

![FineWeb E1 validation PPL mean +/- std, clean comparison](results/iclr26_e1_figures/fineweb_clean_validation_ppl_mean_std.svg)

![FineWeb E1 training loss mean +/- std, clean comparison](results/iclr26_e1_figures/fineweb_clean_training_loss_mean_std.svg)

FineWeb validation-loss checkpoint table, mean +/- sample std:

| Method | 500 | 1000 | 1500 | 2000 | 2500 | 3050 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MatrixPolicy | 5.4032 +/- 0.0156 | 4.9221 +/- 0.0161 | 4.6271 +/- 0.0136 | 4.4672 +/- 0.0109 | 4.3721 +/- 0.0112 | 4.3195 +/- 0.0124 |
| RLB+AdamW | 5.5178 +/- 0.0196 | 5.0226 +/- 0.0213 | 4.7466 +/- 0.0175 | 4.5966 +/- 0.0132 | 4.5147 +/- 0.0129 | 4.4705 +/- 0.0133 |
| SiLU+AdamW | 5.5394 +/- 0.0181 | 5.0366 +/- 0.0139 | 4.7579 +/- 0.0118 | 4.6047 +/- 0.0092 | 4.5202 +/- 0.0087 | 4.4758 +/- 0.0097 |
| RLB+Lion | 5.5608 +/- 0.0036 | 4.9461 +/- 0.0228 | 4.6521 +/- 0.0108 | 4.4989 +/- 0.0078 | 4.4130 +/- 0.0078 | 4.3671 +/- 0.0075 |
| SiLU+Lion | 5.6186 +/- 0.0104 | 4.9885 +/- 0.0180 | 4.6808 +/- 0.0122 | 4.5218 +/- 0.0101 | 4.4307 +/- 0.0075 | 4.3825 +/- 0.0083 |
| RLB+SOAP | 5.5532 +/- 0.0487 | 5.0552 +/- 0.0394 | 4.8945 +/- 0.2541 | 4.6072 +/- 0.0207 | 4.5216 +/- 0.0175 | 4.4850 +/- 0.0255 |
| SiLU+SOAP | 5.7363 +/- 0.0492 | 5.2006 +/- 0.1139 | 4.8417 +/- 0.0132 | 4.6430 +/- 0.0128 | 4.5362 +/- 0.0096 | 4.4840 +/- 0.0112 |
| RLB+Muon | 5.9544 +/- 0.0286 | 5.1847 +/- 0.0223 | 4.8656 +/- 0.0210 | 4.6768 +/- 0.0130 | 4.5746 +/- 0.0120 | 4.5216 +/- 0.0120 |
| SiLU+Muon | 5.9974 +/- 0.0298 | 5.1931 +/- 0.0229 | 4.8588 +/- 0.0247 | 4.6661 +/- 0.0269 | 4.5666 +/- 0.0259 | 4.5163 +/- 0.0264 |
| RLB+ScheduleFree | 6.0415 +/- 0.0245 | 5.5256 +/- 0.0232 | 5.2750 +/- 0.0196 | 5.1298 +/- 0.0199 | 5.0436 +/- 0.0196 | 4.9877 +/- 0.0196 |
| SiLU+ScheduleFree | 6.0727 +/- 0.0207 | 5.5504 +/- 0.0249 | 5.3021 +/- 0.0209 | 5.1573 +/- 0.0198 | 5.0707 +/- 0.0193 | 5.0142 +/- 0.0188 |
| RLB+CAME | 6.0994 +/- 0.0162 | 5.6161 +/- 0.0098 | 5.3788 +/- 0.0155 | 5.2417 +/- 0.0161 | 5.1651 +/- 0.0167 | 5.1251 +/- 0.0173 |
| SiLU+CAME | 6.1551 +/- 0.0249 | 5.6317 +/- 0.0207 | 5.3872 +/- 0.0165 | 5.2496 +/- 0.0148 | 5.1725 +/- 0.0124 | 5.1325 +/- 0.0126 |
| RLB+ADeMaMix | 36.3461 +/- 22.7023 | 41680578.9202 +/- 72192221.5413 | 44753414.1145 +/- 63289962.8122 (n=2) | 107762.9375 +/- 0.0000 (n=1) | 8023604.0000 +/- 0.0000 (n=1) | 3022914304.0000 +/- 0.0000 (n=1) |
| SiLU+ADeMaMix | 6.5763 +/- 0.2655 | 258.3239 +/- 44.9409 | 83.0640 +/- 6.9598 | 69.4284 +/- 15.1007 | 57.6825 +/- 14.4696 | 51.9965 +/- 15.1322 |

#### Dolma-sample

All-method view:

![Dolma-sample E1 validation loss mean +/- std, all methods](results/iclr26_e1_figures/dolma_sample_core_validation_loss_mean_std.svg)

![Dolma-sample E1 validation PPL mean +/- std, all methods](results/iclr26_e1_figures/dolma_sample_core_validation_ppl_mean_std.svg)

![Dolma-sample E1 training loss mean +/- std, all methods](results/iclr26_e1_figures/dolma_sample_core_training_loss_mean_std.svg)

Clean comparison view:

![Dolma-sample E1 validation loss mean +/- std, clean comparison](results/iclr26_e1_figures/dolma_sample_clean_validation_loss_mean_std.svg)

![Dolma-sample E1 validation PPL mean +/- std, clean comparison](results/iclr26_e1_figures/dolma_sample_clean_validation_ppl_mean_std.svg)

![Dolma-sample E1 training loss mean +/- std, clean comparison](results/iclr26_e1_figures/dolma_sample_clean_training_loss_mean_std.svg)

Dolma-sample validation-loss checkpoint table, mean +/- sample std:

| Method | 500 | 1000 | 1500 | 2000 | 2500 | 3050 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MatrixPolicy | 5.4560 +/- 0.0189 | 4.9527 +/- 0.0144 | 4.6415 +/- 0.0038 | 4.4745 +/- 0.0059 | 4.3769 +/- 0.0053 | 4.3239 +/- 0.0052 |
| RLB+AdamW | 5.5707 +/- 0.0183 | 5.0703 +/- 0.0112 | 4.7756 +/- 0.0054 | 4.6177 +/- 0.0004 | 4.5315 +/- 0.0014 | 4.4881 +/- 0.0009 |
| SiLU+AdamW | 5.5934 +/- 0.0195 | 5.0809 +/- 0.0222 | 4.7821 +/- 0.0118 | 4.6197 +/- 0.0050 | 4.5310 +/- 0.0031 | 4.4862 +/- 0.0012 |
| RLB+Lion | 5.5929 +/- 0.0339 | 4.9824 +/- 0.0271 | 4.6652 +/- 0.0129 | 4.5053 +/- 0.0076 | 4.4155 +/- 0.0075 | 4.3693 +/- 0.0056 |
| SiLU+Lion | 5.6628 +/- 0.0202 | 5.0312 +/- 0.0103 | 4.6966 +/- 0.0021 | 4.5289 +/- 0.0051 | 4.4359 +/- 0.0042 | 4.3878 +/- 0.0046 |
| RLB+SOAP | 5.6096 +/- 0.0145 | 5.3383 +/- 0.2027 | 4.8130 +/- 0.0153 | 4.6354 +/- 0.0064 | 4.5549 +/- 0.0233 | 4.5029 +/- 0.0157 |
| SiLU+SOAP | 6.0964 +/- 0.2025 | 5.2723 +/- 0.0950 | 4.8553 +/- 0.0074 | 4.6570 +/- 0.0040 | 4.5539 +/- 0.0057 | 4.4987 +/- 0.0035 |
| RLB+Muon | 6.0065 +/- 0.0095 | 5.2374 +/- 0.0163 | 4.9153 +/- 0.0067 | 4.7237 +/- 0.0071 | 4.6130 +/- 0.0088 | 4.5586 +/- 0.0093 |
| SiLU+Muon | 6.0530 +/- 0.0100 | 5.2462 +/- 0.0160 | 4.9072 +/- 0.0101 | 4.7048 +/- 0.0142 | 4.5973 +/- 0.0117 | 4.5443 +/- 0.0108 |
| RLB+ScheduleFree | 6.0972 +/- 0.0252 | 5.5799 +/- 0.0211 | 5.3207 +/- 0.0142 | 5.1714 +/- 0.0137 | 5.0815 +/- 0.0141 | 5.0241 +/- 0.0136 |
| SiLU+ScheduleFree | 6.1296 +/- 0.0239 | 5.6030 +/- 0.0177 | 5.3454 +/- 0.0127 | 5.1974 +/- 0.0125 | 5.1075 +/- 0.0140 | 5.0494 +/- 0.0149 |
| RLB+CAME | 6.1255 +/- 0.0218 | 5.6432 +/- 0.0166 | 5.3986 +/- 0.0175 | 5.2610 +/- 0.0162 | 5.1826 +/- 0.0171 | 5.1425 +/- 0.0164 |
| SiLU+CAME | 6.1808 +/- 0.0170 | 5.6664 +/- 0.0188 | 5.4216 +/- 0.0168 | 5.2824 +/- 0.0187 | 5.2049 +/- 0.0189 | 5.1650 +/- 0.0189 |
| RLB+ADeMaMix | 12.7395 +/- 3.3039 | 84.8182 +/- 36.9919 | 27552.1354 +/- 46889.3594 | 10779257.3654 +/- 18662299.4102 | 570914604.1487 +/- 624651995.0860 | 1775543041.7179 +/- 2510996321.2383 (n=2) |
| SiLU+ADeMaMix | 6.4467 +/- 0.1480 | 168.0197 +/- 131.3544 | 5706.2408 +/- 5051.5183 | 10605.0472 +/- 10104.8045 | 30618.4067 +/- 44187.0559 | 28427.1854 +/- 40987.7167 |

#### C4

All-method view:

![C4 E1 validation loss mean +/- std, all methods](results/iclr26_e1_figures/c4_en_core_validation_loss_mean_std.svg)

![C4 E1 validation PPL mean +/- std, all methods](results/iclr26_e1_figures/c4_en_core_validation_ppl_mean_std.svg)

![C4 E1 training loss mean +/- std, all methods](results/iclr26_e1_figures/c4_en_core_training_loss_mean_std.svg)

Clean comparison view:

![C4 E1 validation loss mean +/- std, clean comparison](results/iclr26_e1_figures/c4_en_clean_validation_loss_mean_std.svg)

![C4 E1 validation PPL mean +/- std, clean comparison](results/iclr26_e1_figures/c4_en_clean_validation_ppl_mean_std.svg)

![C4 E1 training loss mean +/- std, clean comparison](results/iclr26_e1_figures/c4_en_clean_training_loss_mean_std.svg)

C4 validation-loss checkpoint table, mean +/- sample std:

| Method | 500 | 1000 | 1500 | 2000 | 2500 | 3050 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MatrixPolicy | 5.4130 +/- 0.0080 | 4.8952 +/- 0.0089 | 4.5937 +/- 0.0136 | 4.4342 +/- 0.0138 | 4.3381 +/- 0.0173 | 4.2864 +/- 0.0193 |
| RLB+AdamW | 5.5330 +/- 0.0118 | 5.0084 +/- 0.0302 | 4.7194 +/- 0.0216 | 4.5692 +/- 0.0192 | 4.4856 +/- 0.0202 | 4.4426 +/- 0.0198 |
| SiLU+AdamW | 5.5616 +/- 0.0107 | 5.0231 +/- 0.0277 | 4.7320 +/- 0.0184 | 4.5768 +/- 0.0160 | 4.4906 +/- 0.0152 | 4.4469 +/- 0.0160 |
| RLB+Lion | 5.5873 +/- 0.0133 | 4.9229 +/- 0.0351 | 4.6203 +/- 0.0214 | 4.4674 +/- 0.0203 | 4.3807 +/- 0.0205 | 4.3357 +/- 0.0209 |
| SiLU+Lion | 5.6375 +/- 0.0126 | 4.9687 +/- 0.0115 | 4.6536 +/- 0.0128 | 4.4926 +/- 0.0136 | 4.4011 +/- 0.0144 | 4.3536 +/- 0.0156 |
| RLB+SOAP | 5.6443 +/- 0.0198 | 5.8223 +/- 1.3681 | 4.7627 +/- 0.0155 | 4.5935 +/- 0.0112 | 4.5049 +/- 0.0106 | 4.4572 +/- 0.0104 |
| SiLU+SOAP | 5.8159 +/- 0.1176 | 5.2260 +/- 0.1436 | 4.8170 +/- 0.0138 | 4.6212 +/- 0.0160 | 4.5109 +/- 0.0162 | 4.4582 +/- 0.0146 |
| RLB+Muon | 5.9708 +/- 0.0186 | 5.1843 +/- 0.0141 | 4.8502 +/- 0.0156 | 4.6540 +/- 0.0238 | 4.5479 +/- 0.0229 | 4.4945 +/- 0.0220 |
| SiLU+Muon | 6.0187 +/- 0.0276 | 5.1896 +/- 0.0198 | 4.8377 +/- 0.0242 | 4.6374 +/- 0.0268 | 4.5341 +/- 0.0244 | 4.4824 +/- 0.0233 |
| RLB+ScheduleFree | 6.0659 +/- 0.0184 | 5.5375 +/- 0.0151 | 5.2736 +/- 0.0142 | 5.1194 +/- 0.0127 | 5.0285 +/- 0.0139 | 4.9701 +/- 0.0149 |
| SiLU+ScheduleFree | 6.1120 +/- 0.0195 | 5.5750 +/- 0.0171 | 5.3125 +/- 0.0171 | 5.1580 +/- 0.0157 | 5.0659 +/- 0.0156 | 5.0064 +/- 0.0160 |
| RLB+CAME | 6.0983 +/- 0.0098 | 5.6122 +/- 0.0075 | 5.3640 +/- 0.0117 | 5.2218 +/- 0.0131 | 5.1413 +/- 0.0141 | 5.1000 +/- 0.0153 |
| SiLU+CAME | 6.1693 +/- 0.0157 | 5.6365 +/- 0.0144 | 5.3864 +/- 0.0149 | 5.2439 +/- 0.0156 | 5.1645 +/- 0.0170 | 5.1230 +/- 0.0178 |
| RLB+ADeMaMix | 38.4298 +/- 25.9135 | 1775.8338 +/- 2904.0404 | 5460.5850 +/- 8375.8913 | 677978.0837 +/- 1152345.9869 | 538384685.3267 +/- 932479138.1133 | 449788003.0859 +/- 779025043.4338 |
| SiLU+ADeMaMix | 6.6677 +/- 0.2286 | 718.5009 +/- 821.5844 | 579.6937 +/- 406.8724 | 377.1749 +/- 304.4101 | 351.8885 +/- 347.3860 | 344.9990 +/- 372.6250 |

## Current E2 M0/300M Results

E2 M0/300M is complete for DCLM rows `240-284`, FineWeb-Edu rows `285-329`, FineWeb rows `330-374`, Dolma-sample rows `375-419`, and C4 rows `420-464`. Each completed cell has three seeds, 15 fixed methods per seed, final eval at step `9150`, `32768` global tokens/step, and about `299.8M` train tokens per run. MatrixPolicy entries below use `manifests/iclr26_matrixpolicy_safe_speed_e2_manifest.csv` and raw JSONL from `runs/iclr26_main/E2_matrixpolicy_safe_speed_300m/`; comparator methods use the completed main E2 rows.

### DCLM

Tracked package: `results/iclr26_e2_dclm_2026_06_10/`.

Completed: 2026-06-10. Manifest rows `240-284` define the full DCLM E2 M0/300M cell: 3 seeds x 15 fixed methods. All 45 paper-facing rows have final eval at step `9150`.

Each row uses `32768` global tokens/step for about `299.8M` train tokens. Validation uses the E2 DCLM slice from the manifest: `val_skip_tokens=610000000`, `val_tokens=8000000`, `eval_interval=50`.
MatrixPolicy entries use the accepted safe-speed replacement JSONL rows for the same method and seed; the `row` column remains the matched main-manifest E2 row, while `source_phase`/`source_row_id` record the actual timed run.

#### Final Validation Loss

| Method | Final val loss mean +/- sample std | Min | Max | Notes |
| --- | ---: | ---: | ---: | --- |
| rlb_matrixpolicy_original | 3.956069 +/- 0.030752 | 3.921971 | 3.981703 |  |
| silu_lion | 3.993430 +/- 0.023038 | 3.968264 | 4.013479 |  |
| rlb_muon | 3.993489 +/- 0.029634 | 3.961723 | 4.020390 |  |
| rlb_lion | 3.994293 +/- 0.030088 | 3.960352 | 4.017691 |  |
| silu_muon | 3.997266 +/- 0.030472 | 3.964678 | 4.025052 |  |
| silu_adamw | 4.049337 +/- 0.027469 | 4.018327 | 4.070612 |  |
| rlb_adamw | 4.052915 +/- 0.028179 | 4.021017 | 4.074428 |  |
| rlb_soap | 4.076804 +/- 0.040305 | 4.034326 | 4.114511 |  |
| silu_soap | 4.096430 +/- 0.029988 | 4.062710 | 4.120108 |  |
| rlb_schedulefree | 4.356261 +/- 0.033232 | 4.318152 | 4.379206 |  |
| silu_schedulefree | 4.365672 +/- 0.029805 | 4.332936 | 4.391239 |  |
| silu_came | 4.368189 +/- 0.022586 | 4.344955 | 4.390067 |  |
| rlb_came | 4.450294 +/- 0.034021 | 4.428269 | 4.489478 |  |
| rlb_ademamix | nan/diverged | nan | nan | 3 diverged/non-finite seeds |
| silu_ademamix | nan/diverged | nan | nan | 3 diverged/non-finite seeds |

MatrixPolicy is best on all three DCLM E2 seeds. Mean final val loss is `3.956069 +/- 0.030752`; the next-best aggregate methods are `silu_lion` at `3.993430 +/- 0.023038`, `rlb_muon` at `3.993489 +/- 0.029634`, `rlb_lion` at `3.994293 +/- 0.030088`.

#### Per-Seed MatrixPolicy Gap

| Seed | MatrixPolicy final loss | Best non-MP method | Best non-MP final loss | Gap |
| ---: | ---: | --- | ---: | ---: |
| 1337 | 3.964531 | rlb_muon | 3.998354 | 0.033823 |
| 2027 | 3.981703 | silu_lion | 4.013479 | 0.031776 |
| 3407 | 3.921971 | rlb_lion | 3.960352 | 0.038381 |

#### Runtime Summary

`summary.total_seconds` is training-harness wall time for the manifest row. It excludes Slurm queue wait, dependency wait, token-cache construction, extension compilation, and launcher overhead.

| Method | Runs | Mean runtime | Std | Range | Mean s/step | Mean tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| silu_lion | 3 | 67.6 min | 12.0 min | 60.5-81.5 min | 0.4265 | 78442.0 |
| silu_schedulefree | 3 | 68.9 min | 11.9 min | 61.6-82.6 min | 0.4343 | 76941.0 |
| rlb_matrixpolicy_original | 3 | 69.6 min | 2.8 min | 66.4-71.3 min | 0.4304 | 76212.4 |
| silu_adamw | 3 | 70.9 min | 10.0 min | 61.1-81.2 min | 0.4478 | 74246.1 |
| silu_muon | 3 | 71.4 min | 14.8 min | 59.4-87.9 min | 0.4497 | 74940.2 |
| rlb_schedulefree | 3 | 78.3 min | 0.1 min | 78.1-78.4 min | 0.4835 | 67778.1 |
| silu_came | 3 | 80.2 min | 11.7 min | 66.7-87.2 min | 0.5085 | 65529.0 |
| rlb_lion | 3 | 83.6 min | 11.5 min | 76.9-96.9 min | 0.5184 | 64027.8 |
| rlb_adamw | 3 | 83.6 min | 10.9 min | 77.3-96.3 min | 0.5188 | 63911.7 |
| silu_ademamix | 3 | 85.3 min | 18.1 min | 64.5-96.1 min | 0.4823 | 69300.4 |
| rlb_soap | 3 | 85.6 min | 11.0 min | 79.2-98.2 min | 0.5310 | 62405.0 |
| silu_soap | 3 | 86.9 min | 12.3 min | 72.8-94.5 min | 0.5487 | 60587.7 |
| rlb_muon | 3 | 89.0 min | 11.4 min | 82.3-102.1 min | 0.5530 | 59894.1 |
| rlb_came | 3 | 90.0 min | 10.6 min | 83.9-102.3 min | 0.5604 | 59034.6 |
| rlb_ademamix | 3 | 107.9 min | 12.1 min | 100.8-121.9 min | 0.5201 | 63823.4 |

#### Token-To-Target Savings

This table asks how many training tokens were needed to first reach a validation-loss threshold. It does not change the fixed-budget protocol; all completed runs still trained to about `299.8M` tokens. The readout uses the native eval cadence of 50 steps, or `1.64M` tokens.

`Second-best` means the fastest non-MatrixPolicy method to reach the target within the same seed. `AdamW` means the standard `silu_adamw` row. Savings and proportions are computed only on seeds where both MatrixPolicy and the comparator reached the target, so the comparison column explicitly shows `MP -> comparator` tokens and the shared seed count.

| Target loss | MP all-hit mean | Vs fastest non-MP: MP -> comparator (seeds) | Saved | Saved % | Vs SiLU+AdamW: MP -> AdamW (seeds) | Saved | Saved % |
| ---: | ---: | --- | ---: | ---: | --- | ---: | ---: |
| 4.40 | 74.3M | 74.3M -> 80.8M (3/3) | 6.6M | 8.1% | 74.3M -> 93.4M (3/3) | 19.1M | 20.5% |
| 4.30 | 102.1M | 102.1M -> 104.9M (3/3) | 2.7M | 2.6% | 102.1M -> 120.7M (3/3) | 18.6M | 15.4% |
| 4.20 | 133.3M | 133.3M -> 139.3M (3/3) | 6.0M | 4.3% | 133.3M -> 161.1M (3/3) | 27.9M | 17.3% |
| 4.10 | 175.9M | 175.9M -> 187.9M (3/3) | 12.0M | 6.4% | 175.9M -> 227.7M (3/3) | 51.9M | 22.8% |
| 4.05 | 204.3M | 204.3M -> 222.8M (3/3) | 18.6M | 8.3% | 183.5M -> 244.1M (1/3) | 60.6M | 24.8% |
| 4.00 | 243.6M | 231.8M -> 267.9M (2/3) | 36.0M | 13.5% | not reached (0/3) | not reached | n/a |

### FineWeb-Edu

Tracked package: `results/iclr26_e2_fineweb_edu_2026_06_12/`.

Completed: 2026-06-12. Manifest rows `285-329` define the full FineWeb-Edu E2 M0/300M cell: 3 seeds x 15 fixed methods. All 45 paper-facing rows have final eval at step `9150`.

Each row uses `32768` global tokens/step for about `299.8M` train tokens. Validation uses the E2 FineWeb-Edu slice from the manifest: `val_skip_tokens=610000000`, `val_tokens=8000000`, `eval_interval=50`.
MatrixPolicy entries use the accepted safe-speed replacement JSONL rows for the same method and seed; the `row` column remains the matched main-manifest E2 row, while `source_phase`/`source_row_id` record the actual timed run.

#### Final Validation Loss

| Method | Final val loss mean +/- sample std | Min | Max | Notes |
| --- | ---: | ---: | ---: | --- |
| rlb_matrixpolicy_original | 3.707768 +/- 0.018711 | 3.691080 | 3.727996 |  |
| rlb_muon | 3.738164 +/- 0.021014 | 3.718084 | 3.760002 |  |
| silu_lion | 3.744017 +/- 0.020802 | 3.727149 | 3.767261 |  |
| rlb_lion | 3.745142 +/- 0.021429 | 3.727976 | 3.769158 |  |
| silu_muon | 3.745389 +/- 0.017006 | 3.732584 | 3.764685 |  |
| silu_adamw | 3.803482 +/- 0.018186 | 3.788790 | 3.823822 |  |
| rlb_adamw | 3.806861 +/- 0.017650 | 3.790723 | 3.825709 |  |
| rlb_soap | 3.830115 +/- 0.019851 | 3.813878 | 3.852245 |  |
| silu_soap | 3.862925 +/- 0.020105 | 3.844579 | 3.884418 |  |
| rlb_schedulefree | 4.136537 +/- 0.021710 | 4.116814 | 4.159799 |  |
| silu_came | 4.150283 +/- 0.021107 | 4.137987 | 4.174655 |  |
| silu_schedulefree | 4.155890 +/- 0.023847 | 4.134111 | 4.181372 |  |
| rlb_came | 4.220261 +/- 0.036032 | 4.188819 | 4.259580 |  |
| rlb_ademamix | nan/diverged | nan | nan | 3 diverged/non-finite seeds |
| silu_ademamix | nan/diverged | nan | nan | 3 diverged/non-finite seeds |

MatrixPolicy is best on all three FineWeb-Edu E2 seeds. Mean final val loss is `3.707768 +/- 0.018711`; the next-best aggregate methods are `rlb_muon` at `3.738164 +/- 0.021014`, `silu_lion` at `3.744017 +/- 0.020802`, `rlb_lion` at `3.745142 +/- 0.021429`.

#### Per-Seed MatrixPolicy Gap

| Seed | MatrixPolicy final loss | Best non-MP method | Best non-MP final loss | Gap |
| ---: | ---: | --- | ---: | ---: |
| 1337 | 3.691080 | rlb_muon | 3.718084 | 0.027004 |
| 2027 | 3.727996 | rlb_muon | 3.760002 | 0.032006 |
| 3407 | 3.704228 | rlb_muon | 3.736407 | 0.032179 |

#### Runtime Summary

`summary.total_seconds` is training-harness wall time for the manifest row. It excludes Slurm queue wait, dependency wait, token-cache construction, extension compilation, and launcher overhead.

| Method | Runs | Mean runtime | Std | Range | Mean s/step | Mean tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| rlb_matrixpolicy_original | 3 | 69.6 min | 2.9 min | 66.3-71.4 min | 0.4304 | 76215.7 |
| silu_muon | 3 | 72.8 min | 12.9 min | 63.8-87.6 min | 0.4585 | 72920.4 |
| silu_lion | 3 | 74.5 min | 11.5 min | 61.2-81.1 min | 0.4709 | 70923.0 |
| silu_adamw | 3 | 75.2 min | 14.0 min | 61.4-89.4 min | 0.4754 | 70683.2 |
| silu_schedulefree | 3 | 75.5 min | 11.9 min | 61.8-82.4 min | 0.4781 | 69904.1 |
| silu_came | 3 | 80.2 min | 11.7 min | 66.7-87.0 min | 0.5079 | 65602.6 |
| rlb_schedulefree | 3 | 86.6 min | 14.4 min | 78.3-103.3 min | 0.5388 | 61973.9 |
| silu_soap | 3 | 86.8 min | 12.0 min | 72.9-93.8 min | 0.5475 | 60685.2 |
| silu_ademamix | 3 | 88.5 min | 10.7 min | 76.5-97.0 min | 0.4762 | 70175.4 |
| rlb_lion | 3 | 91.2 min | 12.7 min | 77.0-101.3 min | 0.5685 | 58526.8 |
| rlb_came | 3 | 92.6 min | 15.0 min | 83.9-110.0 min | 0.5774 | 57751.3 |
| rlb_soap | 3 | 93.5 min | 12.6 min | 79.3-103.3 min | 0.5832 | 56995.4 |
| rlb_adamw | 3 | 100.6 min | 8.5 min | 95.7-110.4 min | 0.6304 | 52246.8 |
| rlb_muon | 3 | 102.9 min | 3.1 min | 101.1-106.5 min | 0.6429 | 51009.0 |
| rlb_ademamix | 3 | 108.2 min | 14.2 min | 98.8-124.5 min | 0.5356 | 62475.8 |

#### Token-To-Target Savings

This table asks how many training tokens were needed to first reach a validation-loss threshold. It does not change the fixed-budget protocol; all completed runs still trained to about `299.8M` tokens. The readout uses the native eval cadence of 50 steps, or `1.64M` tokens.

`Second-best` means the fastest non-MatrixPolicy method to reach the target within the same seed. `AdamW` means the standard `silu_adamw` row. Savings and proportions are computed only on seeds where both MatrixPolicy and the comparator reached the target, so the comparison column explicitly shows `MP -> comparator` tokens and the shared seed count.

| Target loss | MP all-hit mean | Vs fastest non-MP: MP -> comparator (seeds) | Saved | Saved % | Vs SiLU+AdamW: MP -> AdamW (seeds) | Saved | Saved % |
| ---: | ---: | --- | ---: | ---: | --- | ---: | ---: |
| 4.20 | 74.3M | 74.3M -> 81.4M (3/3) | 7.1M | 8.7% | 74.3M -> 93.4M (3/3) | 19.1M | 20.5% |
| 4.10 | 100.5M | 100.5M -> 102.7M (3/3) | 2.2M | 2.1% | 100.5M -> 118.0M (3/3) | 17.5M | 14.8% |
| 4.00 | 127.8M | 127.8M -> 130.5M (3/3) | 2.7M | 2.1% | 127.8M -> 151.8M (3/3) | 24.0M | 15.8% |
| 3.90 | 163.3M | 163.3M -> 167.7M (3/3) | 4.4M | 2.6% | 163.3M -> 200.4M (3/3) | 37.1M | 18.5% |
| 3.85 | 185.7M | 185.7M -> 191.1M (3/3) | 5.5M | 2.9% | 185.7M -> 237.0M (3/3) | 51.3M | 21.7% |
| 3.80 | 211.9M | 211.9M -> 224.5M (3/3) | 12.6M | 5.6% | 205.6M -> 287.5M (2/3) | 81.9M | 28.5% |
| 3.75 | 248.5M | 239.2M -> 262.1M (2/3) | 22.9M | 8.8% | not reached (0/3) | not reached | n/a |

### FineWeb

Tracked package: `results/iclr26_e2_fineweb_2026_06_15/`.

Completed: 2026-06-15. Manifest rows `330-374` define the full FineWeb E2 M0/300M cell: 3 seeds x 15 fixed methods. All 45 paper-facing rows have final eval at step `9150`.

Each row uses `32768` global tokens/step for about `299.8M` train tokens. Validation uses the E2 FineWeb slice from the manifest: `val_skip_tokens=610000000`, `val_tokens=8000000`, `eval_interval=50`.
MatrixPolicy entries use the accepted safe-speed replacement JSONL rows for the same method and seed; the `row` column remains the matched main-manifest E2 row, while `source_phase`/`source_row_id` record the actual timed run.

#### Final Validation Loss

| Method | Final val loss mean +/- sample std | Min | Max | Notes |
| --- | ---: | ---: | ---: | --- |
| rlb_matrixpolicy_original | 3.964892 +/- 0.009459 | 3.959276 | 3.975813 |  |
| rlb_muon | 4.001245 +/- 0.011375 | 3.991416 | 4.013704 |  |
| rlb_lion | 4.001381 +/- 0.012800 | 3.991274 | 4.015774 |  |
| silu_lion | 4.001499 +/- 0.008463 | 3.995715 | 4.011213 |  |
| silu_muon | 4.006567 +/- 0.012834 | 3.996716 | 4.021081 |  |
| silu_adamw | 4.061199 +/- 0.010087 | 4.053473 | 4.072610 |  |
| rlb_adamw | 4.062934 +/- 0.009826 | 4.054112 | 4.073524 |  |
| rlb_soap | 4.084052 +/- 0.007895 | 4.077687 | 4.092887 |  |
| silu_soap | 4.113942 +/- 0.010095 | 4.104786 | 4.124768 |  |
| rlb_schedulefree | 4.381390 +/- 0.009252 | 4.374903 | 4.391984 |  |
| silu_schedulefree | 4.397873 +/- 0.010600 | 4.390438 | 4.410011 |  |
| silu_came | 4.406034 +/- 0.018922 | 4.393516 | 4.427802 |  |
| rlb_came | 4.473173 +/- 0.001144 | 4.471895 | 4.474098 |  |
| silu_ademamix | 1361.414062 +/- 0.000000 | 1361.414062 | 1361.414062 | 2 diverged/non-finite seeds |
| rlb_ademamix | nan/diverged | nan | nan | 3 diverged/non-finite seeds |

MatrixPolicy is best on all three FineWeb E2 seeds. Mean final val loss is `3.964892 +/- 0.009459`; the next-best aggregate methods are `rlb_muon` at `4.001245 +/- 0.011375`, `rlb_lion` at `4.001381 +/- 0.012800`, `silu_lion` at `4.001499 +/- 0.008463`.

#### Per-Seed MatrixPolicy Gap

| Seed | MatrixPolicy final loss | Best non-MP method | Best non-MP final loss | Gap |
| ---: | ---: | --- | ---: | ---: |
| 1337 | 3.975813 | silu_lion | 4.011213 | 0.035400 |
| 2027 | 3.959276 | rlb_lion | 3.991274 | 0.031998 |
| 3407 | 3.959586 | rlb_lion | 3.997095 | 0.037509 |

#### Runtime Summary

`summary.total_seconds` is training-harness wall time for the manifest row. It excludes Slurm queue wait, dependency wait, token-cache construction, extension compilation, and launcher overhead.

| Method | Runs | Mean runtime | Std | Range | Mean s/step | Mean tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| silu_schedulefree | 3 | 61.7 min | 0.4 min | 61.3-62.1 min | 0.3875 | 84557.5 |
| silu_lion | 3 | 64.6 min | 6.3 min | 61.0-71.9 min | 0.4066 | 81108.9 |
| rlb_matrixpolicy_original | 3 | 67.1 min | 3.5 min | 65.0-71.2 min | 0.4146 | 79179.4 |
| silu_muon | 3 | 69.7 min | 6.2 min | 65.6-76.8 min | 0.4385 | 75078.6 |
| silu_soap | 3 | 72.3 min | 0.6 min | 71.7-72.9 min | 0.4571 | 71696.4 |
| rlb_adamw | 3 | 76.3 min | 1.3 min | 74.8-77.2 min | 0.4700 | 69737.2 |
| silu_adamw | 3 | 76.4 min | 18.9 min | 60.2-97.1 min | 0.4840 | 70588.2 |
| rlb_lion | 3 | 79.6 min | 14.1 min | 68.6-95.5 min | 0.4920 | 68070.8 |
| rlb_muon | 3 | 85.0 min | 14.3 min | 74.0-101.2 min | 0.5263 | 63461.5 |
| rlb_soap | 3 | 85.4 min | 11.1 min | 76.9-97.9 min | 0.5299 | 62572.7 |
| rlb_schedulefree | 3 | 85.6 min | 10.0 min | 77.3-96.7 min | 0.5304 | 62408.2 |
| rlb_came | 3 | 91.1 min | 10.1 min | 82.8-102.3 min | 0.5662 | 58393.3 |
| rlb_ademamix | 3 | 104.0 min | 12.7 min | 96.0-118.7 min | 0.5141 | 64685.3 |
| silu_came | 3 | 147.2 min | 139.2 min | 66.6-307.9 min | 0.9476 | 57383.1 |
| silu_ademamix | 3 | 150.6 min | 143.3 min | 62.6-316.0 min | 0.9251 | 60189.7 |

#### Token-To-Target Savings

This table asks how many training tokens were needed to first reach a validation-loss threshold. It does not change the fixed-budget protocol; all completed runs still trained to about `299.8M` tokens. The readout uses the native eval cadence of 50 steps, or `1.64M` tokens.

`Second-best` means the fastest non-MatrixPolicy method to reach the target within the same seed. `AdamW` means the standard `silu_adamw` row. Savings and proportions are computed only on seeds where both MatrixPolicy and the comparator reached the target, so the comparison column explicitly shows `MP -> comparator` tokens and the shared seed count.

| Target loss | MP all-hit mean | Vs fastest non-MP: MP -> comparator (seeds) | Saved | Saved % | Vs SiLU+AdamW: MP -> AdamW (seeds) | Saved | Saved % |
| ---: | ---: | --- | ---: | ---: | --- | ---: | ---: |
| 4.40 | 83.6M | 83.6M -> 87.9M (3/3) | 4.4M | 5.0% | 83.6M -> 102.1M (3/3) | 18.6M | 18.2% |
| 4.30 | 112.5M | 112.5M -> 113.0M (3/3) | 0.5M | 0.5% | 112.5M -> 132.2M (3/3) | 19.7M | 14.9% |
| 4.20 | 143.1M | 143.1M -> 148.5M (3/3) | 5.5M | 3.7% | 143.1M -> 173.1M (3/3) | 30.0M | 17.4% |
| 4.10 | 186.8M | 186.8M -> 196.6M (3/3) | 9.8M | 5.0% | 186.8M -> 241.4M (3/3) | 54.6M | 22.6% |
| 4.05 | 214.6M | 214.6M -> 232.1M (3/3) | 17.5M | 7.5% | not reached (0/3) | not reached | n/a |
| 4.00 | 252.3M | 247.4M -> 285.9M (2/3) | 38.5M | 13.5% | not reached (0/3) | not reached | n/a |

### Dolma-sample

Tracked package: `results/iclr26_e2_dolma_sample_2026_06_17/`.

Completed: 2026-06-17. Manifest rows `375-419` define the full Dolma-sample E2 M0/300M cell: 3 seeds x 15 fixed methods. All 45 paper-facing rows have final eval at step `9150`.

Each row uses `32768` global tokens/step for about `299.8M` train tokens. Validation uses the E2 Dolma-sample slice from the manifest: `val_skip_tokens=610000000`, `val_tokens=8000000`, `eval_interval=50`.
MatrixPolicy entries use the accepted safe-speed replacement JSONL rows for the same method and seed; the `row` column remains the matched main-manifest E2 row, while `source_phase`/`source_row_id` record the actual timed run.

#### Final Validation Loss

| Method | Final val loss mean +/- sample std | Min | Max | Notes |
| --- | ---: | ---: | ---: | --- |
| rlb_matrixpolicy_original | 3.808954 +/- 0.006442 | 3.801525 | 3.813003 |  |
| rlb_lion | 3.842503 +/- 0.009333 | 3.832392 | 3.850787 |  |
| silu_lion | 3.847523 +/- 0.009363 | 3.836884 | 3.854513 |  |
| rlb_muon | 3.848206 +/- 0.008937 | 3.838500 | 3.856095 |  |
| silu_muon | 3.858114 +/- 0.010066 | 3.846854 | 3.866242 |  |
| silu_adamw | 3.903690 +/- 0.009091 | 3.893635 | 3.911328 |  |
| rlb_adamw | 3.906369 +/- 0.007279 | 3.898687 | 3.913164 |  |
| rlb_soap | 3.920517 +/- 0.011663 | 3.909389 | 3.932650 |  |
| silu_soap | 3.956834 +/- 0.009319 | 3.946098 | 3.962839 |  |
| rlb_schedulefree | 4.205442 +/- 0.010523 | 4.193336 | 4.212402 |  |
| silu_schedulefree | 4.215105 +/- 0.005319 | 4.210405 | 4.220879 |  |
| silu_came | 4.249166 +/- 0.026981 | 4.223768 | 4.277492 |  |
| rlb_came | 4.285696 +/- 0.038229 | 4.241558 | 4.308310 |  |
| rlb_ademamix | nan/diverged | nan | nan | 3 diverged/non-finite seeds |
| silu_ademamix | nan/diverged | nan | nan | 3 diverged/non-finite seeds |

MatrixPolicy is best on all three Dolma-sample E2 seeds. Mean final val loss is `3.808954 +/- 0.006442`; the next-best aggregate methods are `rlb_lion` at `3.842503 +/- 0.009333`, `silu_lion` at `3.847523 +/- 0.009363`, `rlb_muon` at `3.848206 +/- 0.008937`.

#### Per-Seed MatrixPolicy Gap

| Seed | MatrixPolicy final loss | Best non-MP method | Best non-MP final loss | Gap |
| ---: | ---: | --- | ---: | ---: |
| 1337 | 3.813003 | rlb_lion | 3.850787 | 0.037784 |
| 2027 | 3.812334 | rlb_lion | 3.844330 | 0.031996 |
| 3407 | 3.801525 | rlb_lion | 3.832392 | 0.030867 |

#### Runtime Summary

`summary.total_seconds` is training-harness wall time for the manifest row. It excludes Slurm queue wait, dependency wait, token-cache construction, extension compilation, and launcher overhead.

| Method | Runs | Mean runtime | Std | Range | Mean s/step | Mean tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| silu_schedulefree | 3 | 60.3 min | 1.9 min | 58.7-62.4 min | 0.3777 | 86810.7 |
| silu_lion | 3 | 62.8 min | 5.2 min | 58.5-68.5 min | 0.3942 | 83530.1 |
| silu_adamw | 3 | 63.5 min | 6.1 min | 58.4-70.3 min | 0.3991 | 82647.8 |
| silu_came | 3 | 65.6 min | 2.3 min | 63.7-68.2 min | 0.4116 | 79666.0 |
| rlb_matrixpolicy_original | 3 | 69.2 min | 3.7 min | 64.9-71.4 min | 0.4278 | 76744.9 |
| silu_muon | 3 | 78.0 min | 21.0 min | 64.2-102.2 min | 0.4937 | 69498.1 |
| rlb_schedulefree | 3 | 78.7 min | 1.7 min | 76.8-80.2 min | 0.4858 | 67481.3 |
| silu_soap | 3 | 79.6 min | 11.8 min | 71.5-93.2 min | 0.5025 | 66102.2 |
| silu_ademamix | 3 | 79.7 min | 12.5 min | 70.9-94.0 min | 0.4279 | 78310.3 |
| rlb_came | 3 | 84.5 min | 1.8 min | 82.8-86.4 min | 0.5235 | 62615.8 |
| rlb_lion | 3 | 85.9 min | 10.7 min | 74.4-95.5 min | 0.5329 | 62238.9 |
| rlb_adamw | 3 | 86.0 min | 9.7 min | 76.3-95.7 min | 0.5332 | 62059.0 |
| rlb_soap | 3 | 87.9 min | 10.6 min | 76.7-97.6 min | 0.5462 | 60664.4 |
| rlb_muon | 3 | 102.6 min | 11.5 min | 91.5-114.5 min | 0.6406 | 51656.8 |
| rlb_ademamix | 3 | 112.1 min | 16.7 min | 100.5-131.2 min | 0.5596 | 60279.4 |

#### Token-To-Target Savings

This table asks how many training tokens were needed to first reach a validation-loss threshold. It does not change the fixed-budget protocol; all completed runs still trained to about `299.8M` tokens. The readout uses the native eval cadence of 50 steps, or `1.64M` tokens.

`Second-best` means the fastest non-MatrixPolicy method to reach the target within the same seed. `AdamW` means the standard `silu_adamw` row. Savings and proportions are computed only on seeds where both MatrixPolicy and the comparator reached the target, so the comparison column explicitly shows `MP -> comparator` tokens and the shared seed count.

| Target loss | MP all-hit mean | Vs fastest non-MP: MP -> comparator (seeds) | Saved | Saved % | Vs SiLU+AdamW: MP -> AdamW (seeds) | Saved | Saved % |
| ---: | ---: | --- | ---: | ---: | --- | ---: | ---: |
| 4.20 | 93.9M | 93.9M -> 94.5M (3/3) | 0.5M | 0.6% | 93.9M -> 110.9M (3/3) | 16.9M | 15.3% |
| 4.10 | 123.4M | 123.4M -> 124.0M (3/3) | 0.5M | 0.4% | 123.4M -> 145.3M (3/3) | 21.8M | 15.0% |
| 4.00 | 158.9M | 158.9M -> 163.8M (3/3) | 4.9M | 3.0% | 158.9M -> 195.0M (3/3) | 36.0M | 18.5% |
| 3.95 | 181.9M | 181.9M -> 190.6M (3/3) | 8.7M | 4.6% | 181.9M -> 232.7M (3/3) | 50.8M | 21.8% |
| 3.90 | 209.7M | 209.7M -> 223.9M (3/3) | 14.2M | 6.3% | 204.8M -> 283.4M (1/3) | 78.6M | 27.7% |
| 3.85 | 246.3M | 244.9M -> 276.9M (2/3) | 31.9M | 11.5% | not reached (0/3) | not reached | n/a |
| 3.82 | 280.2M | not reached (0/3) | not reached | n/a | not reached (0/3) | not reached | n/a |

### C4

Tracked package: `results/iclr26_e2_c4_2026_06_19/`.

Completed: 2026-06-19. Manifest rows `420-464` define the full C4 E2 M0/300M cell: 3 seeds x 15 fixed methods. All 45 paper-facing rows have final eval at step `9150`.

Each row uses `32768` global tokens/step for about `299.8M` train tokens. Validation uses the E2 C4 slice from the manifest: `val_skip_tokens=0`, `val_tokens=8000000`, `eval_interval=50`.
MatrixPolicy entries use the accepted safe-speed replacement JSONL rows for the same method and seed; the `row` column remains the matched main-manifest E2 row, while `source_phase`/`source_row_id` record the actual timed run.

#### Final Validation Loss

| Method | Final val loss mean +/- sample std | Min | Max | Notes |
| --- | ---: | ---: | ---: | --- |
| rlb_matrixpolicy_original | 3.883021 +/- 0.014134 | 3.872143 | 3.898997 |  |
| rlb_muon | 3.915858 +/- 0.016066 | 3.902808 | 3.933801 |  |
| rlb_lion | 3.919576 +/- 0.014201 | 3.907313 | 3.935135 |  |
| silu_lion | 3.921326 +/- 0.010538 | 3.913904 | 3.933388 |  |
| silu_muon | 3.925105 +/- 0.013434 | 3.911359 | 3.938204 |  |
| silu_adamw | 3.981105 +/- 0.012752 | 3.969709 | 3.994878 |  |
| rlb_adamw | 3.981627 +/- 0.011081 | 3.973570 | 3.994264 |  |
| rlb_soap | 4.005075 +/- 0.009183 | 3.999109 | 4.015650 |  |
| silu_soap | 4.034903 +/- 0.010776 | 4.027885 | 4.047310 |  |
| rlb_schedulefree | 4.303756 +/- 0.011329 | 4.294799 | 4.316491 |  |
| silu_schedulefree | 4.316317 +/- 0.010736 | 4.308978 | 4.328640 |  |
| silu_came | 4.329752 +/- 0.014752 | 4.314537 | 4.343993 |  |
| rlb_came | 4.363289 +/- 0.062600 | 4.315578 | 4.434171 |  |
| rlb_ademamix | nan/diverged | nan | nan | 3 diverged/non-finite seeds |
| silu_ademamix | nan/diverged | nan | nan | 3 diverged/non-finite seeds |

MatrixPolicy is best on all three C4 E2 seeds. Mean final val loss is `3.883021 +/- 0.014134`; the next-best aggregate methods are `rlb_muon` at `3.915858 +/- 0.016066`, `rlb_lion` at `3.919576 +/- 0.014201`, `silu_lion` at `3.921326 +/- 0.010538`.

#### Per-Seed MatrixPolicy Gap

| Seed | MatrixPolicy final loss | Best non-MP method | Best non-MP final loss | Gap |
| ---: | ---: | --- | ---: | ---: |
| 1337 | 3.877924 | rlb_muon | 3.910965 | 0.033041 |
| 2027 | 3.898997 | silu_lion | 3.933388 | 0.034391 |
| 3407 | 3.872143 | rlb_muon | 3.902808 | 0.030665 |

#### Runtime Summary

`summary.total_seconds` is training-harness wall time for the manifest row. It excludes Slurm queue wait, dependency wait, token-cache construction, extension compilation, and launcher overhead.

| Method | Runs | Mean runtime | Std | Range | Mean s/step | Mean tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| rlb_matrixpolicy_original | 3 | 67.0 min | 3.7 min | 64.8-71.3 min | 0.4141 | 79286.3 |
| silu_lion | 3 | 72.0 min | 15.9 min | 53.7-81.6 min | 0.4555 | 74962.8 |
| silu_adamw | 3 | 73.1 min | 13.1 min | 58.0-81.0 min | 0.4623 | 72773.1 |
| silu_schedulefree | 3 | 76.3 min | 12.4 min | 62.0-84.0 min | 0.4827 | 69317.2 |
| silu_muon | 3 | 78.1 min | 16.1 min | 59.6-88.2 min | 0.4937 | 68698.4 |
| silu_came | 3 | 80.7 min | 12.1 min | 66.7-88.0 min | 0.5119 | 65146.4 |
| silu_soap | 3 | 87.2 min | 12.3 min | 73.0-94.6 min | 0.5517 | 60276.0 |
| rlb_adamw | 3 | 87.3 min | 19.6 min | 74.8-109.9 min | 0.5431 | 62415.6 |
| rlb_lion | 3 | 90.0 min | 11.2 min | 77.0-97.0 min | 0.5599 | 59256.4 |
| rlb_schedulefree | 3 | 90.3 min | 10.4 min | 78.4-97.8 min | 0.5619 | 58941.4 |
| silu_ademamix | 3 | 90.7 min | 11.4 min | 77.6-98.5 min | 0.4792 | 69794.4 |
| rlb_muon | 3 | 95.6 min | 11.4 min | 82.4-102.7 min | 0.5945 | 55719.1 |
| rlb_soap | 3 | 96.3 min | 16.0 min | 79.3-111.2 min | 0.6018 | 55635.6 |
| rlb_came | 3 | 101.4 min | 16.6 min | 83.8-116.8 min | 0.6350 | 52682.7 |
| rlb_ademamix | 3 | 114.3 min | 23.2 min | 90.0-136.3 min | 0.6055 | 55622.2 |

#### Token-To-Target Savings

This table asks how many training tokens were needed to first reach a validation-loss threshold. It does not change the fixed-budget protocol; all completed runs still trained to about `299.8M` tokens. The readout uses the native eval cadence of 50 steps, or `1.64M` tokens.

`Second-best` means the fastest non-MatrixPolicy method to reach the target within the same seed. `AdamW` means the standard `silu_adamw` row. Savings and proportions are computed only on seeds where both MatrixPolicy and the comparator reached the target, so the comparison column explicitly shows `MP -> comparator` tokens and the shared seed count.

| Target loss | MP all-hit mean | Vs fastest non-MP: MP -> comparator (seeds) | Saved | Saved % | Vs SiLU+AdamW: MP -> AdamW (seeds) | Saved | Saved % |
| ---: | ---: | --- | ---: | ---: | --- | ---: | ---: |
| 4.40 | 67.7M | 67.7M -> 72.6M (3/3) | 4.9M | 6.8% | 67.7M -> 86.8M (3/3) | 19.1M | 22.0% |
| 4.30 | 90.7M | 90.7M -> 93.4M (3/3) | 2.7M | 2.9% | 90.7M -> 108.1M (3/3) | 17.5M | 16.2% |
| 4.20 | 118.5M | 118.5M -> 119.1M (3/3) | 0.5M | 0.5% | 118.5M -> 139.8M (3/3) | 21.3M | 15.2% |
| 4.10 | 151.3M | 151.3M -> 156.2M (3/3) | 4.9M | 3.1% | 151.3M -> 185.1M (3/3) | 33.9M | 18.3% |
| 4.05 | 171.5M | 171.5M -> 179.1M (3/3) | 7.6M | 4.3% | 171.5M -> 216.3M (3/3) | 44.8M | 20.7% |
| 4.00 | 196.1M | 196.1M -> 207.5M (3/3) | 11.5M | 5.5% | 196.1M -> 264.9M (3/3) | 68.8M | 26.0% |

### E2 Runtime Tables

#### DCLM

| Combo | Runs | Mean runtime | Std | Range | Mean s/step | Mean tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RLB+MatrixPolicy | 3 | 69.6 min | 2.8 min | 66.4 min-71.3 min | 0.4304 | 76212.4 |
| SiLU+AdamW | 3 | 70.9 min | 10.0 min | 61.1 min-81.2 min | 0.4478 | 74246.1 |
| RLB+AdamW | 3 | 83.6 min | 10.9 min | 77.3 min-96.3 min | 0.5188 | 63911.7 |
| SiLU+Muon | 3 | 71.4 min | 14.8 min | 59.4 min-87.9 min | 0.4497 | 74940.2 |
| RLB+Muon | 3 | 89.0 min | 11.4 min | 82.3 min-102.1 min | 0.5530 | 59894.1 |
| SiLU+Lion | 3 | 67.6 min | 12.0 min | 60.5 min-81.5 min | 0.4265 | 78442.0 |
| RLB+Lion | 3 | 83.6 min | 11.5 min | 76.9 min-96.9 min | 0.5184 | 64027.8 |
| SiLU+SOAP | 3 | 86.9 min | 12.3 min | 72.8 min-94.5 min | 0.5487 | 60587.7 |
| RLB+SOAP | 3 | 85.6 min | 11.0 min | 79.2 min-98.2 min | 0.5310 | 62405.0 |
| SiLU+ADeMaMix | 3 | 85.3 min | 18.1 min | 64.5 min-96.1 min | 0.4823 | 69300.4 |
| RLB+ADeMaMix | 3 | 107.9 min | 12.1 min | 100.8 min-121.9 min | 0.5201 | 63823.4 |
| SiLU+CAME | 3 | 80.2 min | 11.7 min | 66.7 min-87.2 min | 0.5085 | 65529.0 |
| RLB+CAME | 3 | 90.0 min | 10.6 min | 83.9 min-102.3 min | 0.5604 | 59034.6 |
| SiLU+ScheduleFree | 3 | 68.9 min | 11.9 min | 61.6 min-82.6 min | 0.4343 | 76941.0 |
| RLB+ScheduleFree | 3 | 78.3 min | 0.1 min | 78.1 min-78.4 min | 0.4835 | 67778.1 |

#### FineWeb-Edu

| Combo | Runs | Mean runtime | Std | Range | Mean s/step | Mean tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RLB+MatrixPolicy | 3 | 69.6 min | 2.9 min | 66.3 min-71.4 min | 0.4304 | 76215.7 |
| SiLU+AdamW | 3 | 75.2 min | 14.0 min | 61.4 min-89.4 min | 0.4754 | 70683.2 |
| RLB+AdamW | 3 | 100.6 min | 8.5 min | 95.7 min-110.4 min | 0.6304 | 52246.8 |
| SiLU+Muon | 3 | 72.8 min | 12.9 min | 63.8 min-87.6 min | 0.4585 | 72920.4 |
| RLB+Muon | 3 | 102.9 min | 3.1 min | 101.1 min-106.5 min | 0.6429 | 51009.0 |
| SiLU+Lion | 3 | 74.5 min | 11.5 min | 61.2 min-81.1 min | 0.4709 | 70923.0 |
| RLB+Lion | 3 | 91.2 min | 12.7 min | 77.0 min-101.3 min | 0.5685 | 58526.8 |
| SiLU+SOAP | 3 | 86.8 min | 12.0 min | 72.9 min-93.8 min | 0.5475 | 60685.2 |
| RLB+SOAP | 3 | 93.5 min | 12.6 min | 79.3 min-103.3 min | 0.5832 | 56995.4 |
| SiLU+ADeMaMix | 3 | 88.5 min | 10.7 min | 76.5 min-97.0 min | 0.4762 | 70175.4 |
| RLB+ADeMaMix | 3 | 108.2 min | 14.2 min | 98.8 min-124.5 min | 0.5356 | 62475.8 |
| SiLU+CAME | 3 | 80.2 min | 11.7 min | 66.7 min-87.0 min | 0.5079 | 65602.6 |
| RLB+CAME | 3 | 92.6 min | 15.0 min | 83.9 min-110.0 min | 0.5774 | 57751.3 |
| SiLU+ScheduleFree | 3 | 75.5 min | 11.9 min | 61.8 min-82.4 min | 0.4781 | 69904.1 |
| RLB+ScheduleFree | 3 | 86.6 min | 14.4 min | 78.3 min-103.3 min | 0.5388 | 61973.9 |

#### FineWeb

| Combo | Runs | Mean runtime | Std | Range | Mean s/step | Mean tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RLB+MatrixPolicy | 3 | 67.1 min | 3.5 min | 65.0 min-71.2 min | 0.4146 | 79179.4 |
| SiLU+AdamW | 3 | 76.4 min | 18.9 min | 60.2 min-97.1 min | 0.4840 | 70588.2 |
| RLB+AdamW | 3 | 76.3 min | 1.3 min | 74.8 min-77.2 min | 0.4700 | 69737.2 |
| SiLU+Muon | 3 | 69.7 min | 6.2 min | 65.6 min-76.8 min | 0.4385 | 75078.6 |
| RLB+Muon | 3 | 85.0 min | 14.3 min | 74.0 min-101.2 min | 0.5263 | 63461.5 |
| SiLU+Lion | 3 | 64.6 min | 6.3 min | 61.0 min-71.9 min | 0.4066 | 81108.9 |
| RLB+Lion | 3 | 79.6 min | 14.1 min | 68.6 min-95.5 min | 0.4920 | 68070.8 |
| SiLU+SOAP | 3 | 72.3 min | 0.6 min | 71.7 min-72.9 min | 0.4571 | 71696.4 |
| RLB+SOAP | 3 | 85.4 min | 11.1 min | 76.9 min-97.9 min | 0.5299 | 62572.7 |
| SiLU+ADeMaMix | 3 | 150.6 min | 143.3 min | 62.6 min-316.0 min | 0.9251 | 60189.7 |
| RLB+ADeMaMix | 3 | 104.0 min | 12.7 min | 96.0 min-118.7 min | 0.5141 | 64685.3 |
| SiLU+CAME | 3 | 147.2 min | 139.2 min | 66.6 min-307.9 min | 0.9476 | 57383.1 |
| RLB+CAME | 3 | 91.1 min | 10.1 min | 82.8 min-102.3 min | 0.5662 | 58393.3 |
| SiLU+ScheduleFree | 3 | 61.7 min | 0.4 min | 61.3 min-62.1 min | 0.3875 | 84557.5 |
| RLB+ScheduleFree | 3 | 85.6 min | 10.0 min | 77.3 min-96.7 min | 0.5304 | 62408.2 |

#### Dolma-sample

| Combo | Runs | Mean runtime | Std | Range | Mean s/step | Mean tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RLB+MatrixPolicy | 3 | 69.2 min | 3.7 min | 64.9 min-71.4 min | 0.4278 | 76744.9 |
| SiLU+AdamW | 3 | 63.5 min | 6.1 min | 58.4 min-70.3 min | 0.3991 | 82647.8 |
| RLB+AdamW | 3 | 86.0 min | 9.7 min | 76.3 min-95.7 min | 0.5332 | 62059.0 |
| SiLU+Muon | 3 | 78.0 min | 21.0 min | 64.2 min-102.2 min | 0.4937 | 69498.1 |
| RLB+Muon | 3 | 102.6 min | 11.5 min | 91.5 min-114.5 min | 0.6406 | 51656.8 |
| SiLU+Lion | 3 | 62.8 min | 5.2 min | 58.5 min-68.5 min | 0.3942 | 83530.1 |
| RLB+Lion | 3 | 85.9 min | 10.7 min | 74.4 min-95.5 min | 0.5329 | 62238.9 |
| SiLU+SOAP | 3 | 79.6 min | 11.8 min | 71.5 min-93.2 min | 0.5025 | 66102.2 |
| RLB+SOAP | 3 | 87.9 min | 10.6 min | 76.7 min-97.6 min | 0.5462 | 60664.4 |
| SiLU+ADeMaMix | 3 | 79.7 min | 12.5 min | 70.9 min-94.0 min | 0.4279 | 78310.3 |
| RLB+ADeMaMix | 3 | 112.1 min | 16.7 min | 100.5 min-131.2 min | 0.5596 | 60279.4 |
| SiLU+CAME | 3 | 65.6 min | 2.3 min | 63.7 min-68.2 min | 0.4116 | 79666.0 |
| RLB+CAME | 3 | 84.5 min | 1.8 min | 82.8 min-86.4 min | 0.5235 | 62615.8 |
| SiLU+ScheduleFree | 3 | 60.3 min | 1.9 min | 58.7 min-62.4 min | 0.3777 | 86810.7 |
| RLB+ScheduleFree | 3 | 78.7 min | 1.7 min | 76.8 min-80.2 min | 0.4858 | 67481.3 |

#### C4

| Combo | Runs | Mean runtime | Std | Range | Mean s/step | Mean tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RLB+MatrixPolicy | 3 | 67.0 min | 3.7 min | 64.8 min-71.3 min | 0.4141 | 79286.3 |
| SiLU+AdamW | 3 | 73.1 min | 13.1 min | 58.0 min-81.0 min | 0.4623 | 72773.1 |
| RLB+AdamW | 3 | 87.3 min | 19.6 min | 74.8 min-109.9 min | 0.5431 | 62415.6 |
| SiLU+Muon | 3 | 78.1 min | 16.1 min | 59.6 min-88.2 min | 0.4937 | 68698.4 |
| RLB+Muon | 3 | 95.6 min | 11.4 min | 82.4 min-102.7 min | 0.5945 | 55719.1 |
| SiLU+Lion | 3 | 72.0 min | 15.9 min | 53.7 min-81.6 min | 0.4555 | 74962.8 |
| RLB+Lion | 3 | 90.0 min | 11.2 min | 77.0 min-97.0 min | 0.5599 | 59256.4 |
| SiLU+SOAP | 3 | 87.2 min | 12.3 min | 73.0 min-94.6 min | 0.5517 | 60276.0 |
| RLB+SOAP | 3 | 96.3 min | 16.0 min | 79.3 min-111.2 min | 0.6018 | 55635.6 |
| SiLU+ADeMaMix | 3 | 90.7 min | 11.4 min | 77.6 min-98.5 min | 0.4792 | 69794.4 |
| RLB+ADeMaMix | 3 | 114.3 min | 23.2 min | 90.0 min-136.3 min | 0.6055 | 55622.2 |
| SiLU+CAME | 3 | 80.7 min | 12.1 min | 66.7 min-88.0 min | 0.5119 | 65146.4 |
| RLB+CAME | 3 | 101.4 min | 16.6 min | 83.8 min-116.8 min | 0.6350 | 52682.7 |
| SiLU+ScheduleFree | 3 | 76.3 min | 12.4 min | 62.0 min-84.0 min | 0.4827 | 69317.2 |
| RLB+ScheduleFree | 3 | 90.3 min | 10.4 min | 78.4 min-97.8 min | 0.5619 | 58941.4 |

### E2 Dense Curves And Checkpoint Tables

Completed E2 M0/300M datasets: DCLM, FineWeb-Edu, FineWeb, Dolma-sample, and C4. Figures use every native JSONL log point from step 500 through 9150. Validation curves use every 50-step eval; training-loss curves use every 10-step train log. Shaded bands are mean +/- 1 sample std over three seeds.

MatrixPolicy curves use the accepted safe-speed replacement JSONL rows when the generator is called with `--matrixpolicy-manifest`; all other methods use the main E2 manifest rows.

Final validation-loss overview across completed E2 datasets. Lower is better; cells are mean +/- sample std over three seeds.

| Method | DCLM final | FineWeb-Edu final | FineWeb final | Dolma-sample final | C4 final |
| --- | ---: | ---: | ---: | ---: | ---: |
| MatrixPolicy | 3.9561 +/- 0.0308 | 3.7078 +/- 0.0187 | 3.9649 +/- 0.0095 | 3.8090 +/- 0.0064 | 3.8830 +/- 0.0141 |
| RLB+AdamW | 4.0529 +/- 0.0282 | 3.8069 +/- 0.0176 | 4.0629 +/- 0.0098 | 3.9064 +/- 0.0073 | 3.9816 +/- 0.0111 |
| SiLU+AdamW | 4.0493 +/- 0.0275 | 3.8035 +/- 0.0182 | 4.0612 +/- 0.0101 | 3.9037 +/- 0.0091 | 3.9811 +/- 0.0128 |
| RLB+Lion | 3.9943 +/- 0.0301 | 3.7451 +/- 0.0214 | 4.0014 +/- 0.0128 | 3.8425 +/- 0.0093 | 3.9196 +/- 0.0142 |
| SiLU+Lion | 3.9934 +/- 0.0230 | 3.7440 +/- 0.0208 | 4.0015 +/- 0.0085 | 3.8475 +/- 0.0094 | 3.9213 +/- 0.0105 |
| RLB+SOAP | 4.0768 +/- 0.0403 | 3.8301 +/- 0.0199 | 4.0841 +/- 0.0079 | 3.9205 +/- 0.0117 | 4.0051 +/- 0.0092 |
| SiLU+SOAP | 4.0964 +/- 0.0300 | 3.8629 +/- 0.0201 | 4.1139 +/- 0.0101 | 3.9568 +/- 0.0093 | 4.0349 +/- 0.0108 |
| RLB+Muon | 3.9935 +/- 0.0296 | 3.7382 +/- 0.0210 | 4.0012 +/- 0.0114 | 3.8482 +/- 0.0089 | 3.9159 +/- 0.0161 |
| SiLU+Muon | 3.9973 +/- 0.0305 | 3.7454 +/- 0.0170 | 4.0066 +/- 0.0128 | 3.8581 +/- 0.0101 | 3.9251 +/- 0.0134 |
| RLB+ScheduleFree | 4.3563 +/- 0.0332 | 4.1365 +/- 0.0217 | 4.3814 +/- 0.0093 | 4.2054 +/- 0.0105 | 4.3038 +/- 0.0113 |
| SiLU+ScheduleFree | 4.3657 +/- 0.0298 | 4.1559 +/- 0.0238 | 4.3979 +/- 0.0106 | 4.2151 +/- 0.0053 | 4.3163 +/- 0.0107 |
| RLB+CAME | 4.4503 +/- 0.0340 | 4.2203 +/- 0.0360 | 4.4732 +/- 0.0011 | 4.2857 +/- 0.0382 | 4.3633 +/- 0.0626 |
| SiLU+CAME | 4.3682 +/- 0.0226 | 4.1503 +/- 0.0211 | 4.4060 +/- 0.0189 | 4.2492 +/- 0.0270 | 4.3298 +/- 0.0148 |
| RLB+ADeMaMix | -- | -- | -- | -- | -- |
| SiLU+ADeMaMix | -- | -- | 1361.4141 +/- 0.0000 (n=1) | -- | -- |

#### DCLM

All-method view:

![DCLM E2 validation loss mean +/- std, all methods](results/iclr26_e2_figures/dclm_core_validation_loss_mean_std.svg)

![DCLM E2 validation PPL mean +/- std, all methods](results/iclr26_e2_figures/dclm_core_validation_ppl_mean_std.svg)

![DCLM E2 training loss mean +/- std, all methods](results/iclr26_e2_figures/dclm_core_training_loss_mean_std.svg)

Clean comparison view:

![DCLM E2 validation loss mean +/- std, clean comparison](results/iclr26_e2_figures/dclm_clean_validation_loss_mean_std.svg)

![DCLM E2 validation PPL mean +/- std, clean comparison](results/iclr26_e2_figures/dclm_clean_validation_ppl_mean_std.svg)

![DCLM E2 training loss mean +/- std, clean comparison](results/iclr26_e2_figures/dclm_clean_training_loss_mean_std.svg)

DCLM E2 validation-loss checkpoint table, mean +/- sample std:

| Method | 1000 | 2000 | 4000 | 6000 | 8000 | 9150 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MatrixPolicy | 4.8529 +/- 0.0421 | 4.4386 +/- 0.0299 | 4.2040 +/- 0.0285 | 4.0603 +/- 0.0314 | 3.9779 +/- 0.0303 | 3.9561 +/- 0.0308 |
| RLB+AdamW | 4.9827 +/- 0.0317 | 4.5493 +/- 0.0274 | 4.2708 +/- 0.0276 | 4.1410 +/- 0.0304 | 4.0705 +/- 0.0278 | 4.0529 +/- 0.0282 |
| SiLU+AdamW | 4.9903 +/- 0.0383 | 4.5538 +/- 0.0272 | 4.2691 +/- 0.0242 | 4.1383 +/- 0.0272 | 4.0667 +/- 0.0268 | 4.0493 +/- 0.0275 |
| RLB+Lion | 4.9162 +/- 0.0369 | 4.4816 +/- 0.0324 | 4.2186 +/- 0.0282 | 4.0858 +/- 0.0307 | 4.0130 +/- 0.0306 | 3.9943 +/- 0.0301 |
| SiLU+Lion | 4.9565 +/- 0.0352 | 4.5005 +/- 0.0231 | 4.2217 +/- 0.0219 | 4.0876 +/- 0.0241 | 4.0126 +/- 0.0228 | 3.9934 +/- 0.0230 |
| RLB+SOAP | 5.0873 +/- 0.1169 | 4.6116 +/- 0.0673 | 4.3194 +/- 0.0553 | 4.1703 +/- 0.0414 | 4.0963 +/- 0.0407 | 4.0768 +/- 0.0403 |
| SiLU+SOAP | 5.1360 +/- 0.0184 | 4.6909 +/- 0.0570 | 4.3574 +/- 0.0355 | 4.1959 +/- 0.0296 | 4.1163 +/- 0.0290 | 4.0964 +/- 0.0300 |
| RLB+Muon | 5.1264 +/- 0.0284 | 4.5681 +/- 0.0266 | 4.2253 +/- 0.0291 | 4.0836 +/- 0.0301 | 4.0104 +/- 0.0297 | 3.9935 +/- 0.0296 |
| SiLU+Muon | 5.1356 +/- 0.0339 | 4.5702 +/- 0.0325 | 4.2298 +/- 0.0306 | 4.0879 +/- 0.0310 | 4.0158 +/- 0.0305 | 3.9973 +/- 0.0305 |
| RLB+ScheduleFree | 5.4308 +/- 0.0313 | 5.0115 +/- 0.0331 | 4.6380 +/- 0.0371 | 4.4610 +/- 0.0349 | 4.3814 +/- 0.0327 | 4.3563 +/- 0.0332 |
| SiLU+ScheduleFree | 5.4545 +/- 0.0291 | 5.0363 +/- 0.0325 | 4.6521 +/- 0.0334 | 4.4730 +/- 0.0313 | 4.3908 +/- 0.0298 | 4.3657 +/- 0.0298 |
| RLB+CAME | 5.5176 +/- 0.0411 | 5.1128 +/- 0.0298 | 4.7533 +/- 0.0260 | 4.5664 +/- 0.0297 | 4.4742 +/- 0.0332 | 4.4503 +/- 0.0340 |
| SiLU+CAME | 5.5228 +/- 0.0322 | 5.1088 +/- 0.0252 | 4.6850 +/- 0.0205 | 4.4791 +/- 0.0221 | 4.3907 +/- 0.0223 | 4.3682 +/- 0.0226 |
| RLB+ADeMaMix | 7522.7668 +/- 12918.9198 | -- | -- | -- | -- | -- |
| SiLU+ADeMaMix | 1709.6780 +/- 585.9580 | 2931159.8867 +/- 4767520.6794 | 11241.5732 +/- 0.0000 (n=1) | 196344.0312 +/- 0.0000 (n=1) | 34853138432.0000 +/- 0.0000 (n=1) | -- |

#### FineWeb-Edu

All-method view:

![FineWeb-Edu E2 validation loss mean +/- std, all methods](results/iclr26_e2_figures/fineweb_edu_core_validation_loss_mean_std.svg)

![FineWeb-Edu E2 validation PPL mean +/- std, all methods](results/iclr26_e2_figures/fineweb_edu_core_validation_ppl_mean_std.svg)

![FineWeb-Edu E2 training loss mean +/- std, all methods](results/iclr26_e2_figures/fineweb_edu_core_training_loss_mean_std.svg)

Clean comparison view:

![FineWeb-Edu E2 validation loss mean +/- std, clean comparison](results/iclr26_e2_figures/fineweb_edu_clean_validation_loss_mean_std.svg)

![FineWeb-Edu E2 validation PPL mean +/- std, clean comparison](results/iclr26_e2_figures/fineweb_edu_clean_validation_ppl_mean_std.svg)

![FineWeb-Edu E2 training loss mean +/- std, clean comparison](results/iclr26_e2_figures/fineweb_edu_clean_training_loss_mean_std.svg)

FineWeb-Edu E2 validation-loss checkpoint table, mean +/- sample std:

| Method | 1000 | 2000 | 4000 | 6000 | 8000 | 9150 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MatrixPolicy | 4.6980 +/- 0.0159 | 4.2436 +/- 0.0239 | 3.9878 +/- 0.0212 | 3.8271 +/- 0.0175 | 3.7342 +/- 0.0193 | 3.7078 +/- 0.0187 |
| RLB+AdamW | 4.8421 +/- 0.0247 | 4.3623 +/- 0.0256 | 4.0533 +/- 0.0217 | 3.9064 +/- 0.0178 | 3.8278 +/- 0.0189 | 3.8069 +/- 0.0176 |
| SiLU+AdamW | 4.8639 +/- 0.0168 | 4.3679 +/- 0.0200 | 4.0543 +/- 0.0197 | 3.9046 +/- 0.0184 | 3.8255 +/- 0.0186 | 3.8035 +/- 0.0182 |
| RLB+Lion | 4.7857 +/- 0.0153 | 4.2970 +/- 0.0278 | 3.9981 +/- 0.0250 | 3.8475 +/- 0.0214 | 3.7664 +/- 0.0217 | 3.7451 +/- 0.0214 |
| SiLU+Lion | 4.8099 +/- 0.0075 | 4.3046 +/- 0.0220 | 3.9982 +/- 0.0231 | 3.8478 +/- 0.0196 | 3.7659 +/- 0.0215 | 3.7440 +/- 0.0208 |
| RLB+SOAP | 4.8854 +/- 0.0289 | 5.1550 +/- 1.2456 | 4.0786 +/- 0.0068 | 5.7475 +/- 3.1597 | 3.8498 +/- 0.0276 | 3.8301 +/- 0.0199 |
| SiLU+SOAP | 5.0606 +/- 0.0646 | 4.5418 +/- 0.0216 | 4.1519 +/- 0.0262 | 3.9789 +/- 0.0195 | 3.8866 +/- 0.0212 | 3.8629 +/- 0.0201 |
| RLB+Muon | 4.9853 +/- 0.0142 | 4.3609 +/- 0.0198 | 3.9965 +/- 0.0210 | 3.8409 +/- 0.0206 | 3.7600 +/- 0.0216 | 3.7382 +/- 0.0210 |
| SiLU+Muon | 4.9953 +/- 0.0177 | 4.3557 +/- 0.0228 | 4.0009 +/- 0.0183 | 3.8475 +/- 0.0161 | 3.7663 +/- 0.0175 | 3.7454 +/- 0.0170 |
| RLB+ScheduleFree | 5.4400 +/- 0.0107 | 4.9005 +/- 0.0164 | 4.4358 +/- 0.0273 | 4.2460 +/- 0.0229 | 4.1624 +/- 0.0223 | 4.1365 +/- 0.0217 |
| SiLU+ScheduleFree | 5.4788 +/- 0.0185 | 4.9453 +/- 0.0231 | 4.4643 +/- 0.0258 | 4.2687 +/- 0.0247 | 4.1826 +/- 0.0246 | 4.1559 +/- 0.0238 |
| RLB+CAME | 5.5138 +/- 0.0158 | 4.9893 +/- 0.0190 | 4.5507 +/- 0.0308 | 4.3457 +/- 0.0328 | 4.2470 +/- 0.0364 | 4.2203 +/- 0.0360 |
| SiLU+CAME | 5.5271 +/- 0.0210 | 5.0004 +/- 0.0184 | 4.5080 +/- 0.0339 | 4.2755 +/- 0.0247 | 4.1761 +/- 0.0218 | 4.1503 +/- 0.0211 |
| RLB+ADeMaMix | 27698.5969 +/- 47596.7828 | 4962161216.0000 +/- 6009468964.3189 (n=2) | -- | -- | -- | -- |
| SiLU+ADeMaMix | 1388.9467 +/- 438.1649 | 929392.9245 +/- 1048745.5316 | 1370518.8750 +/- 0.0000 (n=1) | -- | -- | -- |

#### FineWeb

All-method view:

![FineWeb E2 validation loss mean +/- std, all methods](results/iclr26_e2_figures/fineweb_core_validation_loss_mean_std.svg)

![FineWeb E2 validation PPL mean +/- std, all methods](results/iclr26_e2_figures/fineweb_core_validation_ppl_mean_std.svg)

![FineWeb E2 training loss mean +/- std, all methods](results/iclr26_e2_figures/fineweb_core_training_loss_mean_std.svg)

Clean comparison view:

![FineWeb E2 validation loss mean +/- std, clean comparison](results/iclr26_e2_figures/fineweb_clean_validation_loss_mean_std.svg)

![FineWeb E2 validation PPL mean +/- std, clean comparison](results/iclr26_e2_figures/fineweb_clean_validation_ppl_mean_std.svg)

![FineWeb E2 training loss mean +/- std, clean comparison](results/iclr26_e2_figures/fineweb_clean_training_loss_mean_std.svg)

FineWeb E2 validation-loss checkpoint table, mean +/- sample std:

| Method | 1000 | 2000 | 4000 | 6000 | 8000 | 9150 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MatrixPolicy | 4.9064 +/- 0.0141 | 4.4819 +/- 0.0077 | 4.2340 +/- 0.0095 | 4.0778 +/- 0.0078 | 3.9895 +/- 0.0091 | 3.9649 +/- 0.0095 |
| RLB+AdamW | 5.0393 +/- 0.0164 | 4.5920 +/- 0.0064 | 4.2998 +/- 0.0082 | 4.1575 +/- 0.0082 | 4.0815 +/- 0.0090 | 4.0629 +/- 0.0098 |
| SiLU+AdamW | 5.0527 +/- 0.0192 | 4.6013 +/- 0.0095 | 4.2990 +/- 0.0101 | 4.1566 +/- 0.0097 | 4.0805 +/- 0.0098 | 4.0612 +/- 0.0101 |
| RLB+Lion | 4.9617 +/- 0.0095 | 4.5227 +/- 0.0115 | 4.2419 +/- 0.0147 | 4.0992 +/- 0.0131 | 4.0210 +/- 0.0131 | 4.0014 +/- 0.0128 |
| SiLU+Lion | 5.0084 +/- 0.0139 | 4.5399 +/- 0.0078 | 4.2458 +/- 0.0089 | 4.1009 +/- 0.0102 | 4.0221 +/- 0.0085 | 4.0015 +/- 0.0085 |
| RLB+SOAP | 5.1011 +/- 0.0489 | 4.6501 +/- 0.0149 | 4.3392 +/- 0.0139 | 4.1790 +/- 0.0052 | 4.1141 +/- 0.0255 | 4.0841 +/- 0.0079 |
| SiLU+SOAP | 5.2775 +/- 0.0437 | 4.7942 +/- 0.1152 | 4.4121 +/- 0.0185 | 4.2201 +/- 0.0089 | 4.1358 +/- 0.0106 | 4.1139 +/- 0.0101 |
| RLB+Muon | 5.1846 +/- 0.0121 | 4.5976 +/- 0.0155 | 4.2490 +/- 0.0133 | 4.0981 +/- 0.0112 | 4.0219 +/- 0.0106 | 4.0012 +/- 0.0114 |
| SiLU+Muon | 5.1938 +/- 0.0130 | 4.5962 +/- 0.0095 | 4.2524 +/- 0.0119 | 4.1027 +/- 0.0121 | 4.0266 +/- 0.0117 | 4.0066 +/- 0.0128 |
| RLB+ScheduleFree | 5.5429 +/- 0.0168 | 5.0898 +/- 0.0162 | 4.6658 +/- 0.0107 | 4.4862 +/- 0.0090 | 4.4064 +/- 0.0091 | 4.3814 +/- 0.0093 |
| SiLU+ScheduleFree | 5.5684 +/- 0.0117 | 5.1193 +/- 0.0137 | 4.6901 +/- 0.0124 | 4.5060 +/- 0.0105 | 4.4236 +/- 0.0103 | 4.3979 +/- 0.0106 |
| RLB+CAME | 5.6199 +/- 0.0164 | 5.1868 +/- 0.0100 | 4.7914 +/- 0.0037 | 4.5916 +/- 0.0004 | 4.4979 +/- 0.0010 | 4.4732 +/- 0.0011 |
| SiLU+CAME | 5.6395 +/- 0.0158 | 5.1935 +/- 0.0228 | 4.7432 +/- 0.0503 | 4.5220 +/- 0.0254 | 4.4301 +/- 0.0193 | 4.4060 +/- 0.0189 |
| RLB+ADeMaMix | 257.9755 +/- 250.1753 | 1633877367.1458 +/- 2533289655.0567 | -- | -- | -- | -- |
| SiLU+ADeMaMix | 1046.3267 +/- 385.1340 | 386.1834 +/- 120.1352 | 18508.4055 +/- 25038.3317 (n=2) | 1100.8079 +/- 0.0000 (n=1) | 2696.9573 +/- 0.0000 (n=1) | 1361.4141 +/- 0.0000 (n=1) |

#### Dolma-sample

All-method view:

![Dolma-sample E2 validation loss mean +/- std, all methods](results/iclr26_e2_figures/dolma_sample_core_validation_loss_mean_std.svg)

![Dolma-sample E2 validation PPL mean +/- std, all methods](results/iclr26_e2_figures/dolma_sample_core_validation_ppl_mean_std.svg)

![Dolma-sample E2 training loss mean +/- std, all methods](results/iclr26_e2_figures/dolma_sample_core_training_loss_mean_std.svg)

Clean comparison view:

![Dolma-sample E2 validation loss mean +/- std, clean comparison](results/iclr26_e2_figures/dolma_sample_clean_validation_loss_mean_std.svg)

![Dolma-sample E2 validation PPL mean +/- std, clean comparison](results/iclr26_e2_figures/dolma_sample_clean_validation_ppl_mean_std.svg)

![Dolma-sample E2 training loss mean +/- std, clean comparison](results/iclr26_e2_figures/dolma_sample_clean_training_loss_mean_std.svg)

Dolma-sample E2 validation-loss checkpoint table, mean +/- sample std:

| Method | 1000 | 2000 | 4000 | 6000 | 8000 | 9150 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MatrixPolicy | 4.7049 +/- 0.0027 | 4.3098 +/- 0.0098 | 4.0731 +/- 0.0100 | 3.9197 +/- 0.0083 | 3.8321 +/- 0.0063 | 3.8090 +/- 0.0064 |
| RLB+AdamW | 4.8388 +/- 0.0089 | 4.4218 +/- 0.0075 | 4.1373 +/- 0.0091 | 3.9965 +/- 0.0085 | 3.9246 +/- 0.0078 | 3.9064 +/- 0.0073 |
| SiLU+AdamW | 4.8563 +/- 0.0014 | 4.4240 +/- 0.0099 | 4.1350 +/- 0.0106 | 3.9963 +/- 0.0097 | 3.9227 +/- 0.0091 | 3.9037 +/- 0.0091 |
| RLB+Lion | 4.7699 +/- 0.0206 | 4.3509 +/- 0.0139 | 4.0789 +/- 0.0131 | 3.9381 +/- 0.0103 | 3.8620 +/- 0.0103 | 3.8425 +/- 0.0093 |
| SiLU+Lion | 4.8041 +/- 0.0041 | 4.3710 +/- 0.0090 | 4.0847 +/- 0.0104 | 3.9425 +/- 0.0101 | 3.8670 +/- 0.0094 | 3.8475 +/- 0.0094 |
| RLB+SOAP | 4.9194 +/- 0.0832 | 4.4464 +/- 0.0075 | 4.1718 +/- 0.0153 | 4.0179 +/- 0.0151 | 3.9404 +/- 0.0129 | 3.9205 +/- 0.0117 |
| SiLU+SOAP | 5.0463 +/- 0.0622 | 4.5589 +/- 0.0265 | 4.2314 +/- 0.0103 | 4.0647 +/- 0.0121 | 3.9780 +/- 0.0081 | 3.9568 +/- 0.0093 |
| RLB+Muon | 4.9890 +/- 0.0053 | 4.4319 +/- 0.0059 | 4.0907 +/- 0.0101 | 3.9421 +/- 0.0084 | 3.8673 +/- 0.0091 | 3.8482 +/- 0.0089 |
| SiLU+Muon | 4.9932 +/- 0.0054 | 4.4333 +/- 0.0060 | 4.1013 +/- 0.0120 | 3.9535 +/- 0.0103 | 3.8776 +/- 0.0095 | 3.8581 +/- 0.0101 |
| RLB+ScheduleFree | 5.3286 +/- 0.0067 | 4.8747 +/- 0.0082 | 4.4745 +/- 0.0132 | 4.3037 +/- 0.0121 | 4.2285 +/- 0.0111 | 4.2054 +/- 0.0105 |
| SiLU+ScheduleFree | 5.3512 +/- 0.0087 | 4.8985 +/- 0.0056 | 4.4895 +/- 0.0060 | 4.3161 +/- 0.0042 | 4.2391 +/- 0.0055 | 4.2151 +/- 0.0053 |
| RLB+CAME | 5.3901 +/- 0.0074 | 4.9666 +/- 0.0087 | 4.5937 +/- 0.0186 | 4.4014 +/- 0.0335 | 4.3098 +/- 0.0387 | 4.2857 +/- 0.0382 |
| SiLU+CAME | 5.4110 +/- 0.0084 | 4.9728 +/- 0.0149 | 4.5654 +/- 0.0444 | 4.3689 +/- 0.0414 | 4.2734 +/- 0.0294 | 4.2492 +/- 0.0270 |
| RLB+ADeMaMix | 17928.5126 +/- 15450.5860 | 2482927104.0000 +/- 2466098821.8411 (n=2) | -- | -- | -- | -- |
| SiLU+ADeMaMix | 1127.4188 +/- 977.4469 | 1608323.0966 +/- 2563013.6932 | 70147874816.0000 +/- 0.0000 (n=1) | -- | -- | -- |

#### C4

All-method view:

![C4 E2 validation loss mean +/- std, all methods](results/iclr26_e2_figures/c4_en_core_validation_loss_mean_std.svg)

![C4 E2 validation PPL mean +/- std, all methods](results/iclr26_e2_figures/c4_en_core_validation_ppl_mean_std.svg)

![C4 E2 training loss mean +/- std, all methods](results/iclr26_e2_figures/c4_en_core_training_loss_mean_std.svg)

Clean comparison view:

![C4 E2 validation loss mean +/- std, clean comparison](results/iclr26_e2_figures/c4_en_clean_validation_loss_mean_std.svg)

![C4 E2 validation PPL mean +/- std, clean comparison](results/iclr26_e2_figures/c4_en_clean_validation_ppl_mean_std.svg)

![C4 E2 training loss mean +/- std, clean comparison](results/iclr26_e2_figures/c4_en_clean_training_loss_mean_std.svg)

C4 E2 validation-loss checkpoint table, mean +/- sample std:

| Method | 1000 | 2000 | 4000 | 6000 | 8000 | 9150 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MatrixPolicy | 4.8403 +/- 0.0107 | 4.4058 +/- 0.0092 | 4.1549 +/- 0.0104 | 3.9977 +/- 0.0132 | 3.9071 +/- 0.0130 | 3.8830 +/- 0.0141 |
| RLB+AdamW | 4.9932 +/- 0.0304 | 4.5256 +/- 0.0118 | 4.2208 +/- 0.0102 | 4.0771 +/- 0.0129 | 4.0006 +/- 0.0114 | 3.9816 +/- 0.0111 |
| SiLU+AdamW | 5.0066 +/- 0.0246 | 4.5330 +/- 0.0144 | 4.2225 +/- 0.0137 | 4.0777 +/- 0.0133 | 4.0005 +/- 0.0114 | 3.9811 +/- 0.0128 |
| RLB+Lion | 4.9039 +/- 0.0314 | 4.4466 +/- 0.0124 | 4.1606 +/- 0.0132 | 4.0185 +/- 0.0153 | 3.9388 +/- 0.0142 | 3.9196 +/- 0.0142 |
| SiLU+Lion | 4.9530 +/- 0.0226 | 4.4680 +/- 0.0107 | 4.1678 +/- 0.0108 | 4.0211 +/- 0.0103 | 3.9407 +/- 0.0113 | 3.9213 +/- 0.0105 |
| RLB+SOAP | 5.0488 +/- 0.0029 | 4.7715 +/- 0.3408 | 4.2553 +/- 0.0107 | 4.1052 +/- 0.0110 | 4.0232 +/- 0.0090 | 4.0051 +/- 0.0092 |
| SiLU+SOAP | 5.1948 +/- 0.0604 | 4.6889 +/- 0.0405 | 4.3065 +/- 0.0112 | 4.1452 +/- 0.0118 | 4.0716 +/- 0.0224 | 4.0349 +/- 0.0108 |
| RLB+Muon | 5.1512 +/- 0.0253 | 4.5265 +/- 0.0163 | 4.1724 +/- 0.0128 | 4.0171 +/- 0.0154 | 3.9357 +/- 0.0151 | 3.9159 +/- 0.0161 |
| SiLU+Muon | 5.1609 +/- 0.0341 | 4.5248 +/- 0.0117 | 4.1777 +/- 0.0099 | 4.0254 +/- 0.0140 | 3.9448 +/- 0.0125 | 3.9251 +/- 0.0134 |
| RLB+ScheduleFree | 5.5256 +/- 0.0312 | 5.0440 +/- 0.0228 | 4.5957 +/- 0.0190 | 4.4095 +/- 0.0144 | 4.3288 +/- 0.0122 | 4.3038 +/- 0.0113 |
| SiLU+ScheduleFree | 5.5598 +/- 0.0344 | 5.0728 +/- 0.0242 | 4.6171 +/- 0.0131 | 4.4261 +/- 0.0100 | 4.3422 +/- 0.0104 | 4.3163 +/- 0.0107 |
| RLB+CAME | 5.5928 +/- 0.0198 | 5.1278 +/- 0.0290 | 4.7109 +/- 0.0498 | 4.4906 +/- 0.0632 | 4.3875 +/- 0.0628 | 4.3633 +/- 0.0626 |
| SiLU+CAME | 5.6174 +/- 0.0280 | 5.1482 +/- 0.0327 | 4.6857 +/- 0.0411 | 4.4522 +/- 0.0187 | 4.3543 +/- 0.0153 | 4.3298 +/- 0.0148 |
| RLB+ADeMaMix | 414.2971 +/- 77.7608 | 553095678.5990 +/- 943240014.0699 | 1687480172544.0000 +/- 0.0000 (n=1) | -- | -- | -- |
| SiLU+ADeMaMix | 1104.9656 +/- 565.1435 | 180791.1562 +/- 0.0000 (n=1) | -- | -- | -- | -- |

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
