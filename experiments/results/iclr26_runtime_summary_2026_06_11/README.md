# ICLR26 Runtime Summary

Generated: 2026-06-29.

This package summarizes clean per optimizer/activation-combo runtime from JSONL `summary` records. The runtime field is `summary.total_seconds`, i.e. training-harness wall time for a manifest row. It excludes Slurm queue wait, dependency wait, token-cache construction, extension compilation, launcher overhead, and pre-restart partial attempts. MatrixPolicy replacement rows must pass JSONL integrity checks and the denylisted-node guard; an optional per-step timing ceiling can be enabled manually, but is off by default. Failures abort generation and require rerun/repair rather than exclusion. Early-stop rows are retained and counted explicitly rather than excluded.

Included in tracked runtime aggregates:

- E1 M0/100M clean rows: `225` rows. E1 FineWeb-Edu seed `2027` job `158117` had `Restarts=6`; rows `75-80` are retained because their completed JSONL timings match adjacent seeds. Original rows `81-88` are skipped because the existing artifacts cannot reconstruct trusted per-row runtime after multiple preempted allocations and partial JSONLs. Completed clean repair overlay rows for E1 FineWeb-Edu seed `2027` rows `81-88`: `8/8`. Row `89` is replaced by the completed MatrixPolicy replacement rerun when available.
- Non-MatrixPolicy RLB optimizer controls overlaid from global-rational/no-local-atom (`rlb_fused_global_rational`) runs: `210` aggregate row-count contributions. Early-stop rows retained in runtime aggregates: `30`.
- E2 M0/300M DCLM completed cell: `45` rows, one dataset x three seeds x 15 methods.
- E2 M0/300M FineWeb-Edu completed cell: `45` rows, one dataset x three seeds x 15 methods.
- E2 M0/300M FineWeb completed cell: `45` rows, one dataset x three seeds x 15 methods.
- E2 M0/300M Dolma-sample completed cell: `45` rows, one dataset x three seeds x 15 methods.
- E2 M0/300M C4 completed cell: `45` rows, one dataset x three seeds x 15 methods.

Excluded from tracked runtime aggregates:

- Original E1 FineWeb-Edu seed `2027` rows `81-88`: `8` rows skipped from the main manifest runtime source. They are overlaid from the completed clean repair manifest `experiments/manifests/iclr26_e1_fineweb_edu_seed2027_runtime_repair_manifest.csv`.
- Rows `465+` are outside E2.

No raw Slurm-elapsed E1 aggregate is tracked in this package. Runtime aggregates use completed JSONL `summary.total_seconds` only for clean row attempts; original restart-contaminated rows `81-88` are not assigned inferred row times.

Clean rows summarized: `450`.

## E1 M0/100M All Datasets

| Combo | Runs | Early stops | Mean runtime | Std | Range | Mean s/step | Mean tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RLB+MatrixPolicy | 15 | 0 | 24.5 min | 2.2 min | 22.4 min-27.8 min | 0.4563 | 72344.1 |
| SiLU+AdamW | 15 | 0 | 27.6 min | 5.3 min | 18.2 min-33.7 min | 0.5248 | 65212.8 |
| RLB+AdamW | 15 | 0 | 23.7 min | 1.6 min | 21.1 min-25.7 min | 0.4395 | 74907.4 |
| SiLU+Muon | 15 | 0 | 29.4 min | 5.4 min | 20.1 min-36.3 min | 0.5604 | 60754.1 |
| RLB+Muon | 15 | 0 | 26.0 min | 0.9 min | 24.6 min-26.9 min | 0.4829 | 67938.0 |
| SiLU+Lion | 15 | 0 | 27.9 min | 6.2 min | 18.1 min-40.1 min | 0.5304 | 65164.9 |
| RLB+Lion | 15 | 0 | 23.8 min | 1.3 min | 21.1 min-25.2 min | 0.4404 | 74648.1 |
| SiLU+SOAP | 15 | 0 | 30.5 min | 5.9 min | 21.4 min-38.8 min | 0.5802 | 58784.6 |
| RLB+SOAP | 15 | 0 | 24.9 min | 0.8 min | 23.6 min-25.7 min | 0.4627 | 70893.2 |
| SiLU+ADeMaMix | 15 | 0 | 26.9 min | 5.8 min | 18.0 min-35.1 min | 0.5119 | 67431.3 |
| RLB+ADeMaMix | 15 | 15 | 4.9 min | 2.5 min | 3.2 min-13.3 min | 0.4548 | 72235.5 |
| SiLU+CAME | 15 | 0 | 28.3 min | 5.8 min | 19.3 min-36.3 min | 0.5382 | 63804.1 |
| RLB+CAME | 15 | 0 | 25.8 min | 1.3 min | 23.0 min-27.3 min | 0.4805 | 68386.2 |
| SiLU+ScheduleFree | 15 | 0 | 26.7 min | 5.8 min | 17.8 min-33.9 min | 0.5072 | 68064.2 |
| RLB+ScheduleFree | 15 | 0 | 24.2 min | 1.3 min | 21.4 min-25.5 min | 0.4495 | 73108.8 |

## E2 M0/300M DCLM

| Combo | Runs | Early stops | Mean runtime | Std | Range | Mean s/step | Mean tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RLB+MatrixPolicy | 3 | 0 | 67.7 min | 5.3 min | 62.7 min-73.2 min | 0.4207 | 78223.9 |
| SiLU+AdamW | 3 | 0 | 70.9 min | 10.0 min | 61.1 min-81.2 min | 0.4478 | 74246.1 |
| RLB+AdamW | 3 | 0 | 71.7 min | 7.1 min | 63.5 min-75.9 min | 0.4434 | 74441.1 |
| SiLU+Muon | 3 | 0 | 71.4 min | 14.8 min | 59.4 min-87.9 min | 0.4497 | 74940.2 |
| RLB+Muon | 3 | 0 | 72.5 min | 7.3 min | 68.2 min-81.0 min | 0.4493 | 73427.9 |
| SiLU+Lion | 3 | 0 | 67.6 min | 12.0 min | 60.5 min-81.5 min | 0.4265 | 78442.0 |
| RLB+Lion | 3 | 0 | 71.8 min | 7.6 min | 63.0 min-76.3 min | 0.4437 | 74469.6 |
| SiLU+SOAP | 3 | 0 | 86.9 min | 12.3 min | 72.8 min-94.5 min | 0.5487 | 60587.7 |
| RLB+SOAP | 3 | 0 | 71.2 min | 6.6 min | 65.2 min-78.3 min | 0.4402 | 74873.3 |
| SiLU+ADeMaMix | 3 | 0 | 85.3 min | 18.1 min | 64.5 min-96.1 min | 0.4823 | 69300.4 |
| RLB+ADeMaMix | 3 | 3 | 5.4 min | 1.9 min | 3.9 min-7.6 min | 0.4645 | 71263.8 |
| SiLU+CAME | 3 | 0 | 80.2 min | 11.7 min | 66.7 min-87.2 min | 0.5085 | 65529.0 |
| RLB+CAME | 3 | 0 | 75.4 min | 7.0 min | 68.9 min-82.8 min | 0.4660 | 70690.7 |
| SiLU+ScheduleFree | 3 | 0 | 68.9 min | 11.9 min | 61.6 min-82.6 min | 0.4343 | 76941.0 |
| RLB+ScheduleFree | 3 | 0 | 72.5 min | 7.3 min | 64.1 min-77.7 min | 0.4486 | 73597.4 |

## E2 M0/300M FineWeb-Edu

| Combo | Runs | Early stops | Mean runtime | Std | Range | Mean s/step | Mean tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RLB+MatrixPolicy | 3 | 0 | 65.6 min | 2.6 min | 62.6 min-67.1 min | 0.4070 | 80599.0 |
| SiLU+AdamW | 3 | 0 | 75.2 min | 14.0 min | 61.4 min-89.4 min | 0.4754 | 70683.2 |
| RLB+AdamW | 3 | 0 | 66.4 min | 1.7 min | 65.3 min-68.3 min | 0.4097 | 80014.3 |
| SiLU+Muon | 3 | 0 | 72.8 min | 12.9 min | 63.8 min-87.6 min | 0.4585 | 72920.4 |
| RLB+Muon | 3 | 0 | 78.0 min | 2.5 min | 76.6 min-80.9 min | 0.4852 | 67576.3 |
| SiLU+Lion | 3 | 0 | 74.5 min | 11.5 min | 61.2 min-81.1 min | 0.4709 | 70923.0 |
| RLB+Lion | 3 | 0 | 66.1 min | 1.5 min | 65.2 min-67.8 min | 0.4078 | 80381.3 |
| SiLU+SOAP | 3 | 0 | 86.8 min | 12.0 min | 72.9 min-93.8 min | 0.5475 | 60685.2 |
| RLB+SOAP | 3 | 0 | 71.4 min | 3.7 min | 67.2 min-73.6 min | 0.4427 | 74163.4 |
| SiLU+ADeMaMix | 3 | 0 | 88.5 min | 10.7 min | 76.5 min-97.0 min | 0.4762 | 70175.4 |
| RLB+ADeMaMix | 3 | 3 | 4.7 min | 0.4 min | 4.3 min-5.1 min | 0.4286 | 76584.8 |
| SiLU+CAME | 3 | 0 | 80.2 min | 11.7 min | 66.7 min-87.0 min | 0.5079 | 65602.6 |
| RLB+CAME | 3 | 0 | 75.2 min | 3.6 min | 71.1 min-77.3 min | 0.4678 | 70158.2 |
| SiLU+ScheduleFree | 3 | 0 | 75.5 min | 11.9 min | 61.8 min-82.4 min | 0.4781 | 69904.1 |
| RLB+ScheduleFree | 3 | 0 | 71.0 min | 3.9 min | 66.6 min-73.9 min | 0.4401 | 74619.9 |

## E2 M0/300M FineWeb

| Combo | Runs | Early stops | Mean runtime | Std | Range | Mean s/step | Mean tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RLB+MatrixPolicy | 3 | 0 | 65.8 min | 2.9 min | 62.5 min-67.7 min | 0.4078 | 80467.9 |
| SiLU+AdamW | 3 | 0 | 76.4 min | 18.9 min | 60.2 min-97.1 min | 0.4840 | 70588.2 |
| RLB+AdamW | 3 | 0 | 69.6 min | 3.4 min | 65.6 min-71.8 min | 0.4309 | 76194.5 |
| SiLU+Muon | 3 | 0 | 69.7 min | 6.2 min | 65.6 min-76.8 min | 0.4385 | 75078.6 |
| RLB+Muon | 3 | 0 | 71.4 min | 4.9 min | 67.0 min-76.6 min | 0.4430 | 74205.6 |
| SiLU+Lion | 3 | 0 | 64.6 min | 6.3 min | 61.0 min-71.9 min | 0.4066 | 81108.9 |
| RLB+Lion | 3 | 0 | 68.2 min | 5.4 min | 61.9 min-71.3 min | 0.4219 | 78036.9 |
| SiLU+SOAP | 3 | 0 | 72.3 min | 0.6 min | 71.7 min-72.9 min | 0.4571 | 71696.4 |
| RLB+SOAP | 3 | 0 | 68.2 min | 5.0 min | 63.7 min-73.6 min | 0.4223 | 77878.7 |
| SiLU+ADeMaMix | 3 | 0 | 150.6 min | 143.3 min | 62.6 min-316.0 min | 0.9251 | 60189.7 |
| RLB+ADeMaMix | 3 | 3 | 4.6 min | 0.4 min | 4.1 min-4.9 min | 0.4335 | 75932.0 |
| SiLU+CAME | 3 | 0 | 147.2 min | 139.2 min | 66.6 min-307.9 min | 0.9476 | 57383.1 |
| RLB+CAME | 3 | 0 | 72.0 min | 5.1 min | 67.3 min-77.5 min | 0.4473 | 73518.9 |
| SiLU+ScheduleFree | 3 | 0 | 61.7 min | 0.4 min | 61.3 min-62.1 min | 0.3875 | 84557.5 |
| RLB+ScheduleFree | 3 | 0 | 69.2 min | 5.5 min | 62.8 min-72.5 min | 0.4285 | 76829.9 |

## E2 M0/300M Dolma-sample

| Combo | Runs | Early stops | Mean runtime | Std | Range | Mean s/step | Mean tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RLB+MatrixPolicy | 3 | 0 | 73.8 min | 14.9 min | 62.5 min-90.7 min | 0.4600 | 73238.4 |
| SiLU+AdamW | 3 | 0 | 63.5 min | 6.1 min | 58.4 min-70.3 min | 0.3991 | 82647.8 |
| RLB+AdamW | 3 | 0 | 65.1 min | 5.8 min | 61.8 min-71.8 min | 0.4025 | 81843.6 |
| SiLU+Muon | 3 | 0 | 78.0 min | 21.0 min | 64.2 min-102.2 min | 0.4937 | 69498.1 |
| RLB+Muon | 3 | 0 | 73.5 min | 5.9 min | 66.7 min-77.1 min | 0.4562 | 72159.8 |
| SiLU+Lion | 3 | 0 | 62.8 min | 5.2 min | 58.5 min-68.5 min | 0.3942 | 83530.1 |
| RLB+Lion | 3 | 0 | 64.8 min | 5.6 min | 61.5 min-71.3 min | 0.4002 | 82307.0 |
| SiLU+SOAP | 3 | 0 | 79.6 min | 11.8 min | 71.5 min-93.2 min | 0.5025 | 66102.2 |
| RLB+SOAP | 3 | 0 | 70.2 min | 5.8 min | 63.5 min-73.7 min | 0.4351 | 75686.6 |
| SiLU+ADeMaMix | 3 | 0 | 79.7 min | 12.5 min | 70.9 min-94.0 min | 0.4279 | 78310.3 |
| RLB+ADeMaMix | 3 | 3 | 5.4 min | 2.5 min | 3.8 min-8.3 min | 0.4134 | 79606.0 |
| SiLU+CAME | 3 | 0 | 65.6 min | 2.3 min | 63.7 min-68.2 min | 0.4116 | 79666.0 |
| RLB+CAME | 3 | 0 | 74.1 min | 5.9 min | 67.3 min-77.6 min | 0.4608 | 71450.7 |
| SiLU+ScheduleFree | 3 | 0 | 60.3 min | 1.9 min | 58.7 min-62.4 min | 0.3777 | 86810.7 |
| RLB+ScheduleFree | 3 | 0 | 66.1 min | 5.8 min | 62.7 min-72.7 min | 0.4085 | 80634.0 |

## E2 M0/300M C4

| Combo | Runs | Early stops | Mean runtime | Std | Range | Mean s/step | Mean tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RLB+MatrixPolicy | 3 | 0 | 64.2 min | 2.9 min | 62.5 min-67.6 min | 0.3976 | 82517.0 |
| SiLU+AdamW | 3 | 0 | 73.1 min | 13.1 min | 58.0 min-81.0 min | 0.4623 | 72773.1 |
| RLB+AdamW | 3 | 0 | 68.4 min | 5.5 min | 62.0 min-71.7 min | 0.4237 | 77714.4 |
| SiLU+Muon | 3 | 0 | 78.1 min | 16.1 min | 59.6 min-88.2 min | 0.4937 | 68698.4 |
| RLB+Muon | 3 | 0 | 74.5 min | 6.6 min | 66.9 min-78.5 min | 0.4631 | 71183.7 |
| SiLU+Lion | 3 | 0 | 72.0 min | 15.9 min | 53.7 min-81.6 min | 0.4555 | 74962.8 |
| RLB+Lion | 3 | 0 | 68.0 min | 5.4 min | 61.7 min-71.1 min | 0.4206 | 78275.3 |
| SiLU+SOAP | 3 | 0 | 87.2 min | 12.3 min | 73.0 min-94.6 min | 0.5517 | 60276.0 |
| RLB+SOAP | 3 | 0 | 78.0 min | 17.2 min | 63.7 min-97.2 min | 0.4864 | 69648.4 |
| SiLU+ADeMaMix | 3 | 0 | 90.7 min | 11.4 min | 77.6 min-98.5 min | 0.4792 | 69794.4 |
| RLB+ADeMaMix | 3 | 3 | 4.8 min | 0.9 min | 3.9 min-5.7 min | 0.4339 | 75864.4 |
| SiLU+CAME | 3 | 0 | 80.7 min | 12.1 min | 66.7 min-88.0 min | 0.5119 | 65146.4 |
| RLB+CAME | 3 | 0 | 80.3 min | 14.8 min | 67.3 min-96.5 min | 0.5011 | 66954.0 |
| SiLU+ScheduleFree | 3 | 0 | 76.3 min | 12.4 min | 62.0 min-84.0 min | 0.4827 | 69317.2 |
| RLB+ScheduleFree | 3 | 0 | 66.0 min | 5.4 min | 62.8 min-72.3 min | 0.4083 | 80613.8 |

## Files

- `runtime_by_scope_method_clean.csv`: clean per-combo aggregate.
- `runtime_by_dataset_method_clean.csv`: clean per-combo aggregate split by dataset.
- `runtime_per_row.csv`: one record per clean included manifest row.
