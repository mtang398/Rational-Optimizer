# ICLR26 Runtime Summary

Generated: 2026-06-19.

This package summarizes clean per optimizer/activation-combo runtime from completed JSONL `summary` records. The runtime field is `summary.total_seconds`, i.e. training-harness wall time for a manifest row. It excludes Slurm queue wait, dependency wait, token-cache construction, extension compilation, and launcher overhead.

Included in tracked runtime aggregates:

- E1 M0/100M clean rows: `210` rows. E1 FineWeb-Edu seed `2027` rows `75-89` are excluded because Slurm job `158117` completed with `Restarts=6` and produced restart/node-contaminated throughput outliers.
- E2 M0/300M DCLM completed cell: `45` rows, one dataset x three seeds x 15 methods.
- E2 M0/300M FineWeb-Edu completed cell: `45` rows, one dataset x three seeds x 15 methods.
- E2 M0/300M FineWeb completed cell: `45` rows, one dataset x three seeds x 15 methods.
- E2 M0/300M Dolma-sample completed cell: `45` rows, one dataset x three seeds x 15 methods.
- E2 M0/300M C4 completed cell: `45` rows, one dataset x three seeds x 15 methods.

Excluded from tracked runtime aggregates:

- E1 FineWeb-Edu seed `2027` rows `75-89`: `15` rows, restart-contaminated.
- Rows `465+` are outside E2.

No raw all-completed E1 aggregate is tracked in this package. The contaminated E1 rows are omitted from both aggregate CSVs and `runtime_per_row.csv`.

Clean rows summarized: `435`.

## E1 M0/100M All Datasets

| Combo | Runs | Mean runtime | Std | Range | Mean s/step | Mean tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RLB+MatrixPolicy | 14 | 32.0 min | 4.9 min | 23.4 min-38.3 min | 0.6032 | 55759.6 |
| SiLU+AdamW | 14 | 27.2 min | 5.3 min | 18.2 min-33.1 min | 0.5162 | 66240.2 |
| RLB+AdamW | 14 | 31.8 min | 4.9 min | 23.4 min-38.1 min | 0.5945 | 56615.9 |
| SiLU+Muon | 14 | 29.0 min | 5.4 min | 20.1 min-36.3 min | 0.5524 | 61608.8 |
| RLB+Muon | 14 | 33.5 min | 5.0 min | 25.1 min-40.4 min | 0.6275 | 53497.1 |
| SiLU+Lion | 14 | 27.0 min | 5.4 min | 18.1 min-34.1 min | 0.5134 | 66772.1 |
| RLB+Lion | 14 | 31.7 min | 5.0 min | 23.3 min-38.5 min | 0.5927 | 56845.1 |
| SiLU+SOAP | 14 | 31.1 min | 5.5 min | 22.1 min-38.8 min | 0.5928 | 57179.6 |
| RLB+SOAP | 14 | 32.4 min | 5.1 min | 24.0 min-40.0 min | 0.6076 | 55439.4 |
| SiLU+ADeMaMix | 14 | 27.6 min | 5.5 min | 18.6 min-35.1 min | 0.5244 | 65288.4 |
| RLB+ADeMaMix | 14 | 33.7 min | 5.2 min | 26.2 min-40.3 min | 0.6049 | 55715.5 |
| SiLU+CAME | 14 | 28.9 min | 5.5 min | 19.9 min-36.3 min | 0.5507 | 61913.4 |
| RLB+CAME | 14 | 33.8 min | 5.0 min | 25.4 min-39.9 min | 0.6342 | 52903.9 |
| SiLU+ScheduleFree | 14 | 27.3 min | 5.4 min | 18.4 min-33.9 min | 0.5196 | 65899.0 |
| RLB+ScheduleFree | 14 | 32.3 min | 5.0 min | 23.7 min-38.3 min | 0.6042 | 55739.4 |

## E2 M0/300M DCLM

| Combo | Runs | Mean runtime | Std | Range | Mean s/step | Mean tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RLB+MatrixPolicy | 3 | 84.8 min | 10.3 min | 78.8 min-96.7 min | 0.5293 | 62518.4 |
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

## E2 M0/300M FineWeb-Edu

| Combo | Runs | Mean runtime | Std | Range | Mean s/step | Mean tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RLB+MatrixPolicy | 3 | 91.3 min | 10.6 min | 79.1 min-97.7 min | 0.5713 | 57960.3 |
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

## E2 M0/300M FineWeb

| Combo | Runs | Mean runtime | Std | Range | Mean s/step | Mean tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RLB+MatrixPolicy | 3 | 77.4 min | 2.9 min | 74.1 min-79.4 min | 0.4808 | 68224.0 |
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

## E2 M0/300M Dolma-sample

| Combo | Runs | Mean runtime | Std | Range | Mean s/step | Mean tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RLB+MatrixPolicy | 3 | 84.5 min | 12.0 min | 75.3 min-98.1 min | 0.5266 | 63110.0 |
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

## E2 M0/300M C4

| Combo | Runs | Mean runtime | Std | Range | Mean s/step | Mean tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RLB+MatrixPolicy | 3 | 91.6 min | 11.0 min | 78.9 min-98.6 min | 0.5739 | 57743.7 |
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

## Files

- `runtime_by_scope_method_clean.csv`: clean per-combo aggregate.
- `runtime_by_dataset_method_clean.csv`: clean per-combo aggregate split by dataset.
- `runtime_per_row.csv`: one record per clean included manifest row.
