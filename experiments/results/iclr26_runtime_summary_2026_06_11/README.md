# ICLR26 Runtime Summary

Generated: 2026-06-11.

This package summarizes per optimizer/activation-combo runtime from completed JSONL `summary` records. The runtime field is `summary.total_seconds`, i.e. training-harness wall time for a manifest row. It excludes Slurm queue wait, dependency wait, token-cache construction, extension compilation, and other launcher overhead. That is the comparable per-combo runtime because E1 jobs ran whole 15-row cells inside one Slurm allocation.

Included:

- E1 M0/100M all completed datasets: `225` rows, five datasets x three seeds x 15 methods.
- E2 M0/300M completed DCLM cell: `45` rows, one dataset x three seeds x 15 methods.

Excluded:

- E2 FineWeb-Edu rows `285-329`, because that dataset cell is still in progress.
- E2 rows `330+`, because they have not been queued/completed yet.

## Runtime Quality Note

E1 FineWeb-Edu seed `2027` rows `75-89` ran in Slurm job `158117`, which completed with `Restarts=6`. The final JSONL files for those rows have one config record, one summary record, and no duplicate train steps, so `summary.total_seconds` is not directly summing archived requeue attempts. However, that matched cell is restart/node contaminated and produces pathological throughput outliers, especially rows `81-89`.

The clean tables below therefore exclude rows `75-89` from E1 runtime aggregates. Raw all-completed tables are still written to CSV for provenance.

Clean rows summarized: `255`. Raw completed rows summarized: `270`.

## Clean E1 M0/100M All Datasets

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

## Clean E2 M0/300M DCLM

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

## Raw All-Completed E1 M0/100M All Datasets

| Combo | Runs | Mean runtime | Std | Range | Mean s/step | Mean tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RLB+MatrixPolicy | 15 | 37.1 min | 20.3 min | 23.4 min-108.6 min | 0.7036 | 53078.4 |
| SiLU+AdamW | 15 | 27.6 min | 5.3 min | 18.2 min-33.7 min | 0.5248 | 65212.8 |
| RLB+AdamW | 15 | 32.2 min | 5.1 min | 23.4 min-38.5 min | 0.6032 | 55854.1 |
| SiLU+Muon | 15 | 29.4 min | 5.4 min | 20.1 min-36.3 min | 0.5604 | 60754.1 |
| RLB+Muon | 15 | 33.9 min | 5.0 min | 25.1 min-40.4 min | 0.6349 | 52890.2 |
| SiLU+Lion | 15 | 27.9 min | 6.2 min | 18.1 min-40.1 min | 0.5304 | 65164.9 |
| RLB+Lion | 15 | 32.5 min | 5.8 min | 23.3 min-44.3 min | 0.6091 | 55659.8 |
| SiLU+SOAP | 15 | 36.2 min | 20.5 min | 22.1 min-107.6 min | 0.6931 | 54409.4 |
| RLB+SOAP | 15 | 37.5 min | 20.4 min | 24.0 min-108.9 min | 0.7079 | 52777.8 |
| SiLU+ADeMaMix | 15 | 32.3 min | 19.2 min | 18.6 min-99.0 min | 0.6180 | 62068.9 |
| RLB+ADeMaMix | 15 | 38.5 min | 19.2 min | 26.2 min-105.6 min | 0.7010 | 53068.8 |
| SiLU+CAME | 15 | 34.0 min | 20.5 min | 19.9 min-105.5 min | 0.6508 | 58849.9 |
| RLB+CAME | 15 | 38.9 min | 20.4 min | 25.4 min-110.6 min | 0.7349 | 50395.5 |
| SiLU+ScheduleFree | 15 | 32.5 min | 20.6 min | 18.4 min-104.5 min | 0.6208 | 62578.1 |
| RLB+ScheduleFree | 15 | 37.4 min | 20.4 min | 23.7 min-109.0 min | 0.7047 | 53057.4 |

## Raw All-Completed E2 M0/300M DCLM

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

## Files

- `runtime_by_scope_method_clean.csv`: default clean per-combo aggregate.
- `runtime_by_dataset_method_clean.csv`: default clean per-combo aggregate split by dataset.
- `runtime_by_scope_method.csv`: raw all-completed per-combo aggregate.
- `runtime_by_dataset_method.csv`: raw all-completed per-combo aggregate split by dataset.
- `runtime_per_row.csv`: one record per included completed manifest row.
