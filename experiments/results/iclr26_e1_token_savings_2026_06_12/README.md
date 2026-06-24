# ICLR26 E1 Token-To-Target Savings

Generated from completed E1 M0/100M JSONL eval records. All rows still trained to the fixed budget of about `99.9M` tokens; this is an early-stop/speed-to-target readout only.

Each row uses `32768` global tokens/step and the native E1 eval cadence of 50 steps, or `1.64M` tokens per readout interval.

`Second-best` means the fastest non-MatrixPolicy method to reach the target within the same seed. `AdamW` means the standard `silu_adamw` row. Savings and proportions are computed only on seeds where both MatrixPolicy and the comparator reached the target.

## DCLM

| Target loss | MP all-hit mean | Vs fastest non-MP: MP -> comparator (seeds) | Saved | Saved % | Vs SiLU+AdamW: MP -> AdamW (seeds) | Saved | Saved % |
| ---: | ---: | --- | ---: | ---: | --- | ---: | ---: |
| 4.90 | 30.0M | 30.0M -> 32.2M (3/3) | 2.2M | 6.8% | 30.0M -> 36.0M (3/3) | 6.0M | 16.7% |
| 4.70 | 39.9M | 39.9M -> 42.1M (3/3) | 2.2M | 5.2% | 39.9M -> 48.6M (3/3) | 8.7M | 18.0% |
| 4.55 | 50.2M | 50.2M -> 53.5M (3/3) | 3.3M | 6.1% | 50.2M -> 63.4M (3/3) | 13.1M | 20.7% |
| 4.45 | 60.6M | 60.6M -> 64.4M (3/3) | 3.8M | 5.9% | 60.6M -> 82.5M (3/3) | 21.8M | 26.5% |
| 4.35 | 73.7M | 73.7M -> 83.0M (3/3) | 9.3M | 11.2% | not reached (0/3) | not reached | n/a |
| 4.30 | 84.7M | 85.2M -> 99.9M (1/3) | 14.7M | 14.8% | not reached (0/3) | not reached | n/a |

## FineWeb-Edu

| Target loss | MP all-hit mean | Vs fastest non-MP: MP -> comparator (seeds) | Saved | Saved % | Vs SiLU+AdamW: MP -> AdamW (seeds) | Saved | Saved % |
| ---: | ---: | --- | ---: | ---: | --- | ---: | ---: |
| 4.80 | 30.0M | 30.0M -> 32.2M (3/3) | 2.2M | 6.8% | 30.0M -> 34.4M (3/3) | 4.4M | 12.7% |
| 4.60 | 38.2M | 38.2M -> 39.3M (3/3) | 1.1M | 2.8% | 38.2M -> 45.3M (3/3) | 7.1M | 15.7% |
| 4.40 | 49.7M | 49.7M -> 52.4M (3/3) | 2.7M | 5.2% | 49.7M -> 61.7M (3/3) | 12.0M | 19.5% |
| 4.30 | 58.4M | 58.4M -> 63.4M (3/3) | 4.9M | 7.8% | 58.4M -> 78.1M (3/3) | 19.7M | 25.2% |
| 4.20 | 71.5M | 71.5M -> 80.3M (3/3) | 8.7M | 10.9% | not reached (0/3) | not reached | n/a |
| 4.10 | 95.0M | not reached (0/3) | not reached | n/a | not reached (0/3) | not reached | n/a |

## FineWeb

| Target loss | MP all-hit mean | Vs fastest non-MP: MP -> comparator (seeds) | Saved | Saved % | Vs SiLU+AdamW: MP -> AdamW (seeds) | Saved | Saved % |
| ---: | ---: | --- | ---: | ---: | --- | ---: | ---: |
| 5.00 | 30.6M | 30.6M -> 32.2M (3/3) | 1.6M | 5.1% | 30.6M -> 35.0M (3/3) | 4.4M | 12.5% |
| 4.80 | 38.8M | 38.8M -> 40.4M (3/3) | 1.6M | 4.1% | 38.8M -> 47.0M (3/3) | 8.2M | 17.4% |
| 4.60 | 51.9M | 51.9M -> 55.2M (3/3) | 3.3M | 5.9% | 51.9M -> 67.2M (3/3) | 15.3M | 22.8% |
| 4.50 | 62.3M | 62.3M -> 66.6M (3/3) | 4.4M | 6.6% | 62.3M -> 89.0M (3/3) | 26.8M | 30.1% |
| 4.40 | 77.0M | 77.0M -> 86.3M (3/3) | 9.3M | 10.8% | not reached (0/3) | not reached | n/a |
| 4.35 | 88.5M | not reached (0/3) | not reached | n/a | not reached (0/3) | not reached | n/a |

## Dolma-sample

| Target loss | MP all-hit mean | Vs fastest non-MP: MP -> comparator (seeds) | Saved | Saved % | Vs SiLU+AdamW: MP -> AdamW (seeds) | Saved | Saved % |
| ---: | ---: | --- | ---: | ---: | --- | ---: | ---: |
| 5.00 | 31.7M | 31.7M -> 33.3M (3/3) | 1.6M | 4.9% | 31.7M -> 36.6M (3/3) | 4.9M | 13.4% |
| 4.80 | 39.9M | 39.9M -> 41.5M (3/3) | 1.6M | 3.9% | 39.9M -> 48.1M (3/3) | 8.2M | 17.0% |
| 4.60 | 53.5M | 53.5M -> 56.3M (3/3) | 2.7M | 4.9% | 53.5M -> 69.4M (3/3) | 15.8M | 22.8% |
| 4.50 | 63.4M | 63.4M -> 67.2M (3/3) | 3.8M | 5.7% | 63.4M -> 92.8M (3/3) | 29.5M | 31.8% |
| 4.40 | 77.6M | 77.6M -> 87.4M (3/3) | 9.8M | 11.2% | not reached (0/3) | not reached | n/a |
| 4.35 | 89.6M | not reached (0/3) | not reached | n/a | not reached (0/3) | not reached | n/a |

## C4

| Target loss | MP all-hit mean | Vs fastest non-MP: MP -> comparator (seeds) | Saved | Saved % | Vs SiLU+AdamW: MP -> AdamW (seeds) | Saved | Saved % |
| ---: | ---: | --- | ---: | ---: | --- | ---: | ---: |
| 5.00 | 29.5M | 29.5M -> 31.1M (3/3) | 1.6M | 5.3% | 29.5M -> 34.4M (3/3) | 4.9M | 14.3% |
| 4.80 | 37.7M | 37.7M -> 38.8M (3/3) | 1.1M | 2.8% | 37.7M -> 45.3M (3/3) | 7.6M | 16.9% |
| 4.60 | 49.7M | 49.7M -> 51.9M (3/3) | 2.2M | 4.2% | 49.7M -> 63.4M (3/3) | 13.7M | 21.6% |
| 4.50 | 58.4M | 58.4M -> 61.7M (3/3) | 3.3M | 5.3% | 58.4M -> 80.3M (3/3) | 21.8M | 27.2% |
| 4.40 | 71.5M | 71.5M -> 78.1M (3/3) | 6.6M | 8.4% | not reached (0/3) | not reached | n/a |
| 4.30 | not reached | not reached (0/3) | not reached | n/a | not reached (0/3) | not reached | n/a |

## Files

- `token_savings.csv`: aggregate token-to-target savings by dataset and target.
- `token_savings_per_seed.csv`: per-seed threshold hits and comparator identities.
