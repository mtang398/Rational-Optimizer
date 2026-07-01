# ICLR26 E1 Token-To-Target Savings

Generated from completed E1 M0/100M JSONL eval records. All rows still trained to the fixed budget of about `99.9M` tokens; this is an early-stop/speed-to-target readout only.

Each row uses `32768` global tokens/step and the native E1 eval cadence of 50 steps, or `1.64M` tokens per readout interval.

`Second-best` means the fastest non-MatrixPolicy method to reach the target within the same seed. `AdamW` means the standard `silu_adamw` row. Savings and proportions are computed only on seeds where both MatrixPolicy and the comparator reached the target. When `--replacement-manifest` is supplied, non-MatrixPolicy RLB optimizer controls are replaced by the `rlb_fused_global_rational` rows.

## DCLM

| Target loss | MP all-hit mean | Vs fastest non-MP: MP -> comparator (seeds) | Saved | Saved % | Vs SiLU+AdamW: MP -> AdamW (seeds) | Saved | Saved % |
| ---: | ---: | --- | ---: | ---: | --- | ---: | ---: |
| 4.90 | 30.0M | 30.0M -> 31.1M (3/3) | 1.1M | 3.5% | 30.0M -> 36.0M (3/3) | 6.0M | 16.7% |
| 4.70 | 39.9M | 39.9M -> 41.0M (3/3) | 1.1M | 2.7% | 39.9M -> 48.6M (3/3) | 8.7M | 18.0% |
| 4.55 | 50.2M | 50.2M -> 51.9M (3/3) | 1.6M | 3.2% | 50.2M -> 63.4M (3/3) | 13.1M | 20.7% |
| 4.45 | 60.1M | 60.1M -> 62.8M (3/3) | 2.7M | 4.3% | 60.1M -> 82.5M (3/3) | 22.4M | 27.2% |
| 4.35 | 73.2M | 73.2M -> 79.7M (3/3) | 6.6M | 8.2% | not reached (0/3) | not reached | n/a |
| 4.30 | 83.6M | 82.7M -> 95.0M (2/3) | 12.3M | 12.9% | not reached (0/3) | not reached | n/a |

## FineWeb-Edu

| Target loss | MP all-hit mean | Vs fastest non-MP: MP -> comparator (seeds) | Saved | Saved % | Vs SiLU+AdamW: MP -> AdamW (seeds) | Saved | Saved % |
| ---: | ---: | --- | ---: | ---: | --- | ---: | ---: |
| 4.80 | 30.0M | 30.0M -> 31.1M (3/3) | 1.1M | 3.5% | 30.0M -> 34.4M (3/3) | 4.4M | 12.7% |
| 4.60 | 38.2M | 38.2M -> 39.3M (3/3) | 1.1M | 2.8% | 38.2M -> 45.3M (3/3) | 7.1M | 15.7% |
| 4.40 | 49.7M | 49.7M -> 51.9M (3/3) | 2.2M | 4.2% | 49.7M -> 61.7M (3/3) | 12.0M | 19.5% |
| 4.30 | 59.5M | 59.5M -> 62.3M (3/3) | 2.7M | 4.4% | 59.5M -> 78.1M (3/3) | 18.6M | 23.8% |
| 4.20 | 71.5M | 71.5M -> 78.1M (3/3) | 6.6M | 8.4% | not reached (0/3) | not reached | n/a |
| 4.10 | 95.0M | not reached (0/3) | not reached | n/a | not reached (0/3) | not reached | n/a |

## FineWeb

| Target loss | MP all-hit mean | Vs fastest non-MP: MP -> comparator (seeds) | Saved | Saved % | Vs SiLU+AdamW: MP -> AdamW (seeds) | Saved | Saved % |
| ---: | ---: | --- | ---: | ---: | --- | ---: | ---: |
| 5.00 | 30.6M | 30.6M -> 31.7M (3/3) | 1.1M | 3.4% | 30.6M -> 35.0M (3/3) | 4.4M | 12.5% |
| 4.80 | 38.8M | 38.8M -> 39.9M (3/3) | 1.1M | 2.7% | 38.8M -> 47.0M (3/3) | 8.2M | 17.4% |
| 4.60 | 51.9M | 51.9M -> 54.1M (3/3) | 2.2M | 4.0% | 51.9M -> 67.2M (3/3) | 15.3M | 22.8% |
| 4.50 | 62.3M | 62.3M -> 65.5M (3/3) | 3.3M | 5.0% | 62.3M -> 89.0M (3/3) | 26.8M | 30.1% |
| 4.40 | 76.5M | 76.5M -> 84.7M (3/3) | 8.2M | 9.7% | not reached (0/3) | not reached | n/a |
| 4.35 | 87.4M | not reached (0/3) | not reached | n/a | not reached (0/3) | not reached | n/a |

## Dolma-sample

| Target loss | MP all-hit mean | Vs fastest non-MP: MP -> comparator (seeds) | Saved | Saved % | Vs SiLU+AdamW: MP -> AdamW (seeds) | Saved | Saved % |
| ---: | ---: | --- | ---: | ---: | --- | ---: | ---: |
| 5.00 | 32.2M | 32.2M -> 32.8M (3/3) | 0.5M | 1.7% | 32.2M -> 36.6M (3/3) | 4.4M | 11.9% |
| 4.80 | 40.4M | 40.4M -> 41.0M (3/3) | 0.5M | 1.3% | 40.4M -> 48.1M (3/3) | 7.6M | 15.9% |
| 4.60 | 54.1M | 54.1M -> 54.6M (3/3) | 0.5M | 1.0% | 54.1M -> 69.4M (3/3) | 15.3M | 22.0% |
| 4.50 | 63.9M | 63.9M -> 66.1M (3/3) | 2.2M | 3.3% | 63.9M -> 92.8M (3/3) | 28.9M | 31.2% |
| 4.40 | 78.1M | 78.1M -> 85.2M (3/3) | 7.1M | 8.3% | not reached (0/3) | not reached | n/a |
| 4.35 | 90.7M | not reached (0/3) | not reached | n/a | not reached (0/3) | not reached | n/a |

## C4

| Target loss | MP all-hit mean | Vs fastest non-MP: MP -> comparator (seeds) | Saved | Saved % | Vs SiLU+AdamW: MP -> AdamW (seeds) | Saved | Saved % |
| ---: | ---: | --- | ---: | ---: | --- | ---: | ---: |
| 5.00 | 29.5M | 29.5M -> 31.1M (3/3) | 1.6M | 5.3% | 29.5M -> 34.4M (3/3) | 4.9M | 14.3% |
| 4.80 | 37.7M | 37.7M -> 38.8M (3/3) | 1.1M | 2.8% | 37.7M -> 45.3M (3/3) | 7.6M | 16.9% |
| 4.60 | 50.2M | 50.2M -> 50.8M (3/3) | 0.5M | 1.1% | 50.2M -> 63.4M (3/3) | 13.1M | 20.7% |
| 4.50 | 59.0M | 59.0M -> 60.6M (3/3) | 1.6M | 2.7% | 59.0M -> 80.3M (3/3) | 21.3M | 26.5% |
| 4.40 | 71.0M | 71.0M -> 76.5M (3/3) | 5.5M | 7.1% | not reached (0/3) | not reached | n/a |
| 4.30 | 94.5M | not reached (0/3) | not reached | n/a | not reached (0/3) | not reached | n/a |

## Files

- `token_savings.csv`: aggregate token-to-target savings by dataset and target.
- `token_savings_per_seed.csv`: per-seed threshold hits and comparator identities.
