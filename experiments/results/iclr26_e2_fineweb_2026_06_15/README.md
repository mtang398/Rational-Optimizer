# ICLR26 E2 FineWeb 300M Summary

Completed: 2026-06-14. Manifest rows `330-374` are the full FineWeb E2 M0/300M cell: 3 seeds x 15 fixed methods. All 45 rows completed with final eval at step `9150`.

Each row uses `32768` global tokens/step for about `299.8M` train tokens. Validation uses the E2 FineWeb slice from the manifest: `val_skip_tokens=610000000`, `val_tokens=8000000`, `eval_interval=50`.

## Final Validation Loss

| Method | Final val loss mean +/- sample std | Min | Max | Notes |
| --- | ---: | ---: | ---: | --- |
| rlb_matrixpolicy_original | 3.965590 +/- 0.008530 | 3.959470 | 3.975334 |  |
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

MatrixPolicy is best on all three FineWeb E2 seeds. Mean final val loss is `3.965590 +/- 0.008530`; the next-best aggregate methods are `rlb_muon` at `4.001245 +/- 0.011375`, `rlb_lion` at `4.001381 +/- 0.012800`, `silu_lion` at `4.001499 +/- 0.008463`.

## Per-Seed MatrixPolicy Gap

| Seed | MatrixPolicy final loss | Best non-MP method | Best non-MP final loss | Gap |
| ---: | ---: | --- | ---: | ---: |
| 1337 | 3.975334 | silu_lion | 4.011213 | 0.035879 |
| 2027 | 3.959470 | rlb_lion | 3.991274 | 0.031804 |
| 3407 | 3.961966 | rlb_lion | 3.997095 | 0.035129 |

## Runtime Summary

`summary.total_seconds` is training-harness wall time for the manifest row. It excludes Slurm queue wait, dependency wait, token-cache construction, extension compilation, and launcher overhead.

| Method | Runs | Mean runtime | Std | Range | Mean s/step | Mean tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| silu_schedulefree | 3 | 61.7 min | 0.4 min | 61.3-62.1 min | 0.3875 | 84557.5 |
| silu_lion | 3 | 64.6 min | 6.3 min | 61.0-71.9 min | 0.4066 | 81108.9 |
| silu_muon | 3 | 69.7 min | 6.2 min | 65.6-76.8 min | 0.4385 | 75078.6 |
| silu_soap | 3 | 72.3 min | 0.6 min | 71.7-72.9 min | 0.4571 | 71696.4 |
| rlb_adamw | 3 | 76.3 min | 1.3 min | 74.8-77.2 min | 0.4700 | 69737.2 |
| silu_adamw | 3 | 76.4 min | 18.9 min | 60.2-97.1 min | 0.4840 | 70588.2 |
| rlb_matrixpolicy_original | 3 | 77.4 min | 2.9 min | 74.1-79.4 min | 0.4808 | 68224.0 |
| rlb_lion | 3 | 79.6 min | 14.1 min | 68.6-95.5 min | 0.4920 | 68070.8 |
| rlb_muon | 3 | 85.0 min | 14.3 min | 74.0-101.2 min | 0.5263 | 63461.5 |
| rlb_soap | 3 | 85.4 min | 11.1 min | 76.9-97.9 min | 0.5299 | 62572.7 |
| rlb_schedulefree | 3 | 85.6 min | 10.0 min | 77.3-96.7 min | 0.5304 | 62408.2 |
| rlb_came | 3 | 91.1 min | 10.1 min | 82.8-102.3 min | 0.5662 | 58393.3 |
| rlb_ademamix | 3 | 104.0 min | 12.7 min | 96.0-118.7 min | 0.5141 | 64685.3 |
| silu_came | 3 | 147.2 min | 139.2 min | 66.6-307.9 min | 0.9476 | 57383.1 |
| silu_ademamix | 3 | 150.6 min | 143.3 min | 62.6-316.0 min | 0.9251 | 60189.7 |

## Dense Curve Figures

All-method view:

![FineWeb E2 validation loss mean +/- std, all methods](../iclr26_e2_figures/fineweb_core_validation_loss_mean_std.svg)

![FineWeb E2 validation PPL mean +/- std, all methods](../iclr26_e2_figures/fineweb_core_validation_ppl_mean_std.svg)

![FineWeb E2 training loss mean +/- std, all methods](../iclr26_e2_figures/fineweb_core_training_loss_mean_std.svg)

Clean comparison view:

![FineWeb E2 validation loss mean +/- std, clean comparison](../iclr26_e2_figures/fineweb_clean_validation_loss_mean_std.svg)

![FineWeb E2 validation PPL mean +/- std, clean comparison](../iclr26_e2_figures/fineweb_clean_validation_ppl_mean_std.svg)

![FineWeb E2 training loss mean +/- std, clean comparison](../iclr26_e2_figures/fineweb_clean_training_loss_mean_std.svg)

## Token-To-Target Savings

This table asks how many training tokens were needed to first reach a validation-loss threshold. It does not change the fixed-budget protocol; all completed runs still trained to about `299.8M` tokens. The readout uses the native eval cadence of 50 steps, or `1.64M` tokens.

`Second-best` means the fastest non-MatrixPolicy method to reach the target within the same seed. `AdamW` means the standard `silu_adamw` row. Savings and proportions are computed only on seeds where both MatrixPolicy and the comparator reached the target, so the comparison column explicitly shows `MP -> comparator` tokens and the shared seed count.

| Target loss | MP all-hit mean | Vs fastest non-MP: MP -> comparator (seeds) | Saved | Saved % | Vs SiLU+AdamW: MP -> AdamW (seeds) | Saved | Saved % |
| ---: | ---: | --- | ---: | ---: | --- | ---: | ---: |
| 4.40 | 83.6M | 83.6M -> 87.9M (3/3) | 4.4M | 5.0% | 83.6M -> 102.1M (3/3) | 18.6M | 18.2% |
| 4.30 | 112.5M | 112.5M -> 113.0M (3/3) | 0.5M | 0.5% | 112.5M -> 132.2M (3/3) | 19.7M | 14.9% |
| 4.20 | 143.6M | 143.6M -> 148.5M (3/3) | 4.9M | 3.3% | 143.6M -> 173.1M (3/3) | 29.5M | 17.0% |
| 4.10 | 187.3M | 187.3M -> 196.6M (3/3) | 9.3M | 4.7% | 187.3M -> 241.4M (3/3) | 54.1M | 22.4% |
| 4.05 | 214.6M | 214.6M -> 232.1M (3/3) | 17.5M | 7.5% | not reached (0/3) | not reached | n/a |
| 4.00 | 252.9M | 248.2M -> 285.9M (2/3) | 37.7M | 13.2% | not reached (0/3) | not reached | n/a |

## Files

- `final_summary.csv`: aggregate final validation losses by method.
- `per_seed_summary.csv`: per-row final results and JSONL provenance paths.
- `runtime_summary.csv`: aggregate runtime by method.
- `token_savings.csv`: aggregate token-to-target savings.
- `token_savings_per_seed.csv`: per-seed threshold hits and comparator identities.
