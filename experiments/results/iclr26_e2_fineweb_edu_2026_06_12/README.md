# ICLR26 E2 FineWeb-Edu 300M Summary

Completed: 2026-06-12. Manifest rows `285-329` define the full FineWeb-Edu E2 M0/300M cell: 3 seeds x 15 fixed methods. All 45 paper-facing rows have final eval at step `9150`.

Each row uses `32768` global tokens/step for about `299.8M` train tokens. Validation uses the E2 FineWeb-Edu slice from the manifest: `val_skip_tokens=610000000`, `val_tokens=8000000`, `eval_interval=50`.
MatrixPolicy entries use the accepted safe-speed replacement JSONL rows for the same method and seed; the `row` column remains the matched main-manifest E2 row, while `source_phase`/`source_row_id` record the actual timed run.

## Final Validation Loss

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

## Per-Seed MatrixPolicy Gap

| Seed | MatrixPolicy final loss | Best non-MP method | Best non-MP final loss | Gap |
| ---: | ---: | --- | ---: | ---: |
| 1337 | 3.691080 | rlb_muon | 3.718084 | 0.027004 |
| 2027 | 3.727996 | rlb_muon | 3.760002 | 0.032006 |
| 3407 | 3.704228 | rlb_muon | 3.736407 | 0.032179 |

## Runtime Summary

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

## Dense Curve Figures

All-method view:

![FineWeb-Edu E2 validation loss mean +/- std, all methods](../iclr26_e2_figures/fineweb_edu_core_validation_loss_mean_std.svg)

![FineWeb-Edu E2 validation PPL mean +/- std, all methods](../iclr26_e2_figures/fineweb_edu_core_validation_ppl_mean_std.svg)

![FineWeb-Edu E2 training loss mean +/- std, all methods](../iclr26_e2_figures/fineweb_edu_core_training_loss_mean_std.svg)

Clean comparison view:

![FineWeb-Edu E2 validation loss mean +/- std, clean comparison](../iclr26_e2_figures/fineweb_edu_clean_validation_loss_mean_std.svg)

![FineWeb-Edu E2 validation PPL mean +/- std, clean comparison](../iclr26_e2_figures/fineweb_edu_clean_validation_ppl_mean_std.svg)

![FineWeb-Edu E2 training loss mean +/- std, clean comparison](../iclr26_e2_figures/fineweb_edu_clean_training_loss_mean_std.svg)

## Token-To-Target Savings

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

## Files

- `final_summary.csv`: aggregate final validation losses by method.
- `per_seed_summary.csv`: per-row final results and JSONL provenance paths.
- `runtime_summary.csv`: aggregate runtime by method.
- `token_savings.csv`: aggregate token-to-target savings.
- `token_savings_per_seed.csv`: per-seed threshold hits and comparator identities.
