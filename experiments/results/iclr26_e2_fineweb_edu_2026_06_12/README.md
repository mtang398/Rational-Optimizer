# ICLR26 E2 FineWeb-Edu 300M Summary

Completed: 2026-06-12. Manifest rows `285-329` define the full FineWeb-Edu E2 M0/300M cell: 3 seeds x 15 fixed methods. The cell contains 45 paper-facing rows; `3` stopped early and are reported as diverged/non-finite rather than excluded.

Each row uses `32768` global tokens/step for about `299.8M` train tokens. Validation uses the E2 FineWeb-Edu slice from the manifest: `val_skip_tokens=610000000`, `val_tokens=8000000`, `eval_interval=50`.
MatrixPolicy entries use replacement JSONL rows for the same method and seed; non-MatrixPolicy RLB optimizer controls use `rlb_fused_global_rational` replacement rows; the `row` column remains the matched main-manifest E2 row, while `source_phase`/`source_row_id` record the actual timed run.

## Final Validation Loss

| Method | Final val loss mean +/- sample std | Min | Max | Notes |
| --- | ---: | ---: | ---: | --- |
| rlb_matrixpolicy_original | 3.701517 +/- 0.021218 | 3.682155 | 3.724200 |  |
| rlb_muon | 3.737328 +/- 0.018698 | 3.717964 | 3.755280 |  |
| rlb_lion | 3.741625 +/- 0.021374 | 3.723632 | 3.765251 |  |
| silu_lion | 3.744017 +/- 0.020802 | 3.727149 | 3.767261 |  |
| silu_muon | 3.745389 +/- 0.017006 | 3.732584 | 3.764685 |  |
| rlb_adamw | 3.802446 +/- 0.020587 | 3.784051 | 3.824684 |  |
| silu_adamw | 3.803482 +/- 0.018186 | 3.788790 | 3.823822 |  |
| rlb_soap | 3.824008 +/- 0.012762 | 3.816362 | 3.838741 |  |
| silu_soap | 3.862925 +/- 0.020105 | 3.844579 | 3.884418 |  |
| rlb_schedulefree | 4.142110 +/- 0.021695 | 4.124335 | 4.166286 |  |
| silu_came | 4.150283 +/- 0.021107 | 4.137987 | 4.174655 |  |
| silu_schedulefree | 4.155890 +/- 0.023847 | 4.134111 | 4.181372 |  |
| rlb_came | 4.225476 +/- 0.035835 | 4.185758 | 4.255385 |  |
| rlb_ademamix | nan/diverged | nan | nan | 3 diverged/non-finite seeds |
| silu_ademamix | nan/diverged | nan | nan | 3 diverged/non-finite seeds |

MatrixPolicy is best on all three FineWeb-Edu E2 seeds. Mean final val loss is `3.701517 +/- 0.021218`; the next-best aggregate methods are `rlb_muon` at `3.737328 +/- 0.018698`, `rlb_lion` at `3.741625 +/- 0.021374`, `silu_lion` at `3.744017 +/- 0.020802`.

## Per-Seed MatrixPolicy Gap

| Seed | MatrixPolicy final loss | Best non-MP method | Best non-MP final loss | Gap |
| ---: | ---: | --- | ---: | ---: |
| 1337 | 3.682155 | rlb_muon | 3.717964 | 0.035810 |
| 2027 | 3.724200 | rlb_muon | 3.755280 | 0.031080 |
| 3407 | 3.698196 | rlb_lion | 3.735992 | 0.037795 |

## Runtime Summary

`summary.total_seconds` is training-harness wall time for the manifest row. It excludes Slurm queue wait, dependency wait, token-cache construction, extension compilation, and launcher overhead. MatrixPolicy replacement rows must pass JSONL integrity checks and the denylisted-node guard; an optional per-step timing ceiling can be enabled manually, but is off by default. Failures abort generation and require rerun/repair rather than exclusion.

| Method | Runs | Early stops | Mean runtime | Std | Range | Mean s/step | Mean tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rlb_ademamix | 3 | 3 | 4.7 min | 0.4 min | 4.3-5.1 min | 0.4286 | 76584.8 |
| rlb_matrixpolicy_original | 3 | 0 | 65.6 min | 2.6 min | 62.6-67.1 min | 0.4070 | 80599.0 |
| rlb_lion | 3 | 0 | 66.1 min | 1.5 min | 65.2-67.8 min | 0.4078 | 80381.3 |
| rlb_adamw | 3 | 0 | 66.4 min | 1.7 min | 65.3-68.3 min | 0.4097 | 80014.3 |
| rlb_schedulefree | 3 | 0 | 71.0 min | 3.9 min | 66.6-73.9 min | 0.4401 | 74619.9 |
| rlb_soap | 3 | 0 | 71.4 min | 3.7 min | 67.2-73.6 min | 0.4427 | 74163.4 |
| silu_muon | 3 | 0 | 72.8 min | 12.9 min | 63.8-87.6 min | 0.4585 | 72920.4 |
| silu_lion | 3 | 0 | 74.5 min | 11.5 min | 61.2-81.1 min | 0.4709 | 70923.0 |
| silu_adamw | 3 | 0 | 75.2 min | 14.0 min | 61.4-89.4 min | 0.4754 | 70683.2 |
| rlb_came | 3 | 0 | 75.2 min | 3.6 min | 71.1-77.3 min | 0.4678 | 70158.2 |
| silu_schedulefree | 3 | 0 | 75.5 min | 11.9 min | 61.8-82.4 min | 0.4781 | 69904.1 |
| rlb_muon | 3 | 0 | 78.0 min | 2.5 min | 76.6-80.9 min | 0.4852 | 67576.3 |
| silu_came | 3 | 0 | 80.2 min | 11.7 min | 66.7-87.0 min | 0.5079 | 65602.6 |
| silu_soap | 3 | 0 | 86.8 min | 12.0 min | 72.9-93.8 min | 0.5475 | 60685.2 |
| silu_ademamix | 3 | 0 | 88.5 min | 10.7 min | 76.5-97.0 min | 0.4762 | 70175.4 |

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
| 4.20 | 74.8M | 74.8M -> 79.7M (3/3) | 4.9M | 6.2% | 74.8M -> 93.4M (3/3) | 18.6M | 19.9% |
| 4.10 | 98.9M | 98.9M -> 99.9M (3/3) | 1.1M | 1.1% | 98.9M -> 118.0M (3/3) | 19.1M | 16.2% |
| 4.00 | 125.6M | 125.6M -> 128.3M (3/3) | 2.7M | 2.1% | 125.6M -> 151.8M (3/3) | 26.2M | 17.3% |
| 3.90 | 161.7M | 161.7M -> 166.6M (3/3) | 4.9M | 3.0% | 161.7M -> 200.4M (3/3) | 38.8M | 19.3% |
| 3.85 | 183.0M | 183.0M -> 191.1M (3/3) | 8.2M | 4.3% | 183.0M -> 237.0M (3/3) | 54.1M | 22.8% |
| 3.80 | 209.7M | 209.7M -> 223.4M (3/3) | 13.7M | 6.1% | 203.2M -> 287.5M (2/3) | 84.4M | 29.3% |
| 3.75 | 243.6M | 233.5M -> 263.0M (2/3) | 29.5M | 11.2% | not reached (0/3) | not reached | n/a |

## Files

- `final_summary.csv`: aggregate final validation losses by method.
- `per_seed_summary.csv`: per-row final results and JSONL provenance paths.
- `runtime_summary.csv`: aggregate runtime by method.
- `token_savings.csv`: aggregate token-to-target savings.
- `token_savings_per_seed.csv`: per-seed threshold hits and comparator identities.
