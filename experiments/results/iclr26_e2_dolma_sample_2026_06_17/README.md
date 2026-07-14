# ICLR26 E2 Dolma-sample 300M Summary

Completed: 2026-06-17. Manifest rows `375-419` define the full Dolma-sample E2 M0/300M cell: 3 seeds x 15 fixed methods. The cell contains 45 paper-facing rows; `3` stopped early and are reported as diverged/non-finite rather than excluded.

Each row uses `32768` global tokens/step for about `299.8M` train tokens. Validation uses the E2 Dolma-sample slice from the manifest: `val_skip_tokens=610000000`, `val_tokens=8000000`, `eval_interval=50`.
MatrixPolicy entries use validated live-statistic-corrected `rlb_fused_global_rational` JSONL rows for the same method and seed; non-MatrixPolicy RLB optimizer controls use global-rational RLB (`rlb_fused_global_rational`) replacement rows; the `row` column remains the matched main-manifest E2 row, while `source_phase`/`source_row_id` record the actual timed run.

## Final Validation Loss

| Method | Final val loss mean +/- sample std | Min | Max | Notes |
| --- | ---: | ---: | ---: | --- |
| rlb_matrixpolicy_original | 3.805292 +/- 0.007943 | 3.796139 | 3.810371 |  |
| rlb_lion | 3.841206 +/- 0.008478 | 3.831467 | 3.846927 |  |
| silu_lion | 3.847523 +/- 0.009363 | 3.836884 | 3.854513 |  |
| rlb_muon | 3.847789 +/- 0.004708 | 3.843204 | 3.852611 |  |
| silu_muon | 3.858114 +/- 0.010066 | 3.846854 | 3.866242 |  |
| silu_adamw | 3.903690 +/- 0.009091 | 3.893635 | 3.911328 |  |
| rlb_adamw | 3.904543 +/- 0.008210 | 3.896088 | 3.912484 |  |
| rlb_soap | 3.926881 +/- 0.012526 | 3.915837 | 3.940492 |  |
| silu_soap | 3.956834 +/- 0.009319 | 3.946098 | 3.962839 |  |
| rlb_schedulefree | 4.212122 +/- 0.011229 | 4.199156 | 4.218638 |  |
| silu_schedulefree | 4.215105 +/- 0.005319 | 4.210405 | 4.220879 |  |
| silu_came | 4.249166 +/- 0.026981 | 4.223768 | 4.277492 |  |
| rlb_came | 4.266527 +/- 0.040923 | 4.233150 | 4.312184 |  |
| rlb_ademamix | nan/diverged | nan | nan | 3 diverged/non-finite seeds |
| silu_ademamix | nan/diverged | nan | nan | 3 diverged/non-finite seeds |

MatrixPolicy is best on all three Dolma-sample E2 seeds. Mean final val loss is `3.805292 +/- 0.007943`; the next-best aggregate methods are `rlb_lion` at `3.841206 +/- 0.008478`, `silu_lion` at `3.847523 +/- 0.009363`, `rlb_muon` at `3.847789 +/- 0.004708`.

## Per-Seed MatrixPolicy Gap

| Seed | MatrixPolicy final loss | Best non-MP method | Best non-MP final loss | Gap |
| ---: | ---: | --- | ---: | ---: |
| 1337 | 3.810371 | rlb_lion | 3.846927 | 0.036556 |
| 2027 | 3.809366 | rlb_lion | 3.845225 | 0.035859 |
| 3407 | 3.796139 | rlb_lion | 3.831467 | 0.035327 |

## Runtime Summary

`summary.total_seconds` is training-harness wall time for the manifest row. It excludes Slurm queue wait, dependency wait, token-cache construction, extension compilation, and launcher overhead. MatrixPolicy replacement rows must pass JSONL integrity checks and the denylisted-node guard; an optional per-step timing ceiling can be enabled manually, but is off by default. Failures abort generation and require rerun/repair rather than exclusion. Node assignments were not matched across methods or corrected MatrixPolicy rows, so wall-clock values are observed allocation-specific measurements rather than hardware-normalized optimizer timings.

| Method | Runs | Early stops | Mean runtime | Std | Range | Mean s/step | Mean tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rlb_ademamix | 3 | 3 | 5.4 min | 2.5 min | 3.8-8.3 min | 0.4134 | 79606.0 |
| silu_schedulefree | 3 | 0 | 60.3 min | 1.9 min | 58.7-62.4 min | 0.3777 | 86810.7 |
| silu_lion | 3 | 0 | 62.8 min | 5.2 min | 58.5-68.5 min | 0.3942 | 83530.1 |
| silu_adamw | 3 | 0 | 63.5 min | 6.1 min | 58.4-70.3 min | 0.3991 | 82647.8 |
| rlb_lion | 3 | 0 | 64.8 min | 5.6 min | 61.5-71.3 min | 0.4002 | 82307.0 |
| rlb_adamw | 3 | 0 | 65.1 min | 5.8 min | 61.8-71.8 min | 0.4025 | 81843.6 |
| silu_came | 3 | 0 | 65.6 min | 2.3 min | 63.7-68.2 min | 0.4116 | 79666.0 |
| rlb_schedulefree | 3 | 0 | 66.1 min | 5.8 min | 62.7-72.7 min | 0.4085 | 80634.0 |
| rlb_soap | 3 | 0 | 70.2 min | 5.8 min | 63.5-73.7 min | 0.4351 | 75686.6 |
| rlb_muon | 3 | 0 | 73.5 min | 5.9 min | 66.7-77.1 min | 0.4562 | 72159.8 |
| rlb_came | 3 | 0 | 74.1 min | 5.9 min | 67.3-77.6 min | 0.4608 | 71450.7 |
| rlb_matrixpolicy_original | 3 | 0 | 76.4 min | 7.1 min | 69.2-83.3 min | 0.4785 | 68920.3 |
| silu_muon | 3 | 0 | 78.0 min | 21.0 min | 64.2-102.2 min | 0.4937 | 69498.1 |
| silu_soap | 3 | 0 | 79.6 min | 11.8 min | 71.5-93.2 min | 0.5025 | 66102.2 |
| silu_ademamix | 3 | 0 | 79.7 min | 12.5 min | 70.9-94.0 min | 0.4279 | 78310.3 |

## Dense Curve Figures

All-method view:

![Dolma-sample E2 validation loss mean +/- std, all methods](../iclr26_e2_figures/dolma_sample_core_validation_loss_mean_std.svg)

![Dolma-sample E2 validation PPL mean +/- std, all methods](../iclr26_e2_figures/dolma_sample_core_validation_ppl_mean_std.svg)

![Dolma-sample E2 training loss mean +/- std, all methods](../iclr26_e2_figures/dolma_sample_core_training_loss_mean_std.svg)

Clean comparison view:

![Dolma-sample E2 validation loss mean +/- std, clean comparison](../iclr26_e2_figures/dolma_sample_clean_validation_loss_mean_std.svg)

![Dolma-sample E2 validation PPL mean +/- std, clean comparison](../iclr26_e2_figures/dolma_sample_clean_validation_ppl_mean_std.svg)

![Dolma-sample E2 training loss mean +/- std, clean comparison](../iclr26_e2_figures/dolma_sample_clean_training_loss_mean_std.svg)

## Token-To-Target Savings

This table asks how many training tokens were needed to first reach a validation-loss threshold. It does not change the fixed-budget protocol; all completed runs still trained to about `299.8M` tokens. The readout uses the native eval cadence of 50 steps, or `1.64M` tokens.

`Second-best` means the fastest non-MatrixPolicy method to reach the target within the same seed. `AdamW` means the standard `silu_adamw` row. Savings and proportions are computed only on seeds where both MatrixPolicy and the comparator reached the target, so the comparison column explicitly shows `MP -> comparator` tokens and the shared seed count.

| Target loss | MP all-hit mean | Vs fastest non-MP: MP -> comparator (seeds) | Saved | Saved % | Vs SiLU+AdamW: MP -> AdamW (seeds) | Saved | Saved % |
| ---: | ---: | --- | ---: | ---: | --- | ---: | ---: |
| 4.20 | 93.4M | 93.4M -> 94.5M (3/3) | 1.1M | 1.2% | 93.4M -> 110.9M (3/3) | 17.5M | 15.8% |
| 4.10 | 121.2M | 121.2M -> 124.5M (3/3) | 3.3M | 2.6% | 121.2M -> 145.3M (3/3) | 24.0M | 16.5% |
| 4.00 | 156.7M | 156.7M -> 162.7M (3/3) | 6.0M | 3.7% | 156.7M -> 195.0M (3/3) | 38.2M | 19.6% |
| 3.95 | 179.1M | 179.1M -> 189.5M (3/3) | 10.4M | 5.5% | 179.1M -> 232.7M (3/3) | 53.5M | 23.0% |
| 3.90 | 205.3M | 205.3M -> 222.8M (3/3) | 17.5M | 7.8% | 199.9M -> 283.4M (1/3) | 83.6M | 29.5% |
| 3.85 | 241.9M | 241.9M -> 280.2M (3/3) | 38.2M | 13.6% | not reached (0/3) | not reached | n/a |
| 3.82 | 274.2M | not reached (0/3) | not reached | n/a | not reached (0/3) | not reached | n/a |

## Files

- `final_summary.csv`: aggregate final validation losses by method.
- `per_seed_summary.csv`: per-row final results and JSONL provenance paths.
- `runtime_summary.csv`: aggregate runtime by method.
- `token_savings.csv`: aggregate token-to-target savings.
- `token_savings_per_seed.csv`: per-seed threshold hits and comparator identities.
