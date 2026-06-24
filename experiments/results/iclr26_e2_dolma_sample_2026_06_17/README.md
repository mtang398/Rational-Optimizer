# ICLR26 E2 Dolma-sample 300M Summary

Completed: 2026-06-17. Manifest rows `375-419` define the full Dolma-sample E2 M0/300M cell: 3 seeds x 15 fixed methods. All 45 paper-facing rows have final eval at step `9150`.

Each row uses `32768` global tokens/step for about `299.8M` train tokens. Validation uses the E2 Dolma-sample slice from the manifest: `val_skip_tokens=610000000`, `val_tokens=8000000`, `eval_interval=50`.
MatrixPolicy entries use the accepted safe-speed replacement JSONL rows for the same method and seed; the `row` column remains the matched main-manifest E2 row, while `source_phase`/`source_row_id` record the actual timed run.

## Final Validation Loss

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

## Per-Seed MatrixPolicy Gap

| Seed | MatrixPolicy final loss | Best non-MP method | Best non-MP final loss | Gap |
| ---: | ---: | --- | ---: | ---: |
| 1337 | 3.813003 | rlb_lion | 3.850787 | 0.037784 |
| 2027 | 3.812334 | rlb_lion | 3.844330 | 0.031996 |
| 3407 | 3.801525 | rlb_lion | 3.832392 | 0.030867 |

## Runtime Summary

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
| 4.20 | 93.9M | 93.9M -> 94.5M (3/3) | 0.5M | 0.6% | 93.9M -> 110.9M (3/3) | 16.9M | 15.3% |
| 4.10 | 123.4M | 123.4M -> 124.0M (3/3) | 0.5M | 0.4% | 123.4M -> 145.3M (3/3) | 21.8M | 15.0% |
| 4.00 | 158.9M | 158.9M -> 163.8M (3/3) | 4.9M | 3.0% | 158.9M -> 195.0M (3/3) | 36.0M | 18.5% |
| 3.95 | 181.9M | 181.9M -> 190.6M (3/3) | 8.7M | 4.6% | 181.9M -> 232.7M (3/3) | 50.8M | 21.8% |
| 3.90 | 209.7M | 209.7M -> 223.9M (3/3) | 14.2M | 6.3% | 204.8M -> 283.4M (1/3) | 78.6M | 27.7% |
| 3.85 | 246.3M | 244.9M -> 276.9M (2/3) | 31.9M | 11.5% | not reached (0/3) | not reached | n/a |
| 3.82 | 280.2M | not reached (0/3) | not reached | n/a | not reached (0/3) | not reached | n/a |

## Files

- `final_summary.csv`: aggregate final validation losses by method.
- `per_seed_summary.csv`: per-row final results and JSONL provenance paths.
- `runtime_summary.csv`: aggregate runtime by method.
- `token_savings.csv`: aggregate token-to-target savings.
- `token_savings_per_seed.csv`: per-seed threshold hits and comparator identities.
