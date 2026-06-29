# ICLR26 E2 C4 300M Summary

Completed: 2026-06-19. Manifest rows `420-464` define the full C4 E2 M0/300M cell: 3 seeds x 15 fixed methods. The cell contains 45 paper-facing rows; `3` stopped early and are reported as diverged/non-finite rather than excluded.

Each row uses `32768` global tokens/step for about `299.8M` train tokens. Validation uses the E2 C4 slice from the manifest: `val_skip_tokens=0`, `val_tokens=8000000`, `eval_interval=50`.
MatrixPolicy entries use replacement JSONL rows for the same method and seed; non-MatrixPolicy RLB optimizer controls use global-rational/no-local-atom replacement rows; the `row` column remains the matched main-manifest E2 row, while `source_phase`/`source_row_id` record the actual timed run.

## Final Validation Loss

| Method | Final val loss mean +/- sample std | Min | Max | Notes |
| --- | ---: | ---: | ---: | --- |
| rlb_matrixpolicy_original | 3.877713 +/- 0.014444 | 3.866602 | 3.894042 |  |
| rlb_lion | 3.913219 +/- 0.013928 | 3.900077 | 3.927818 |  |
| rlb_muon | 3.918867 +/- 0.014875 | 3.908324 | 3.935881 |  |
| silu_lion | 3.921326 +/- 0.010538 | 3.913904 | 3.933388 |  |
| silu_muon | 3.925105 +/- 0.013434 | 3.911359 | 3.938204 |  |
| rlb_adamw | 3.978731 +/- 0.009773 | 3.970975 | 3.989709 |  |
| silu_adamw | 3.981105 +/- 0.012752 | 3.969709 | 3.994878 |  |
| rlb_soap | 4.002413 +/- 0.019369 | 3.990991 | 4.024777 |  |
| silu_soap | 4.034903 +/- 0.010776 | 4.027885 | 4.047310 |  |
| rlb_schedulefree | 4.308438 +/- 0.007124 | 4.302670 | 4.316401 |  |
| silu_schedulefree | 4.316317 +/- 0.010736 | 4.308978 | 4.328640 |  |
| silu_came | 4.329752 +/- 0.014752 | 4.314537 | 4.343993 |  |
| rlb_came | 4.365085 +/- 0.058182 | 4.320882 | 4.431001 |  |
| rlb_ademamix | nan/diverged | nan | nan | 3 diverged/non-finite seeds |
| silu_ademamix | nan/diverged | nan | nan | 3 diverged/non-finite seeds |

MatrixPolicy is best on all three C4 E2 seeds. Mean final val loss is `3.877713 +/- 0.014444`; the next-best aggregate methods are `rlb_lion` at `3.913219 +/- 0.013928`, `rlb_muon` at `3.918867 +/- 0.014875`, `silu_lion` at `3.921326 +/- 0.010538`.

## Per-Seed MatrixPolicy Gap

| Seed | MatrixPolicy final loss | Best non-MP method | Best non-MP final loss | Gap |
| ---: | ---: | --- | ---: | ---: |
| 1337 | 3.872497 | rlb_lion | 3.911761 | 0.039264 |
| 2027 | 3.894042 | rlb_lion | 3.927818 | 0.033777 |
| 3407 | 3.866602 | rlb_lion | 3.900077 | 0.033475 |

## Runtime Summary

`summary.total_seconds` is training-harness wall time for the manifest row. It excludes Slurm queue wait, dependency wait, token-cache construction, extension compilation, and launcher overhead. MatrixPolicy replacement rows must pass JSONL integrity checks and the denylisted-node guard; an optional per-step timing ceiling can be enabled manually, but is off by default. Failures abort generation and require rerun/repair rather than exclusion.

| Method | Runs | Early stops | Mean runtime | Std | Range | Mean s/step | Mean tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rlb_ademamix | 3 | 3 | 4.8 min | 0.9 min | 3.9-5.7 min | 0.4339 | 75864.4 |
| rlb_matrixpolicy_original | 3 | 0 | 64.2 min | 2.9 min | 62.5-67.6 min | 0.3976 | 82517.0 |
| rlb_schedulefree | 3 | 0 | 66.0 min | 5.4 min | 62.8-72.3 min | 0.4083 | 80613.8 |
| rlb_lion | 3 | 0 | 68.0 min | 5.4 min | 61.7-71.1 min | 0.4206 | 78275.3 |
| rlb_adamw | 3 | 0 | 68.4 min | 5.5 min | 62.0-71.7 min | 0.4237 | 77714.4 |
| silu_lion | 3 | 0 | 72.0 min | 15.9 min | 53.7-81.6 min | 0.4555 | 74962.8 |
| silu_adamw | 3 | 0 | 73.1 min | 13.1 min | 58.0-81.0 min | 0.4623 | 72773.1 |
| rlb_muon | 3 | 0 | 74.5 min | 6.6 min | 66.9-78.5 min | 0.4631 | 71183.7 |
| silu_schedulefree | 3 | 0 | 76.3 min | 12.4 min | 62.0-84.0 min | 0.4827 | 69317.2 |
| rlb_soap | 3 | 0 | 78.0 min | 17.2 min | 63.7-97.2 min | 0.4864 | 69648.4 |
| silu_muon | 3 | 0 | 78.1 min | 16.1 min | 59.6-88.2 min | 0.4937 | 68698.4 |
| rlb_came | 3 | 0 | 80.3 min | 14.8 min | 67.3-96.5 min | 0.5011 | 66954.0 |
| silu_came | 3 | 0 | 80.7 min | 12.1 min | 66.7-88.0 min | 0.5119 | 65146.4 |
| silu_soap | 3 | 0 | 87.2 min | 12.3 min | 73.0-94.6 min | 0.5517 | 60276.0 |
| silu_ademamix | 3 | 0 | 90.7 min | 11.4 min | 77.6-98.5 min | 0.4792 | 69794.4 |

## Dense Curve Figures

All-method view:

![C4 E2 validation loss mean +/- std, all methods](../iclr26_e2_figures/c4_en_core_validation_loss_mean_std.svg)

![C4 E2 validation PPL mean +/- std, all methods](../iclr26_e2_figures/c4_en_core_validation_ppl_mean_std.svg)

![C4 E2 training loss mean +/- std, all methods](../iclr26_e2_figures/c4_en_core_training_loss_mean_std.svg)

Clean comparison view:

![C4 E2 validation loss mean +/- std, clean comparison](../iclr26_e2_figures/c4_en_clean_validation_loss_mean_std.svg)

![C4 E2 validation PPL mean +/- std, clean comparison](../iclr26_e2_figures/c4_en_clean_validation_ppl_mean_std.svg)

![C4 E2 training loss mean +/- std, clean comparison](../iclr26_e2_figures/c4_en_clean_training_loss_mean_std.svg)

## Token-To-Target Savings

This table asks how many training tokens were needed to first reach a validation-loss threshold. It does not change the fixed-budget protocol; all completed runs still trained to about `299.8M` tokens. The readout uses the native eval cadence of 50 steps, or `1.64M` tokens.

`Second-best` means the fastest non-MatrixPolicy method to reach the target within the same seed. `AdamW` means the standard `silu_adamw` row. Savings and proportions are computed only on seeds where both MatrixPolicy and the comparator reached the target, so the comparison column explicitly shows `MP -> comparator` tokens and the shared seed count.

| Target loss | MP all-hit mean | Vs fastest non-MP: MP -> comparator (seeds) | Saved | Saved % | Vs SiLU+AdamW: MP -> AdamW (seeds) | Saved | Saved % |
| ---: | ---: | --- | ---: | ---: | --- | ---: | ---: |
| 4.40 | 68.3M | 68.3M -> 72.6M (3/3) | 4.4M | 6.0% | 68.3M -> 86.8M (3/3) | 18.6M | 21.4% |
| 4.30 | 91.8M | 91.8M -> 91.8M (3/3) | 0.0M | 0.0% | 91.8M -> 108.1M (3/3) | 16.4M | 15.2% |
| 4.20 | 117.4M | 117.4M -> 118.5M (3/3) | 1.1M | 0.9% | 117.4M -> 139.8M (3/3) | 22.4M | 16.0% |
| 4.10 | 150.2M | 150.2M -> 153.5M (3/3) | 3.3M | 2.1% | 150.2M -> 185.1M (3/3) | 35.0M | 18.9% |
| 4.05 | 170.4M | 170.4M -> 176.9M (3/3) | 6.6M | 3.7% | 170.4M -> 216.3M (3/3) | 45.9M | 21.2% |
| 4.00 | 195.0M | 195.0M -> 204.8M (3/3) | 9.8M | 4.8% | 195.0M -> 264.9M (3/3) | 69.9M | 26.4% |

## Files

- `final_summary.csv`: aggregate final validation losses by method.
- `per_seed_summary.csv`: per-row final results and JSONL provenance paths.
- `runtime_summary.csv`: aggregate runtime by method.
- `token_savings.csv`: aggregate token-to-target savings.
- `token_savings_per_seed.csv`: per-seed threshold hits and comparator identities.
