# ICLR26 E2 C4 300M Summary

Completed: 2026-06-19. Manifest rows `420-464` are the full C4 E2 M0/300M cell: 3 seeds x 15 fixed methods. All 45 rows completed with final eval at step `9150`.

Each row uses `32768` global tokens/step for about `299.8M` train tokens. Validation uses the E2 C4 slice from the manifest: `val_skip_tokens=0`, `val_tokens=8000000`, `eval_interval=50`.

## Final Validation Loss

| Method | Final val loss mean +/- sample std | Min | Max | Notes |
| --- | ---: | ---: | ---: | --- |
| rlb_matrixpolicy_original | 3.882593 +/- 0.013925 | 3.871160 | 3.898101 |  |
| rlb_muon | 3.915858 +/- 0.016066 | 3.902808 | 3.933801 |  |
| rlb_lion | 3.919576 +/- 0.014201 | 3.907313 | 3.935135 |  |
| silu_lion | 3.921326 +/- 0.010538 | 3.913904 | 3.933388 |  |
| silu_muon | 3.925105 +/- 0.013434 | 3.911359 | 3.938204 |  |
| silu_adamw | 3.981105 +/- 0.012752 | 3.969709 | 3.994878 |  |
| rlb_adamw | 3.981627 +/- 0.011081 | 3.973570 | 3.994264 |  |
| rlb_soap | 4.005075 +/- 0.009183 | 3.999109 | 4.015650 |  |
| silu_soap | 4.034903 +/- 0.010776 | 4.027885 | 4.047310 |  |
| rlb_schedulefree | 4.303756 +/- 0.011329 | 4.294799 | 4.316491 |  |
| silu_schedulefree | 4.316317 +/- 0.010736 | 4.308978 | 4.328640 |  |
| silu_came | 4.329752 +/- 0.014752 | 4.314537 | 4.343993 |  |
| rlb_came | 4.363289 +/- 0.062600 | 4.315578 | 4.434171 |  |
| rlb_ademamix | nan/diverged | nan | nan | 3 diverged/non-finite seeds |
| silu_ademamix | nan/diverged | nan | nan | 3 diverged/non-finite seeds |

MatrixPolicy is best on all three C4 E2 seeds. Mean final val loss is `3.882593 +/- 0.013925`; the next-best aggregate methods are `rlb_muon` at `3.915858 +/- 0.016066`, `rlb_lion` at `3.919576 +/- 0.014201`, `silu_lion` at `3.921326 +/- 0.010538`.

## Per-Seed MatrixPolicy Gap

| Seed | MatrixPolicy final loss | Best non-MP method | Best non-MP final loss | Gap |
| ---: | ---: | --- | ---: | ---: |
| 1337 | 3.878519 | rlb_muon | 3.910965 | 0.032447 |
| 2027 | 3.898101 | silu_lion | 3.933388 | 0.035287 |
| 3407 | 3.871160 | rlb_muon | 3.902808 | 0.031648 |

## Runtime Summary

`summary.total_seconds` is training-harness wall time for the manifest row. It excludes Slurm queue wait, dependency wait, token-cache construction, extension compilation, and launcher overhead.

| Method | Runs | Mean runtime | Std | Range | Mean s/step | Mean tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| silu_lion | 3 | 72.0 min | 15.9 min | 53.7-81.6 min | 0.4555 | 74962.8 |
| silu_adamw | 3 | 73.1 min | 13.1 min | 58.0-81.0 min | 0.4623 | 72773.1 |
| silu_schedulefree | 3 | 76.3 min | 12.4 min | 62.0-84.0 min | 0.4827 | 69317.2 |
| silu_muon | 3 | 78.1 min | 16.1 min | 59.6-88.2 min | 0.4937 | 68698.4 |
| silu_came | 3 | 80.7 min | 12.1 min | 66.7-88.0 min | 0.5119 | 65146.4 |
| silu_soap | 3 | 87.2 min | 12.3 min | 73.0-94.6 min | 0.5517 | 60276.0 |
| rlb_adamw | 3 | 87.3 min | 19.6 min | 74.8-109.9 min | 0.5431 | 62415.6 |
| rlb_lion | 3 | 90.0 min | 11.2 min | 77.0-97.0 min | 0.5599 | 59256.4 |
| rlb_schedulefree | 3 | 90.3 min | 10.4 min | 78.4-97.8 min | 0.5619 | 58941.4 |
| silu_ademamix | 3 | 90.7 min | 11.4 min | 77.6-98.5 min | 0.4792 | 69794.4 |
| rlb_matrixpolicy_original | 3 | 91.6 min | 11.0 min | 78.9-98.6 min | 0.5739 | 57743.7 |
| rlb_muon | 3 | 95.6 min | 11.4 min | 82.4-102.7 min | 0.5945 | 55719.1 |
| rlb_soap | 3 | 96.3 min | 16.0 min | 79.3-111.2 min | 0.6018 | 55635.6 |
| rlb_came | 3 | 101.4 min | 16.6 min | 83.8-116.8 min | 0.6350 | 52682.7 |
| rlb_ademamix | 3 | 114.3 min | 23.2 min | 90.0-136.3 min | 0.6055 | 55622.2 |

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
| 4.40 | 67.7M | 67.7M -> 72.6M (3/3) | 4.9M | 6.8% | 67.7M -> 86.8M (3/3) | 19.1M | 22.0% |
| 4.30 | 90.7M | 90.7M -> 93.4M (3/3) | 2.7M | 2.9% | 90.7M -> 108.1M (3/3) | 17.5M | 16.2% |
| 4.20 | 118.5M | 118.5M -> 119.1M (3/3) | 0.5M | 0.5% | 118.5M -> 139.8M (3/3) | 21.3M | 15.2% |
| 4.10 | 151.3M | 151.3M -> 156.2M (3/3) | 4.9M | 3.1% | 151.3M -> 185.1M (3/3) | 33.9M | 18.3% |
| 4.05 | 170.9M | 170.9M -> 179.1M (3/3) | 8.2M | 4.6% | 170.9M -> 216.3M (3/3) | 45.3M | 21.0% |
| 4.00 | 196.1M | 196.1M -> 207.5M (3/3) | 11.5M | 5.5% | 196.1M -> 264.9M (3/3) | 68.8M | 26.0% |

## Files

- `final_summary.csv`: aggregate final validation losses by method.
- `per_seed_summary.csv`: per-row final results and JSONL provenance paths.
- `runtime_summary.csv`: aggregate runtime by method.
- `token_savings.csv`: aggregate token-to-target savings.
- `token_savings_per_seed.csv`: per-seed threshold hits and comparator identities.
