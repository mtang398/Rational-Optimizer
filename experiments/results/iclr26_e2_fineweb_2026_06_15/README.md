# ICLR26 E2 FineWeb 300M Summary

Completed: 2026-06-15. Manifest rows `330-374` define the full FineWeb E2 M0/300M cell: 3 seeds x 15 fixed methods. The cell contains 45 paper-facing rows; `3` stopped early and are reported as diverged/non-finite rather than excluded.

Each row uses `32768` global tokens/step for about `299.8M` train tokens. Validation uses the E2 FineWeb slice from the manifest: `val_skip_tokens=610000000`, `val_tokens=8000000`, `eval_interval=50`.
MatrixPolicy entries use replacement JSONL rows for the same method and seed; non-MatrixPolicy RLB optimizer controls use global-rational/no-local-atom replacement rows; the `row` column remains the matched main-manifest E2 row, while `source_phase`/`source_row_id` record the actual timed run.

## Final Validation Loss

| Method | Final val loss mean +/- sample std | Min | Max | Notes |
| --- | ---: | ---: | ---: | --- |
| rlb_matrixpolicy_original | 3.962324 +/- 0.008082 | 3.956068 | 3.971449 |  |
| rlb_lion | 3.996049 +/- 0.010524 | 3.987987 | 4.007955 |  |
| rlb_muon | 3.999136 +/- 0.011036 | 3.989010 | 4.010900 |  |
| silu_lion | 4.001499 +/- 0.008463 | 3.995715 | 4.011213 |  |
| silu_muon | 4.006567 +/- 0.012834 | 3.996716 | 4.021081 |  |
| rlb_adamw | 4.060118 +/- 0.011034 | 4.050111 | 4.071951 |  |
| silu_adamw | 4.061199 +/- 0.010087 | 4.053473 | 4.072610 |  |
| rlb_soap | 4.108142 +/- 0.032040 | 4.088771 | 4.145125 |  |
| silu_soap | 4.113942 +/- 0.010095 | 4.104786 | 4.124768 |  |
| rlb_schedulefree | 4.386402 +/- 0.008098 | 4.380481 | 4.395631 |  |
| silu_schedulefree | 4.397873 +/- 0.010600 | 4.390438 | 4.410011 |  |
| silu_came | 4.406034 +/- 0.018922 | 4.393516 | 4.427802 |  |
| rlb_came | 4.480350 +/- 0.006438 | 4.472957 | 4.484715 |  |
| silu_ademamix | 1361.414062 +/- 0.000000 | 1361.414062 | 1361.414062 | 2 diverged/non-finite seeds |
| rlb_ademamix | nan/diverged | nan | nan | 3 diverged/non-finite seeds |

MatrixPolicy is best on all three FineWeb E2 seeds. Mean final val loss is `3.962324 +/- 0.008082`; the next-best aggregate methods are `rlb_lion` at `3.996049 +/- 0.010524`, `rlb_muon` at `3.999136 +/- 0.011036`, `silu_lion` at `4.001499 +/- 0.008463`.

## Per-Seed MatrixPolicy Gap

| Seed | MatrixPolicy final loss | Best non-MP method | Best non-MP final loss | Gap |
| ---: | ---: | --- | ---: | ---: |
| 1337 | 3.971449 | rlb_lion | 4.007955 | 0.036506 |
| 2027 | 3.956068 | rlb_muon | 3.989010 | 0.032943 |
| 3407 | 3.959455 | rlb_lion | 3.987987 | 0.028532 |

## Runtime Summary

`summary.total_seconds` is training-harness wall time for the manifest row. It excludes Slurm queue wait, dependency wait, token-cache construction, extension compilation, and launcher overhead. MatrixPolicy replacement rows must pass JSONL integrity checks and the denylisted-node guard; an optional per-step timing ceiling can be enabled manually, but is off by default. Failures abort generation and require rerun/repair rather than exclusion.

| Method | Runs | Early stops | Mean runtime | Std | Range | Mean s/step | Mean tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rlb_ademamix | 3 | 3 | 4.6 min | 0.4 min | 4.1-4.9 min | 0.4335 | 75932.0 |
| silu_schedulefree | 3 | 0 | 61.7 min | 0.4 min | 61.3-62.1 min | 0.3875 | 84557.5 |
| silu_lion | 3 | 0 | 64.6 min | 6.3 min | 61.0-71.9 min | 0.4066 | 81108.9 |
| rlb_matrixpolicy_original | 3 | 0 | 65.8 min | 2.9 min | 62.5-67.7 min | 0.4078 | 80467.9 |
| rlb_lion | 3 | 0 | 68.2 min | 5.4 min | 61.9-71.3 min | 0.4219 | 78036.9 |
| rlb_soap | 3 | 0 | 68.2 min | 5.0 min | 63.7-73.6 min | 0.4223 | 77878.7 |
| rlb_schedulefree | 3 | 0 | 69.2 min | 5.5 min | 62.8-72.5 min | 0.4285 | 76829.9 |
| rlb_adamw | 3 | 0 | 69.6 min | 3.4 min | 65.6-71.8 min | 0.4309 | 76194.5 |
| silu_muon | 3 | 0 | 69.7 min | 6.2 min | 65.6-76.8 min | 0.4385 | 75078.6 |
| rlb_muon | 3 | 0 | 71.4 min | 4.9 min | 67.0-76.6 min | 0.4430 | 74205.6 |
| rlb_came | 3 | 0 | 72.0 min | 5.1 min | 67.3-77.5 min | 0.4473 | 73518.9 |
| silu_soap | 3 | 0 | 72.3 min | 0.6 min | 71.7-72.9 min | 0.4571 | 71696.4 |
| silu_adamw | 3 | 0 | 76.4 min | 18.9 min | 60.2-97.1 min | 0.4840 | 70588.2 |
| silu_came | 3 | 0 | 147.2 min | 139.2 min | 66.6-307.9 min | 0.9476 | 57383.1 |
| silu_ademamix | 3 | 0 | 150.6 min | 143.3 min | 62.6-316.0 min | 0.9251 | 60189.7 |

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
| 4.40 | 84.1M | 84.1M -> 87.4M (3/3) | 3.3M | 3.8% | 84.1M -> 102.1M (3/3) | 18.0M | 17.6% |
| 4.30 | 110.9M | 110.9M -> 111.4M (3/3) | 0.5M | 0.5% | 110.9M -> 132.2M (3/3) | 21.3M | 16.1% |
| 4.20 | 142.5M | 142.5M -> 145.8M (3/3) | 3.3M | 2.2% | 142.5M -> 173.1M (3/3) | 30.6M | 17.7% |
| 4.10 | 185.7M | 185.7M -> 194.4M (3/3) | 8.7M | 4.5% | 185.7M -> 241.4M (3/3) | 55.7M | 23.1% |
| 4.05 | 213.0M | 213.0M -> 228.8M (3/3) | 15.8M | 6.9% | not reached (0/3) | not reached | n/a |
| 4.00 | 249.6M | 244.9M -> 276.1M (2/3) | 31.1M | 11.3% | not reached (0/3) | not reached | n/a |

## Files

- `final_summary.csv`: aggregate final validation losses by method.
- `per_seed_summary.csv`: per-row final results and JSONL provenance paths.
- `runtime_summary.csv`: aggregate runtime by method.
- `token_savings.csv`: aggregate token-to-target savings.
- `token_savings_per_seed.csv`: per-seed threshold hits and comparator identities.
