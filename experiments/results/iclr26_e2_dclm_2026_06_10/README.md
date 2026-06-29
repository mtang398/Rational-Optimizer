# ICLR26 E2 DCLM 300M Summary

Completed: 2026-06-10. Manifest rows `240-284` define the full DCLM E2 M0/300M cell: 3 seeds x 15 fixed methods. The cell contains 45 paper-facing rows; `3` stopped early and are reported as diverged/non-finite rather than excluded.

Each row uses `32768` global tokens/step for about `299.8M` train tokens. Validation uses the E2 DCLM slice from the manifest: `val_skip_tokens=610000000`, `val_tokens=8000000`, `eval_interval=50`.
MatrixPolicy entries use replacement JSONL rows for the same method and seed; non-MatrixPolicy RLB optimizer controls use global-rational/no-local-atom replacement rows; the `row` column remains the matched main-manifest E2 row, while `source_phase`/`source_row_id` record the actual timed run.

## Final Validation Loss

| Method | Final val loss mean +/- sample std | Min | Max | Notes |
| --- | ---: | ---: | ---: | --- |
| rlb_matrixpolicy_original | 3.951824 +/- 0.028163 | 3.921018 | 3.976248 |  |
| rlb_lion | 3.988719 +/- 0.029477 | 3.955908 | 4.012966 |  |
| rlb_muon | 3.991273 +/- 0.026011 | 3.962160 | 4.012225 |  |
| silu_lion | 3.993430 +/- 0.023038 | 3.968264 | 4.013479 |  |
| silu_muon | 3.997266 +/- 0.030472 | 3.964678 | 4.025052 |  |
| silu_adamw | 4.049337 +/- 0.027469 | 4.018327 | 4.070612 |  |
| rlb_adamw | 4.049575 +/- 0.029796 | 4.016587 | 4.074533 |  |
| rlb_soap | 4.060841 +/- 0.033142 | 4.025086 | 4.090534 |  |
| silu_soap | 4.096430 +/- 0.029988 | 4.062710 | 4.120108 |  |
| rlb_schedulefree | 4.360681 +/- 0.034358 | 4.321111 | 4.382936 |  |
| silu_schedulefree | 4.365672 +/- 0.029805 | 4.332936 | 4.391239 |  |
| silu_came | 4.368189 +/- 0.022586 | 4.344955 | 4.390067 |  |
| rlb_came | 4.450261 +/- 0.042619 | 4.413764 | 4.497098 |  |
| rlb_ademamix | nan/diverged | nan | nan | 3 diverged/non-finite seeds |
| silu_ademamix | nan/diverged | nan | nan | 3 diverged/non-finite seeds |

MatrixPolicy is best on all three DCLM E2 seeds. Mean final val loss is `3.951824 +/- 0.028163`; the next-best aggregate methods are `rlb_lion` at `3.988719 +/- 0.029477`, `rlb_muon` at `3.991273 +/- 0.026011`, `silu_lion` at `3.993430 +/- 0.023038`.

## Per-Seed MatrixPolicy Gap

| Seed | MatrixPolicy final loss | Best non-MP method | Best non-MP final loss | Gap |
| ---: | ---: | --- | ---: | ---: |
| 1337 | 3.958206 | rlb_lion | 3.997283 | 0.039077 |
| 2027 | 3.976248 | rlb_muon | 4.012225 | 0.035977 |
| 3407 | 3.921018 | rlb_lion | 3.955908 | 0.034890 |

## Runtime Summary

`summary.total_seconds` is training-harness wall time for the manifest row. It excludes Slurm queue wait, dependency wait, token-cache construction, extension compilation, and launcher overhead. MatrixPolicy replacement rows must pass JSONL integrity checks and the denylisted-node guard; an optional per-step timing ceiling can be enabled manually, but is off by default. Failures abort generation and require rerun/repair rather than exclusion.

| Method | Runs | Early stops | Mean runtime | Std | Range | Mean s/step | Mean tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rlb_ademamix | 3 | 3 | 5.4 min | 1.9 min | 3.9-7.6 min | 0.4645 | 71263.8 |
| silu_lion | 3 | 0 | 67.6 min | 12.0 min | 60.5-81.5 min | 0.4265 | 78442.0 |
| rlb_matrixpolicy_original | 3 | 0 | 67.7 min | 5.3 min | 62.7-73.2 min | 0.4207 | 78223.9 |
| silu_schedulefree | 3 | 0 | 68.9 min | 11.9 min | 61.6-82.6 min | 0.4343 | 76941.0 |
| silu_adamw | 3 | 0 | 70.9 min | 10.0 min | 61.1-81.2 min | 0.4478 | 74246.1 |
| rlb_soap | 3 | 0 | 71.2 min | 6.6 min | 65.2-78.3 min | 0.4402 | 74873.3 |
| silu_muon | 3 | 0 | 71.4 min | 14.8 min | 59.4-87.9 min | 0.4497 | 74940.2 |
| rlb_adamw | 3 | 0 | 71.7 min | 7.1 min | 63.5-75.9 min | 0.4434 | 74441.1 |
| rlb_lion | 3 | 0 | 71.8 min | 7.6 min | 63.0-76.3 min | 0.4437 | 74469.6 |
| rlb_schedulefree | 3 | 0 | 72.5 min | 7.3 min | 64.1-77.7 min | 0.4486 | 73597.4 |
| rlb_muon | 3 | 0 | 72.5 min | 7.3 min | 68.2-81.0 min | 0.4493 | 73427.9 |
| rlb_came | 3 | 0 | 75.4 min | 7.0 min | 68.9-82.8 min | 0.4660 | 70690.7 |
| silu_came | 3 | 0 | 80.2 min | 11.7 min | 66.7-87.2 min | 0.5085 | 65529.0 |
| silu_ademamix | 3 | 0 | 85.3 min | 18.1 min | 64.5-96.1 min | 0.4823 | 69300.4 |
| silu_soap | 3 | 0 | 86.9 min | 12.3 min | 72.8-94.5 min | 0.5487 | 60587.7 |

## Dense Curve Figures

All-method view:

![DCLM E2 validation loss mean +/- std, all methods](../iclr26_e2_figures/dclm_core_validation_loss_mean_std.svg)

![DCLM E2 validation PPL mean +/- std, all methods](../iclr26_e2_figures/dclm_core_validation_ppl_mean_std.svg)

![DCLM E2 training loss mean +/- std, all methods](../iclr26_e2_figures/dclm_core_training_loss_mean_std.svg)

Clean comparison view:

![DCLM E2 validation loss mean +/- std, clean comparison](../iclr26_e2_figures/dclm_clean_validation_loss_mean_std.svg)

![DCLM E2 validation PPL mean +/- std, clean comparison](../iclr26_e2_figures/dclm_clean_validation_ppl_mean_std.svg)

![DCLM E2 training loss mean +/- std, clean comparison](../iclr26_e2_figures/dclm_clean_training_loss_mean_std.svg)

## Token-To-Target Savings

This table asks how many training tokens were needed to first reach a validation-loss threshold. It does not change the fixed-budget protocol; all completed runs still trained to about `299.8M` tokens. The readout uses the native eval cadence of 50 steps, or `1.64M` tokens.

`Second-best` means the fastest non-MatrixPolicy method to reach the target within the same seed. `AdamW` means the standard `silu_adamw` row. Savings and proportions are computed only on seeds where both MatrixPolicy and the comparator reached the target, so the comparison column explicitly shows `MP -> comparator` tokens and the shared seed count.

| Target loss | MP all-hit mean | Vs fastest non-MP: MP -> comparator (seeds) | Saved | Saved % | Vs SiLU+AdamW: MP -> AdamW (seeds) | Saved | Saved % |
| ---: | ---: | --- | ---: | ---: | --- | ---: | ---: |
| 4.40 | 74.3M | 74.3M -> 80.3M (3/3) | 6.0M | 7.5% | 74.3M -> 93.4M (3/3) | 19.1M | 20.5% |
| 4.30 | 100.5M | 100.5M -> 103.8M (3/3) | 3.3M | 3.2% | 100.5M -> 120.7M (3/3) | 20.2M | 16.7% |
| 4.20 | 131.1M | 131.1M -> 137.1M (3/3) | 6.0M | 4.4% | 131.1M -> 161.1M (3/3) | 30.0M | 18.6% |
| 4.10 | 174.8M | 174.8M -> 185.7M (3/3) | 10.9M | 5.9% | 174.8M -> 227.7M (3/3) | 53.0M | 23.3% |
| 4.05 | 201.0M | 201.0M -> 220.1M (3/3) | 19.1M | 8.7% | 181.9M -> 244.1M (1/3) | 62.3M | 25.5% |
| 4.00 | 238.1M | 227.7M -> 262.1M (2/3) | 34.4M | 13.1% | not reached (0/3) | not reached | n/a |

## Files

- `final_summary.csv`: aggregate final validation losses by method.
- `per_seed_summary.csv`: per-row final results and JSONL provenance paths.
- `runtime_summary.csv`: aggregate runtime by method.
- `token_savings.csv`: aggregate token-to-target savings.
- `token_savings_per_seed.csv`: per-seed threshold hits and comparator identities.
