# ICLR26 E2 DCLM 300M Summary

Completed: 2026-06-10. Manifest rows `240-284` are the full DCLM E2 M0/300M cell: 3 seeds x 15 fixed methods. All 45 rows completed with final eval at step `9150`.

Each row uses `32768` global tokens/step for about `299.8M` train tokens. Validation uses the E2 DCLM slice from the manifest: `val_skip_tokens=610000000`, `val_tokens=8000000`, `eval_interval=50`.

## Final Validation Loss

| Method | Final val loss mean +/- sample std | Min | Max | Notes |
| --- | ---: | ---: | ---: | --- |
| rlb_matrixpolicy_original | 3.957627 +/- 0.030713 | 3.923688 | 3.983507 |  |
| silu_lion | 3.993430 +/- 0.023038 | 3.968264 | 4.013479 |  |
| rlb_muon | 3.993489 +/- 0.029634 | 3.961723 | 4.020390 |  |
| rlb_lion | 3.994293 +/- 0.030088 | 3.960352 | 4.017691 |  |
| silu_muon | 3.997266 +/- 0.030472 | 3.964678 | 4.025052 |  |
| silu_adamw | 4.049337 +/- 0.027469 | 4.018327 | 4.070612 |  |
| rlb_adamw | 4.052915 +/- 0.028179 | 4.021017 | 4.074428 |  |
| rlb_soap | 4.076804 +/- 0.040305 | 4.034326 | 4.114511 |  |
| silu_soap | 4.096430 +/- 0.029988 | 4.062710 | 4.120108 |  |
| rlb_schedulefree | 4.356261 +/- 0.033232 | 4.318152 | 4.379206 |  |
| silu_schedulefree | 4.365672 +/- 0.029805 | 4.332936 | 4.391239 |  |
| silu_came | 4.368189 +/- 0.022586 | 4.344955 | 4.390067 |  |
| rlb_came | 4.450294 +/- 0.034021 | 4.428269 | 4.489478 |  |
| silu_ademamix | nan/diverged | nan | nan | 3 diverged/non-finite seeds |
| rlb_ademamix | nan/diverged | nan | nan | 3 diverged/non-finite seeds |

MatrixPolicy is best on all three DCLM E2 seeds. Mean final val loss is `3.957627 +/- 0.030713`; the next-best aggregate methods are `silu_lion` at `3.993430 +/- 0.023038`, `rlb_muon` at `3.993489 +/- 0.029634`, and `rlb_lion` at `3.994293 +/- 0.030088`.

## Per-Seed MatrixPolicy Gap

| Seed | MatrixPolicy final loss | Best non-MP method | Best non-MP final loss | Gap |
| ---: | ---: | --- | ---: | ---: |
| 1337 | 3.965684 | rlb_muon | 3.998354 | 0.032670 |
| 2027 | 3.983507 | silu_lion | 4.013479 | 0.029972 |
| 3407 | 3.923688 | rlb_lion | 3.960352 | 0.036664 |

## Token-To-Target Savings

This table asks how many training tokens were needed to first reach a validation-loss threshold. It does not change the fixed-budget protocol; all completed runs still trained to about `299.8M` tokens. The readout uses the native eval cadence of 50 steps, or `1.64M` tokens.

`Second-best` means the fastest non-MatrixPolicy method to reach the target within the same seed. `AdamW` means the standard `silu_adamw` row. Savings and proportions are computed only on seeds where both MatrixPolicy and the comparator reached the target, so the comparison column explicitly shows `MP -> comparator` tokens and the shared seed count.

| Target loss | MP all-hit mean | Vs fastest non-MP: MP -> comparator (seeds) | Saved | Saved % | Vs SiLU+AdamW: MP -> AdamW (seeds) | Saved | Saved % |
| ---: | ---: | --- | ---: | ---: | --- | ---: | ---: |
| 4.40 | 74.3M | 74.3M -> 80.8M (3/3) | 6.6M | 8.1% | 74.3M -> 93.4M (3/3) | 19.1M | 20.5% |
| 4.30 | 101.0M | 101.0M -> 104.9M (3/3) | 3.8M | 3.6% | 101.0M -> 120.7M (3/3) | 19.7M | 16.3% |
| 4.20 | 133.3M | 133.3M -> 139.3M (3/3) | 6.0M | 4.3% | 133.3M -> 161.1M (3/3) | 27.9M | 17.3% |
| 4.10 | 176.4M | 176.4M -> 187.9M (3/3) | 11.5M | 6.1% | 176.4M -> 227.7M (3/3) | 51.3M | 22.5% |
| 4.05 | 205.3M | 205.3M -> 222.8M (3/3) | 17.5M | 7.8% | 185.1M -> 244.1M (1/3) | 59.0M | 24.2% |
| 4.00 | 244.7M | 232.7M -> 267.9M (2/3) | 35.2M | 13.1% | not reached (0/3) | not reached | n/a |

## Files

- `final_summary.csv`: aggregate final validation losses by method.
- `per_seed_summary.csv`: per-row final results and JSONL provenance paths.
- `token_savings.csv`: aggregate token-to-target savings.
- `token_savings_per_seed.csv`: per-seed threshold hits and comparator identities.

