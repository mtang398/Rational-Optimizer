# Research TODO

## North-Star Claim

The paper claim stays optimizer-specific:

```text
RLB exposes optimizer-visible structure, and MatrixPolicy uses that structure to improve language-model pretraining loss-vs-compute under matched optimizer configs.
```

## Current Results

### E1 M0/100M Main Suite

Full tables, completed job status, and dense mean +/- std curves are in `experiments/ICLR_RUN_STATUS.md`; exact submitted commands are in `experiments/ICLR_RUN_COMMANDS.md`. E1 M0/100M is complete across all five datasets and three seeds. MatrixPolicy rows use the accepted safe-speed replacement manifest.

| Dataset | MatrixPolicy final val loss | next best current method | gap |
| --- | ---: | ---: | ---: |
| DCLM | 4.256989 +/- 0.004197 | rlb_lion 4.305728 +/- 0.005836 | 0.048739 |
| FineWeb-Edu | 4.088287 +/- 0.009169 | rlb_lion 4.142669 +/- 0.006812 | 0.054382 |
| FineWeb | 4.319472 +/- 0.012370 | rlb_lion 4.367062 +/- 0.007532 | 0.047590 |
| Dolma-sample | 4.323933 +/- 0.005168 | rlb_lion 4.369254 +/- 0.005561 | 0.045321 |
| C4 | 4.286446 +/- 0.019324 | rlb_lion 4.335663 +/- 0.020917 | 0.049217 |

### E2 M0/300M Main Suite

E2 M0/300M is complete across DCLM, FineWeb-Edu, FineWeb, Dolma-sample, and C4. MatrixPolicy rows use the accepted safe-speed replacement manifest; full final, runtime, token-savings, and curve/checkpoint tables are in `experiments/results/iclr26_e2_*` and `experiments/results/iclr26_e2_figures/`.

| Dataset | MatrixPolicy final val loss | next best aggregate method | gap |
| --- | ---: | ---: | ---: |
| DCLM | 3.956069 +/- 0.030752 | silu_lion 3.993430 +/- 0.023038 | 0.037361 |
| FineWeb-Edu | 3.707768 +/- 0.018711 | rlb_muon 3.738164 +/- 0.021014 | 0.030396 |
| FineWeb | 3.964892 +/- 0.009459 | rlb_muon 4.001245 +/- 0.011375 | 0.036353 |
| Dolma-sample | 3.808954 +/- 0.006442 | rlb_lion 3.842503 +/- 0.009333 | 0.033549 |
| C4 | 3.883021 +/- 0.014134 | rlb_muon 3.915858 +/- 0.016066 | 0.032837 |

### WikiText Demo Anchor

```text
RLB MatrixPolicy-Muon:       3.476232 loss / 32.34 PPL
Best SiLU/SwiGLU+AdamW row:  3.549346 loss / 34.79 PPL
Gap:                         0.073114 loss / 2.45 PPL
```

Artifacts:

```text
experiments/results/rlb_matrix_policy_muon_switch_2026_05_28/
```

## Required Experiment Rule

Every reportable comparison must be matched by outer optimizer config:

```text
same phase
same dataset
same model
same token budget
same seed
same validation slice
same sequence length
same global tokens per step
same eval interval
same lr
same min_lr
same weight_decay
```

If AdamW appears with an outer config in a matched cell, MatrixPolicy must appear with the same outer config in that cell. The manifest generator enforces this for main runs.

## Immediate Tasks

1. Keep `experiments/ICLR_RUN_STATUS.md`, README files, and `TODO.md` synchronized whenever result summaries change.
2. Use the completed safe-speed E1/E2 packages as the paper-facing result source for final validation loss, token-to-target, runtime, and curve/checkpoint figures.
3. Finish derived analyses from the completed E1/E2 JSONL: validation AUC, early/mid/late AUC, paired seed gaps, ranks, timing, throughput, GPU-hour accounting, and divergence/failure-adjusted summaries.
4. Run E3 M1 scale, E4 600M horizon, throughput/memory, cross-corpus evaluation, and corpus-shift runs in that order.
5. Run sensitivity maps only after the completed main E1/E2 evidence is frozen.
6. Run method ablations last.

## Mechanism Diagnostics Needed

Required metrics per RLB layer:

| metric | meaning |
| --- | --- |
| group input RMS | whether `W_in` chooses usable domains |
| group output RMS | whether features are used |
| derivative pressure | whether groups are saturated or active |
| denominator/pole margin | rational stability |
| `W_in`/`W_out` norm product | role-scale drift and optimizer-induced imbalance |
| coefficient update norm | rational-shape movement |
| function probe delta | output function change on fixed probe inputs |

## Resource Rules

```text
max 4 A6000 GPUs per job
max 8 A6000 GPUs active total
repo below 200G
raw runs and caches stay ignored
```
