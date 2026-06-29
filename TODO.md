# Research TODO

## North-Star Claim

The paper claim stays optimizer-specific:

```text
RLB exposes optimizer-visible structure, and MatrixPolicy uses that structure to improve language-model pretraining loss-vs-compute under matched optimizer configs.
```

## Current Results

### E1 M0/100M Main Suite

Full tables, completed job status, and dense mean +/- std curves are in `experiments/ICLR_RUN_STATUS.md`; exact submitted commands are in `experiments/ICLR_RUN_COMMANDS.md`. E1 M0/100M is complete across all five datasets and three seeds. MatrixPolicy and all non-MatrixPolicy RLB optimizer controls use corrected global-rational/no-local-atom replacement manifests.

| Dataset | MatrixPolicy final val loss | next best current method | gap |
| --- | ---: | ---: | ---: |
| DCLM | 4.253781 +/- 0.006306 | rlb_lion 4.294575 +/- 0.008320 | 0.040794 |
| FineWeb-Edu | 4.087294 +/- 0.010192 | rlb_lion 4.136091 +/- 0.008299 | 0.048798 |
| FineWeb | 4.316243 +/- 0.012550 | rlb_lion 4.362572 +/- 0.011154 | 0.046329 |
| Dolma-sample | 4.325333 +/- 0.005305 | rlb_lion 4.362160 +/- 0.006582 | 0.036827 |
| C4 | 4.283714 +/- 0.019682 | rlb_lion 4.327134 +/- 0.015977 | 0.043419 |

### E2 M0/300M Main Suite

E2 M0/300M is complete across DCLM, FineWeb-Edu, FineWeb, Dolma-sample, and C4. MatrixPolicy and all non-MatrixPolicy RLB optimizer controls use corrected global-rational/no-local-atom replacement manifests; full final, runtime, token-savings, and curve/checkpoint tables are in `experiments/results/iclr26_e2_*` and `experiments/results/iclr26_e2_figures/`.

| Dataset | MatrixPolicy final val loss | next best aggregate method | gap |
| --- | ---: | ---: | ---: |
| DCLM | 3.951824 +/- 0.028163 | rlb_lion 3.988719 +/- 0.029477 | 0.036895 |
| FineWeb-Edu | 3.701517 +/- 0.021218 | rlb_muon 3.737328 +/- 0.018698 | 0.035811 |
| FineWeb | 3.962324 +/- 0.008082 | rlb_lion 3.996049 +/- 0.010524 | 0.033726 |
| Dolma-sample | 3.806155 +/- 0.007278 | rlb_lion 3.841206 +/- 0.008478 | 0.035051 |
| C4 | 3.877713 +/- 0.014444 | rlb_lion 3.913219 +/- 0.013928 | 0.035505 |

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
2. Use the completed global-rational MatrixPolicy and non-MatrixPolicy RLB-control E1/E2 packages as the paper-facing result source for final validation loss, token-to-target, runtime, and curve/checkpoint figures.
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
