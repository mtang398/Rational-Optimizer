# Research TODO

## North-Star Claim

The paper claim stays optimizer-specific:

```text
RLB exposes optimizer-visible structure, and MatrixPolicy uses that structure to improve language-model pretraining loss-vs-compute under matched optimizer configs.
```

## Current Results

### E1 M0/100M Main Suite

Full tables, completed job status, and dense mean +/- std curves are in `experiments/ICLR_RUN_STATUS.md`; exact submitted commands are in `experiments/ICLR_RUN_COMMANDS.md`. E1 M0/100M is complete across all five datasets and three seeds. MatrixPolicy is the lowest-loss method on DCLM, FineWeb-Edu, FineWeb, Dolma-sample, and C4 under the fixed matched manifest.

| Dataset | MatrixPolicy final val loss | next best current method | gap |
| --- | ---: | ---: | ---: |
| DCLM | 4.256224 +/- 0.004972 | rlb_lion 4.305728 +/- 0.005836 | 0.049505 |
| FineWeb-Edu | 4.088240 +/- 0.009434 | rlb_lion 4.142669 +/- 0.006812 | 0.054429 |
| FineWeb | 4.318581 +/- 0.010914 | rlb_lion 4.367062 +/- 0.007532 | 0.048481 |
| Dolma-sample | 4.323851 +/- 0.004565 | rlb_lion 4.369254 +/- 0.005561 | 0.045403 |
| C4 | 4.285119 +/- 0.020677 | rlb_lion 4.335663 +/- 0.020917 | 0.050544 |

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
2. Finish the E1 derived analyses: validation AUC, early/mid/late AUC, paired seed gaps, ranks, timing, throughput, GPU-hour accounting, and divergence/failure-adjusted summaries.
3. Before launching E2, verify the manifest rows still satisfy the matched-cell rule and that no stale partial-result language remains in Markdown.
4. Monitor the running E2 M0/300M DCLM whole-cell jobs (`294600` rows 240-254, `294599` rows 255-269), then summarize complete matched cells before launching the next E2 wave.
5. Run E3 M1 scale, E4 600M horizon, throughput/memory, cross-corpus evaluation, and corpus-shift runs in that order.
6. Run sensitivity maps only after main M0/E2 curves exist.
7. Run method ablations last.

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
