# Research TODO

## North-Star Claim

The paper claim stays optimizer-specific:

```text
RLB exposes optimizer-visible structure, and MatrixPolicy uses that structure to improve language-model pretraining loss-vs-compute under matched optimizer configs.
```

## Current Results

### E1 M0/100M Main Suite

Full tables, live row status, and dense mean +/- std curves are in `experiments/ICLR_RUN_STATUS.md`; exact submitted commands are in `experiments/ICLR_RUN_COMMANDS.md`. Current completed cells show MatrixPolicy as the lowest-loss method on DCLM, FineWeb-Edu, FineWeb, and Dolma-sample, with C4 still partial.

| Dataset | MatrixPolicy final val loss | next best current method | gap |
| --- | ---: | ---: | ---: |
| DCLM | 4.256224 +/- 0.004972 | rlb_lion 4.305728 +/- 0.005836 | 0.049504 |
| FineWeb-Edu | 4.088240 +/- 0.009434 | rlb_lion 4.142669 +/- 0.006812 | 0.054429 |
| FineWeb | 4.318581 +/- 0.010914 | rlb_lion 4.367062 +/- 0.007532 | 0.048481 |
| Dolma-sample | 4.323851 +/- 0.004565 | rlb_lion 4.369254 +/- 0.005561 | 0.045403 |
| C4 partial, n=2 | 4.281546 +/- 0.027902 | rlb_lion 4.334202 +/- 0.029364 | 0.052656 |

### FineWeb And FineWeb-Edu

| task | MatrixPolicy mean | SiLU+AdamW mean | best non-MatrixPolicy mean | gap vs SiLU+AdamW | gap vs best control |
| --- | ---: | ---: | ---: | ---: | ---: |
| FineWeb | 4.369701 loss / 79.04 PPL | 4.528963 loss / 92.69 PPL | 4.522311 loss / 92.08 PPL | 0.159263 | 0.152302 |
| FineWeb-Edu | 4.069422 loss / 58.52 PPL | 4.223572 loss / 68.28 PPL | 4.223572 loss / 68.28 PPL | 0.154149 | 0.153402 |

Artifacts:

```text
experiments/results/real_lm_multiseed_2026_05_31/
experiments/results/real_lm_screen_2026_05_30/
experiments/runs/real_lm_multiseed_20260531/
```

### WikiText

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

1. Use `experiments/ICLR_RUN_STATUS.md` for the live E1 state and `experiments/ICLR_EXACT_RUN_PLAN.md` as the experiment contract.
2. Keep the active allocation at no more than two 4-GPU jobs; do not submit additional GPU jobs while the E1 dependency chain is already occupying 8 A6000.
3. Let the queued E1 whole-cell jobs continue through `experiments/scripts/run_iclr26_manifest_job.sh`; if Slurm preempts a job, confirm the automatic requeue resumes and completed rows are skipped.
4. Update `experiments/ICLR_RUN_STATUS.md` whenever a job changes state, a row block completes, or a preemption/restart occurs.
5. After each complete matched cell, summarize dense validation curves, training curves, AUC, timing, divergence markers, and exact manifest row IDs.
6. After E1 finishes, build the full E1 summary tables and mean-plus-std curves before launching E2.
7. Run E2 M0 300M after E1 summaries.
8. Run E3 M1 scale, E4 600M horizon, throughput/memory, cross-corpus evaluation, and corpus-shift runs in that order.
9. Run sensitivity maps only after main M0 curves exist.
10. Run method ablations last.

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
