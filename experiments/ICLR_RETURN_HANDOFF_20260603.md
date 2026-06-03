# ICLR Return Handoff - 2026-06-03

Last updated: 2026-06-03T18:20:30-04:00

This file is the live operational handoff for returning to the project. It records completed smoke results, currently running Slurm work, live progress observed from JSONL/logs, and the exact next conditions. Raw run outputs remain ignored; compact summaries and launch infrastructure are tracked.

## Current Slurm State

Active jobs at 2026-06-03T18:20:30-04:00:

| job | state | elapsed | node | purpose | GPU use |
| --- | --- | ---: | --- | --- | ---: |
| `67183` | RUNNING | 00:46:05 | `sun-compute-03` | Phase 1 protocol-lock DCLM AdamW control shard, configs 0-3 | 4 A6000 |
| `67184` | RUNNING | 00:46:01 | `fang-compute-02` | Phase 1 protocol-lock DCLM MatrixPolicy shard, configs 0-3 | 4 A6000 |

Active total: 8 A6000, exactly at the cap. Do not submit another GPU job until one of these exits.

Useful checks when returning:

```bash
squeue -u mt872 -o "%.18i %.9P %.40j %.8T %.10M %.6D %R"
sacct -j 67183,67184,69975,69976,71046,71047,71048,71049 --format=JobID,JobName%30,State,ExitCode,Elapsed,NodeList
sed -n '1,260p' experiments/runs/logs/iclr-protocol-lock-67183.out
sed -n '1,260p' experiments/runs/logs/iclr-protocol-lock-67184.out
```

## Live Phase 1 Protocol-Lock Progress

These are not final results. They are the latest observed in-progress JSONL/log state at 2026-06-03T18:20:30-04:00.

| job | shard | current row | status | latest train step/loss | latest eval step/loss | notes |
| --- | --- | --- | --- | ---: | ---: | --- |
| `67183` | DCLM AdamW control configs 0-3 | `dclm_adamw_lr0.0001_wd0.03_phase1_protocol_lock/silu` | complete | 1525 / 4.9551 | 1525 / 5.0774 | First SiLU+AdamW row complete. |
| `67183` | DCLM AdamW control configs 0-3 | `dclm_adamw_lr0.0001_wd0.03_phase1_protocol_lock/rlb_fused_fixed_strong_ffn` | running | 1160 / 5.0358 | 1150 / 5.1759 | Paired RLB+AdamW row is running. |
| `67184` | DCLM MatrixPolicy configs 0-3 | `dclm_matrix_policy_as2.0_gg0.20_lr0.0002_wd0.03_phase1_protocol_lock/rlb_fused_fixed_strong_ffn` | running | 460 / 5.2513 | 450 / 5.3607 | First MatrixPolicy config is running with dense evals present. |

Submitted Phase 1 commands used the new protocol-lock launcher:

```bash
experiments/scripts/run_iclr_phase1_protocol_lock_20260603.sh
```

Shared protocol for both jobs:

```bash
RATIONAL_OPT_TORCH_FALLBACK=0
CONFIRM_ICLR_PHASE1=1
PROTOCOL_STAGE=protocol_lock
TASKS=dclm
SEEDS=1337
OUTPUT_ROOT=experiments/runs/iclr26_phase1_protocol_lock
TOKEN_CACHE_DIR=experiments/cache/tokens_iclr26_phase1_protocol_lock
RUN_SUFFIX=phase1_protocol_lock
MAX_TRAIN_TOKENS=50000000
MAX_VAL_TOKENS=4000000
STEPS=1525
EVAL_INTERVAL=50
EVAL_BATCHES=10
LOG_INTERVAL=10
BATCH_SIZE=16
GRAD_ACCUM=2
DCLM_VAL_SKIP_TOKENS=110000000
COMMON_EXTRA_ARGS="--layers 18 --d-model 1024 --heads 16 --ffn-dim 3072"
BUILD_EXT=0
```

Shard-specific settings:

| job | optimizer families | grid/chunk |
| --- | --- | --- |
| `67183` | `adamw` | `ADAMW_LRS="0.0001 0.0003 0.0005"`, `ADAMW_WEIGHT_DECAYS="0.03 0.10 0.20"`, `CONFIG_START=0`, `CONFIG_LIMIT=4` |
| `67184` | `rational_matrix_policy_onpolicy` | `MATRIX_POLICY_LRS="0.0002 0.0003 0.0005"`, `MATRIX_POLICY_WEIGHT_DECAYS="0.03 0.10"`, `MATRIX_ADAM_LR_SCALES="2.0 3.0 4.0"`, `MATRIX_GROUP_GAINS="0.20 0.35"`, `CONFIG_START=0`, `CONFIG_LIMIT=4` |

Important: job `67175` was a canceled legacy-named submission and should not be used for results.

## Queued Continuation

The next bounded shards are already queued behind both active jobs and cannot start until `67183` and `67184` both complete successfully. This keeps active GPU use at or below 8 A6000.

| job | state | dependency | shard | GPU use when released |
| --- | --- | --- | --- | ---: |
| `69975` | PENDING, dependency-held | `afterok:67183:67184` | DCLM AdamW control configs 4-7 | 4 A6000 |
| `69976` | PENDING, dependency-held | `afterok:67183:67184` | DCLM MatrixPolicy configs 4-7 | 4 A6000 |
| `71046` | PENDING, dependency-held | `afterok:69975` | DCLM AdamW control config 8 | 4 A6000 |
| `71047` | PENDING, dependency-held | `afterok:69976` | DCLM MatrixPolicy configs 8-11 | 4 A6000 |
| `71048` | PENDING, dependency-held | `afterok:71046` | FineWeb-Edu AdamW control configs 0-3 | 4 A6000 |
| `71049` | PENDING, dependency-held | `afterok:71047` | FineWeb-Edu MatrixPolicy configs 0-3 | 4 A6000 |

Useful dependency check:

```bash
scontrol show job 69975
scontrol show job 69976
scontrol show job 71046
scontrol show job 71047
scontrol show job 71048
scontrol show job 71049
```

## Completed Phase 0A/0B Smoke Results

All rows below used the compiled RLB extension path, job-local extension build directories, exact dataset names/configs from the ICLR run plan, and explicit smoke validation skip (`VAL_SKIP_TOKENS=10000`, `DEFAULT_VAL_SKIP_TOKENS=10000`). Raw JSONL lives under ignored `experiments/runs/iclr26_smoke/`.

Final validation loss, lower is better:

| dataset | SiLU+AdamW | RLB+AdamW | SiLU+Muon | RLB+Muon | SiLU+SOAP | RLB+SOAP | RLB+MatrixPolicy group-stat | best smoke row |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `dclm` | 7.2601 | 7.1610 | 8.8333 | 8.3440 | 7.1443 | 7.1319 | 7.0418 | MatrixPolicy |
| `fineweb_edu` | 7.4095 | 7.2029 | 8.8111 | 8.4145 | 7.1700 | 7.1295 | 7.0805 | MatrixPolicy |
| `dolma_sample` | 6.2900 | 6.2468 | 8.1279 | 7.5320 | 6.0741 | 6.1706 | 6.0709 | MatrixPolicy, near SOAP/SiLU |
| `c4_en` | 7.5214 | 7.3066 | 8.8898 | 8.4984 | 7.2832 | 7.2602 | 7.2037 | MatrixPolicy |

Phase 0A/0B Slurm jobs:

| job | tasks | state | elapsed |
| --- | --- | --- | --- |
| `62426` | `dclm fineweb_edu` | COMPLETED, exit 0 | 00:31:38 |
| `62425` | `dolma_sample c4_en` | COMPLETED, exit 0 | 00:36:49 |

Interpretation: this is feasibility smoke, not paper evidence. It is a useful sanity signal: MatrixPolicy completed and was the best final-validation row on all four Phase 0A/0B datasets.

## Completed Phase 0C M1 Smoke

Job `65084` completed with exit `0:0` in `00:11:12` on `sun-compute-03`.

Command used:

```bash
env RATIONAL_OPT_TORCH_FALLBACK=0 REAL_LM_TASKS="dclm" SEEDS="1337" RUN_SUFFIX="iclr26_smoke_m1_dclm" OUTPUT_ROOT="experiments/runs/iclr26_smoke" TOKEN_CACHE_DIR="experiments/cache/tokens_iclr26_smoke" MAX_TRAIN_TOKENS=4000000 MAX_VAL_TOKENS=200000 STEPS=120 EVAL_INTERVAL=60 EVAL_BATCHES=2 LOG_INTERVAL=10 BATCH_SIZE=8 GRAD_ACCUM=4 INCLUDE_MUON=0 VAL_SKIP_TOKENS=10000 DEFAULT_VAL_SKIP_TOKENS=10000 COMMON_EXTRA_ARGS="--layers 18 --d-model 1024 --heads 16 --ffn-dim 3072" sbatch experiments/scripts/run_real_lm_screen_20260530.sh
```

Final M1 smoke rows:

| row | final step | final val loss | tokens/s |
| --- | ---: | ---: | ---: |
| `dclm_adamw_controls_iclr26_smoke_m1_dclm/silu` | 120 | 6.8513 | 30607.3 |
| `dclm_adamw_controls_iclr26_smoke_m1_dclm/rlb_fused_fixed_strong_ffn` | 120 | 6.7335 | 24243.2 |
| `dclm_matrix_policy_groupstat_iclr26_smoke_m1_dclm/rlb_fused_fixed_strong_ffn` | 120 | 6.7349 | 25005.4 |

Interpretation: at this larger smoke scale, both RLB+AdamW and RLB+MatrixPolicy beat SiLU+AdamW. MatrixPolicy is essentially tied with RLB+AdamW in final validation loss on this short 120-step smoke and is faster than RLB+AdamW in tokens/s for this run.

## Reporting Rule For Future Result Updates

Curves are more important than isolated final numbers. Every future result update should preserve and report:

```text
dense validation-loss curves, with eval interval <= 50 for paper/protocol runs
training-loss curves
mean +/- std curves when multiple seeds exist
AUC / loss-vs-step / loss-vs-token summaries
loss-vs-GPU-hour or wall-clock curves when timing is available
divergence and incomplete-run markers on the trajectory, not only in footnotes
final validation loss as a table column, not the whole story
```

Do not replace curve evidence with a final-number-only summary. Do not launch future paper/protocol runs with 200-step evaluation spacing.

## Rough Finish Estimate

Estimates below are deliberately rough because MatrixPolicy step time changes after warmup and Slurm scheduling can delay dependency release. Current time basis: 2026-06-03T18:20:30-04:00.

| item | expected window | basis |
| --- | --- | --- |
| `67183` active AdamW shard | about 2026-06-03 20:45-21:30 ET | first config nearly complete; three AdamW configs remain after the current paired RLB row. |
| `67184` active MatrixPolicy shard | about 2026-06-04 03:00-04:30 ET | first MatrixPolicy config at step 460/1525 after 46 minutes; four configs total. |
| `69975` / `69976` release | after both `67183` and `67184` complete, likely around 2026-06-04 03:00-04:30 ET | dependency is `afterok:67183:67184`. |
| User return in ~12h | around 2026-06-04 06:20 ET | expected state: `69975` and `69976` should be running, or `71046` may have started if AdamW configs 4-7 already completed. |

Do not treat these as paper timing numbers; use curve summaries and Slurm accounting after completion.

## Pending Tasks By Condition

1. If either active job fails:
   - Do not submit more GPU work.
   - Check `sacct -j <job>` and `experiments/runs/logs/iclr-protocol-lock-<job>.out` first.
   - Patch the exact failing launcher/runtime path, commit, push, then rerun only the failed shard.

2. If both active jobs complete:
   - Run:

```bash
.venv-cu128/bin/python experiments/scripts/summarize_iclr_phase1_protocol_lock.py \
  --run-root experiments/runs/iclr26_phase1_protocol_lock \
  --output-dir experiments/runs/iclr26_phase1_protocol_lock/summary
```

   - Record the compact DCLM protocol-lock summary in this file or a tracked result summary. The summarizer now emits `eval_curves.csv`, `train_curves.csv`, validation-loss/PPL plots, training-loss plots, AUC fields, and dense-curve checks; use those curve artifacts before final validation losses.
   - Jobs `69975` and `69976` should release automatically if both active jobs succeed. If they do not release, inspect their dependency state with `scontrol show job`.

3. Continue only with protocol-locked evidence:
   - Use `dclm` and `fineweb_edu` as specified in the exact plan.
   - Keep raw outputs under ignored run directories.
   - Commit only code, plans, compact summaries, plots, and paper assets.

4. After Phase 1 passes:
   - Move to the main 100M M0 suite in the plan across `dclm`, `fineweb_edu`, `fineweb`, `dolma_sample`, and `c4_en` with seeds `1337`, `2027`, `3407`.
   - Do not start late ablations before main evidence is complete.

5. Always enforce these constraints:
   - Max 4 A6000 per job.
   - Max 8 A6000 active total.
   - Repo below 200G.
   - No substitute dataset/toolchain path for the planned experiment.
   - Keep FineWeb and FineWeb-Edu README curves/data intact.

## Self-Continuation Instructions For Next Codex Run

When the user returns or the session resumes, do this in order:

1. Check `squeue` and `sacct` for `67183,67184,69975,69976,71046,71047,71048,71049`.
2. If any job failed, do not submit new GPU work. Inspect the relevant `experiments/runs/logs/iclr-protocol-lock-<job>.out`, patch the exact cause, commit, push, and rerun only the failed shard.
3. If jobs completed, run the curve-first summarizer, not a final-number-only summary:

```bash
.venv-cu128/bin/python experiments/scripts/summarize_iclr_phase1_protocol_lock.py \
  --run-root experiments/runs/iclr26_phase1_protocol_lock \
  --output-dir experiments/runs/iclr26_phase1_protocol_lock/summary
```

4. Inspect `eval_curves.csv`, `train_curves.csv`, validation-loss plots, training-loss plots, AUC fields, and dense-curve checks before making any claim. Curves are primary; final validation loss is just one column.
5. If active GPU use is below 8 and the dependency chain has stopped because a lineage finished cleanly, queue the next bounded Phase 1 shard from `experiments/ICLR_EXACT_RUN_PLAN.md`, preserving max 4 A6000 per job and max 8 active A6000 total.
6. Update this handoff and push after any completed result summary or new queued jobs.

## Repo/Infrastructure State

The old legacy tuning-named launcher/summarizer surface has been removed from the tracked tree. The active protocol-lock files are:

| file | purpose |
| --- | --- |
| `experiments/scripts/run_iclr_phase1_protocol_lock_20260603.sh` | Protocol-lock Slurm launcher with DCLM support, bounded `CONFIG_START`/`CONFIG_LIMIT` shards, model-size hook, and optional extension-build guard. |
| `experiments/scripts/summarize_iclr_phase1_protocol_lock.py` | Curve-first protocol-lock summarizer with run CSV, ranking CSV, dense eval/train curve CSVs, validation/training plots, AUC fields, and Markdown summary outputs. |
| `experiments/results/iclr26_smoke_20260603/summary.md` | Compact tracked Phase 0 smoke summary. |
| `.gitignore` | Ignores raw `experiments/runs/iclr26_phase1_protocol_lock/` outputs. |

Verified before this handoff rewrite: tracked filenames and tracked text contain no legacy tuning-surface matches outside ignored raw output/cache/log paths.

Paper PDF was already rendered with the Overleaf-style `pdflatex`/`bibtex` path and committed earlier at `paper/iclr_method_draft/main.pdf`.
