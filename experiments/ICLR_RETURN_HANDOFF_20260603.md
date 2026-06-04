# ICLR Return Handoff - 2026-06-03

Last updated: 2026-06-04T14:57:20-04:00

This file is the live operational handoff for returning to the project. It records completed smoke results, currently running Slurm work, live progress observed from JSONL/logs, and the exact next conditions. Raw run outputs remain ignored; compact summaries and launch infrastructure are tracked.

## Current Slurm State

Active jobs at 2026-06-04T14:57:20-04:00:

| job | state | elapsed | node | purpose | GPU use |
| --- | --- | ---: | --- | --- | ---: |
| `71047` | RUNNING | 04:30:38+ | `monakhova-compute-01` | Phase 1 protocol-lock DCLM MatrixPolicy shard, configs 8-11 | 4 A6000 |
| `71048` | RUNNING | 00:38:49+ after restart | `ma-compute-02` | Phase 1 protocol-lock FineWeb-Edu AdamW control shard, configs 0-3 | 4 A6000 |

Active total: 8 A6000, exactly at the cap. Do not submit another GPU job until one exits.

Useful checks when returning:

```bash
squeue -u mt872 -o "%.18i %.9P %.40j %.8T %.10M %.6D %R"
sacct -j 67183,67184,69975,69976,71046,71047,71048,71049,143550,143584,143591,143611 --format=JobID,JobName%30,State,ExitCode,Elapsed,NodeList
sed -n '1,260p' experiments/runs/logs/iclr-protocol-lock-71047.out
sed -n '1,260p' experiments/runs/logs/iclr-protocol-lock-71048.out
```

Completed Phase 1 jobs so far:

| job | state | elapsed | shard |
| --- | --- | ---: | --- |
| `67183` | COMPLETED, exit 0 | 03:03:21 | DCLM AdamW configs 0-3 |
| `67184` | COMPLETED, exit 0 | 05:20:22 | DCLM MatrixPolicy configs 0-3 |
| `69975` | COMPLETED, exit 0 | 06:52:31 | DCLM AdamW configs 4-7 |
| `69976` | COMPLETED, exit 0 | 06:07:10 | DCLM MatrixPolicy configs 4-7 |
| `71046` | COMPLETED, exit 0 | 01:17:56 | DCLM AdamW config 8 |

## Live Phase 1 Protocol-Lock Progress

These are not final Phase 1 results. They are the latest observed in-progress JSONL/log state at 2026-06-04T14:57:20-04:00.

| job | shard | current row | status | latest train step/loss | latest eval step/loss | notes |
| --- | --- | --- | --- | ---: | ---: | --- |
| `71047` | DCLM MatrixPolicy configs 8-11 | `dclm_matrix_policy_as4.0_gg0.20_lr0.0002_wd0.10_phase1_protocol_lock/rlb_fused_fixed_strong_ffn` | running | 870 / 4.7419 | 850 / 4.9356 | Two configs in this shard completed; current row is past halfway. |
| `71048` | FineWeb-Edu AdamW configs 0-3 | `fineweb_edu_adamw_lr0.0001_wd0.03_phase1_protocol_lock/rlb_fused_fixed_strong_ffn` | running after one Slurm restart | 90 / 7.7851 | 50 / 8.8326 | SiLU row completed after restart; paired RLB row is now active with dense evals. |

The curve-first summarizer was run at 2026-06-04T14:57:20-04:00. Tracked artifacts now live under:

```text
experiments/results/iclr26_phase1_protocol_lock_20260604/
```

Summary artifact counts from the latest summarizer run:

| artifact | count |
| --- | ---: |
| runs/groups detected | 31 |
| validation curve points | 948 |
| training curve points | 4560 |
| plots | 8 |

DCLM partial protocol-lock signal from completed rows:

| comparison | best final val loss | full-run val-loss AUC | status |
| --- | ---: | ---: | --- |
| best RLB+AdamW | 4.4516 | 5.1528 | complete DCLM AdamW grid |
| best SiLU+AdamW final-loss row | 4.4691 | 5.1821 | complete DCLM AdamW grid |
| best SiLU+AdamW AUC row | 4.4720 | 5.1813 | complete DCLM AdamW grid |
| best completed MatrixPolicy final-loss row | 4.6385 | 5.3518 | DCLM MatrixPolicy has one config still running |
| best completed MatrixPolicy AUC row | 4.6406 | 5.3505 | DCLM MatrixPolicy has one config still running |

FineWeb-Edu partial protocol-lock signal from current rows:

| comparison | latest/best val loss | full-run val-loss AUC | status |
| --- | ---: | ---: | --- |
| SiLU+AdamW lr=1e-4 wd=0.03 | 5.0715 | 5.8448 | complete after one Slurm restart |
| RLB+AdamW lr=1e-4 wd=0.03 | 8.8326 | 9.9497 | running, only two eval points so far |

Interpretation for internal planning: the completed DCLM AdamW protocol grid supports the RLB architecture under AdamW, but the current MatrixPolicy grid is not a headline win on DCLM. Treat Phase 1 as protocol-lock evidence and wait for the full FineWeb-Edu half before freezing any main-suite settings.

## Queued Continuation

Dependency-held continuation now covers the remaining FineWeb-Edu Phase 1 AdamW and MatrixPolicy shards. This keeps active GPU use at or below 8 A6000.

| job | state | dependency | shard | GPU use when released |
| --- | --- | --- | --- | ---: |
| `71049` | PENDING, dependency-held | `afterok:71047` | FineWeb-Edu MatrixPolicy configs 0-3 | 4 A6000 |
| `143550` | PENDING, dependency-held | `afterok:71048` | FineWeb-Edu AdamW configs 4-7 | 4 A6000 |
| `143584` | PENDING, dependency-held | `afterok:143550` | FineWeb-Edu AdamW config 8 | 4 A6000 |
| `143591` | PENDING, dependency-held | `afterok:71049` | FineWeb-Edu MatrixPolicy configs 4-7 | 4 A6000 |
| `143611` | PENDING, dependency-held | `afterok:143591` | FineWeb-Edu MatrixPolicy configs 8-11 | 4 A6000 |

Useful dependency check:

```bash
scontrol show job 71049
scontrol show job 143550
scontrol show job 143584
scontrol show job 143591
scontrol show job 143611
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

Estimates below are deliberately rough because MatrixPolicy step time changes after warmup and Slurm scheduling can delay dependency release. Current time basis: 2026-06-04T14:19:04-04:00.

| item | expected window | basis |
| --- | --- | --- |
| `71048` active FineWeb-Edu AdamW shard | about 2026-06-04 late evening or later | job requeued once at 14:17 ET; first SiLU row completed after restart and the paired RLB row is active. This shard contains four AdamW configs, each with SiLU and RLB rows. |
| `71047` active DCLM MatrixPolicy shard | about 2026-06-04 17:30-18:45 ET | two configs complete; current row is at step 870/1525 and one config should remain after it. |
| `71049` release | after `71047` completes, likely evening 2026-06-04 | dependency is `afterok:71047`. |
| `143550` release | after restarted `71048` completes | dependency is `afterok:71048`; timing shifted later because of the restart. |
| all currently queued FineWeb-Edu Phase 1 shards | likely 2026-06-05 morning to afternoon | depends on MatrixPolicy row speed and Slurm backfill. |

Do not treat these as paper timing numbers; use curve summaries and Slurm accounting after completion.

## Pending Tasks By Condition

1. If either active job fails:
   - Do not submit more GPU work.
   - Check `sacct -j <job>` and `experiments/runs/logs/iclr-protocol-lock-<job>.out` first.
   - Patch the exact failing launcher/runtime path, commit, push, then rerun only the failed shard.

2. If active jobs complete:
   - Run:

```bash
.venv-cu128/bin/python experiments/scripts/summarize_iclr_phase1_protocol_lock.py \
  --run-root experiments/runs/iclr26_phase1_protocol_lock \
  --output-dir experiments/runs/iclr26_phase1_protocol_lock/summary
```

   - Refresh `experiments/results/iclr26_phase1_protocol_lock_20260604/` from the generated summary artifacts, then update this handoff. The summarizer emits `eval_curves.csv`, `train_curves.csv`, validation-loss/PPL plots, training-loss plots, AUC fields, and dense-curve checks; use those curve artifacts before final validation losses.
   - `143550` should release automatically after `71048`; `71049` should release automatically after `71047`. If either does not release, inspect its dependency state with `scontrol show job`.

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

1. Check `squeue` and `sacct` for `71047,71048,71049,143550,143584,143591,143611` plus completed jobs `67183,67184,69975,69976,71046`.
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
| `experiments/results/iclr26_phase1_protocol_lock_20260604/` | Tracked Phase 1 protocol-lock summaries, dense curves, rankings, and plots generated from ignored raw traces. |
| `.gitignore` | Ignores raw `experiments/runs/iclr26_phase1_protocol_lock/` outputs. |

Verified before this handoff rewrite: tracked filenames and tracked text contain no legacy tuning-surface matches outside ignored raw output/cache/log paths.

Paper PDF was already rendered with the Overleaf-style `pdflatex`/`bibtex` path and committed earlier at `paper/iclr_method_draft/main.pdf`.
