# ICLR Return Handoff - 2026-06-03

Last updated: 2026-06-03T17:35:02-04:00
Current pushed commit before this handoff update: `ccc5ab5`

## Current Slurm State

Active jobs:

| job | state | node | purpose | GPU use |
| --- | --- | --- | --- | --- |
| `67183` | RUNNING | `sun-compute-03` | Phase 1 protocol-lock DCLM AdamW control shard, configs 0-3 | 4 A6000 |
| `67184` | RUNNING | `fang-compute-02` | Phase 1 protocol-lock DCLM MatrixPolicy shard, configs 0-3 | 4 A6000 |

Active total: 8 A6000, exactly at the cap. Do not submit another GPU job until one of these exits.

Useful checks when returning:

```bash
squeue -u mt872 -o "%.18i %.9P %.40j %.8T %.10M %.6D %R"
sacct -j 67183,67184 --format=JobID,JobName%30,State,ExitCode,Elapsed,NodeList
sed -n '1,220p' experiments/runs/logs/iclr-protocol-lock-67183.out
sed -n '1,220p' experiments/runs/logs/iclr-protocol-lock-67184.out
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

## Submitted Continuation

The previous legacy-named submission `67175` was canceled before any result should be used. It was replaced by protocol-lock jobs with the new launcher `experiments/scripts/run_iclr_phase1_protocol_lock_20260603.sh`.

Submitted jobs:

| job | shard | command shape |
| --- | --- | --- |
| `67183` | DCLM AdamW control configs 0-3 | `TASKS=dclm`, `OPTIMIZER_FAMILIES=adamw`, `CONFIG_START=0`, `CONFIG_LIMIT=4` |
| `67184` | DCLM MatrixPolicy configs 0-3 | `TASKS=dclm`, `OPTIMIZER_FAMILIES=rational_matrix_policy_onpolicy`, `CONFIG_START=0`, `CONFIG_LIMIT=4` |

Both use:

```bash
MAX_TRAIN_TOKENS=50000000 MAX_VAL_TOKENS=4000000 STEPS=1525 EVAL_INTERVAL=50 EVAL_BATCHES=10 BATCH_SIZE=16 GRAD_ACCUM=2 DCLM_VAL_SKIP_TOKENS=110000000 COMMON_EXTRA_ARGS="--layers 18 --d-model 1024 --heads 16 --ffn-dim 3072"
```

## Pending Tasks by Condition

1. If `67183` or `67184` fails:
   - Do not submit more GPU work.
   - Check the corresponding log in `experiments/runs/logs/iclr-protocol-lock-<job>.out` and `sacct` first.
   - If the issue is launch/configuration, patch the exact failing path, commit, push, then rerun only the failed shard.

2. If both `67183` and `67184` complete:
   - Run `experiments/scripts/summarize_iclr_phase1_protocol_lock.py` on `experiments/runs/iclr26_phase1_protocol_lock`.
   - Record the compact DCLM protocol-lock summary in this handoff or a tracked result summary.
   - Decide the next two shards from `experiments/ICLR_EXACT_RUN_PLAN.md` while keeping the active total at or below 8 A6000.

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

## Repo/Infrastructure Changes Included in This Update

This update removes the old protocol surface with legacy tuning terminology and replaces it with:

| file | purpose |
| --- | --- |
| `experiments/scripts/run_iclr_phase1_protocol_lock_20260603.sh` | Protocol-lock Slurm launcher with DCLM support, bounded `CONFIG_START`/`CONFIG_LIMIT` shards, model-size hook, and optional extension-build guard. |
| `experiments/scripts/summarize_iclr_phase1_protocol_lock.py` | Protocol-lock JSONL summarizer with run CSV, ranking CSV, and Markdown summary outputs. |
| `.gitignore` | Ignores raw `experiments/runs/iclr26_phase1_protocol_lock/` outputs. |
| `optimizer_design/README.md` and `optimizer_design/baseline_optimizers.py` | Removes legacy tuning wording from optimizer infrastructure docs/comments. |

Paper PDF was already rendered with the Overleaf-style `pdflatex`/`bibtex` path and committed earlier at `paper/iclr_method_draft/main.pdf`.
