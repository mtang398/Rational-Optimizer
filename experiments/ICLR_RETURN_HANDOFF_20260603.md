# ICLR Return Handoff - 2026-06-03

Last updated: 2026-06-03T17:19:13-04:00
Current pushed commit: `c5ceac2`

## Current Slurm State

Active job:

| job | state | node | purpose | GPU use |
| --- | --- | --- | --- | --- |
| `65084` | RUNNING at 00:05:10 when checked | `sun-compute-03` | Phase 0C M1 DCLM smoke | 4 A6000 |

No other project jobs were active when this handoff was written. The active total is 4 A6000, under the 8 A6000 cap.

Useful checks when returning:

```bash
squeue -u mt872 -o "%.18i %.9P %.40j %.8T %.10M %.6D %R"
sacct -j 65084 --format=JobID,JobName%30,State,ExitCode,Elapsed,NodeList
sed -n 1,260p experiments/runs/logs/real-lm-screen-65084.out
```

## Completed Phase 0A/0B Smoke Results

All rows below used the compiled RLB extension path (`RATIONAL_OPT_TORCH_FALLBACK=0`), job-local extension build directories, exact dataset names/configs from the ICLR run plan, and explicit smoke validation skip (`VAL_SKIP_TOKENS=10000`, `DEFAULT_VAL_SKIP_TOKENS=10000`). Raw JSONL lives under ignored `experiments/runs/iclr26_smoke/`.

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

Interpretation: this is a feasibility smoke, not paper evidence. It is nevertheless a strong sanity signal: MatrixPolicy completed and was the best final-val row on all four Phase 0A/0B datasets.

## Current Phase 0C M1 Smoke

Command submitted as job `65084`:

```bash
env RATIONAL_OPT_TORCH_FALLBACK=0 REAL_LM_TASKS="dclm" SEEDS="1337" RUN_SUFFIX="iclr26_smoke_m1_dclm" OUTPUT_ROOT="experiments/runs/iclr26_smoke" TOKEN_CACHE_DIR="experiments/cache/tokens_iclr26_smoke" MAX_TRAIN_TOKENS=4000000 MAX_VAL_TOKENS=200000 STEPS=120 EVAL_INTERVAL=60 EVAL_BATCHES=2 LOG_INTERVAL=10 BATCH_SIZE=8 GRAD_ACCUM=4 INCLUDE_MUON=0 VAL_SKIP_TOKENS=10000 DEFAULT_VAL_SKIP_TOKENS=10000 COMMON_EXTRA_ARGS="--layers 18 --d-model 1024 --heads 16 --ffn-dim 3072" sbatch experiments/scripts/run_real_lm_screen_20260530.sh
```

Current observed M1 rows:

| row | status at handoff | latest val |
| --- | --- | ---: |
| `dclm_adamw_controls_iclr26_smoke_m1_dclm/silu` | complete | 6.8513 |
| `dclm_adamw_controls_iclr26_smoke_m1_dclm/rlb_fused_fixed_strong_ffn` | running, observed through step 10 | 10.5107 from step-1 eval only |
| `dclm_matrix_policy_groupstat_iclr26_smoke_m1_dclm/rlb_fused_fixed_strong_ffn` | not started at handoff | n/a |

Do not interpret the current RLB M1 val yet; it is only the step-1 eval before the scheduled step-60/120 evals.

## Pending Tasks by Condition

1. If job `65084` completes with exit code 0 and all three M1 JSONL summaries exist:
   - Record final M1 values in this file or a compact result summary.
   - Confirm no OOM, no NaN, and that the MatrixPolicy M1 row has a final eval at step 120.
   - Then start Phase 1 baseline protocol-lock jobs from `experiments/ICLR_EXACT_RUN_PLAN.md`, keeping at most two 4-GPU jobs active.

2. If job `65084` fails:
   - Do not start Phase 1.
   - Inspect `experiments/runs/logs/real-lm-screen-65084.out` and `sacct -j 65084` first.
   - If it is a memory failure, keep the same DCLM/M1 model geometry and fix the launch configuration before rerunning Phase 0C.
   - If it is a loader/build/runtime bug, patch the exact failing path, commit, push, and rerun only Phase 0C.

3. If Phase 0C passes:
   - Start Phase 1 only as protocol lock, not ablation and not HPO-centered evidence.
   - Use `dclm` and `fineweb_edu` as specified in the exact plan.
   - Keep raw outputs under ignored run directories; commit only code, plans, compact summaries, plots, and paper assets.

4. If Phase 1 passes:
   - Move to the main 100M M0 suite in the plan across `dclm`, `fineweb_edu`, `fineweb`, `dolma_sample`, and `c4_en` with seeds `1337`, `2027`, `3407`.
   - Do not start late ablations before main evidence is complete.

5. Always enforce these constraints:
   - Max 4 A6000 per job.
   - Max 8 A6000 active total.
   - Repo below 200G.
   - No substitute dataset/toolchain path for the planned experiment.
   - Keep FineWeb and FineWeb-Edu README curves/data intact.

## Repo/Infrastructure Changes Already Pushed

Recent pushed commits include:

| commit | purpose |
| --- | --- |
| `d733962` | Load Dolma sample through the official URL manifest instead of the unsupported legacy dataset script. |
| `8acf02f` | Use compiled RLB extension path for paper jobs by default. |
| `89289da` | Document `ninja` and `zstandard` runtime dependencies. |
| `3e2e79f` | Add explicit validation skip to Phase 0A/0B smoke commands. |
| `9813977` | Use Slurm-job-local extension build directories. |
| `e477c4d` | Add explicit validation skip to Phase 0C smoke command. |
| `c5ceac2` | Ignore raw Phase 0 smoke output directory. |

Paper PDF was already rendered with the Overleaf-style `pdflatex`/`bibtex` path and committed earlier at `paper/iclr_method_draft/main.pdf`.
