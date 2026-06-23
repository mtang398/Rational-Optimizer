# ICLR Experiment Command Log

This file records the commands used for the current manifest-based ICLR experiments. The
Slurm entrypoint is `experiments/scripts/run_iclr26_manifest_job.sh`. For each manifest
row, that launcher calls `bash training/run_lm_optimizer_sweep.sbatch`, which then runs
`training/transformer_lm_compare.py` with the dataset and method arguments from the
manifest.

## Manifest Generation

```bash
python3 experiments/scripts/build_iclr26_main_manifest.py \
  --output experiments/manifests/iclr26_main_manifest.csv \
  --print-summary
```

## E0 Preflight Submissions

Rows `0-14` are E0. They were submitted as two bounded chunks.

```bash
# Returned job 151609: E0 rows 0-8.
CONFIRM_ICLR26_MANIFEST=1 \
MANIFEST=experiments/manifests/iclr26_main_manifest.csv \
ROW_START=0 \
ROW_LIMIT=9 \
sbatch experiments/scripts/run_iclr26_manifest_job.sh
```

```bash
# Returned job 151610: E0 rows 9-14.
CONFIRM_ICLR26_MANIFEST=1 \
MANIFEST=experiments/manifests/iclr26_main_manifest.csv \
ROW_START=9 \
ROW_LIMIT=6 \
sbatch experiments/scripts/run_iclr26_manifest_job.sh
```

## E1 Main 100M Submissions

Rows `15-239` are E1. E1 was submitted in whole 15-row matched cells. Each job uses
4 A6000 GPUs. At most two jobs were intended to be active at once; later waves were
submitted with `afterok` dependencies.

```bash
# Returned job 155411: rows 15-29, dclm seed 1337.
CONFIRM_ICLR26_MANIFEST=1 \
MANIFEST=experiments/manifests/iclr26_main_manifest.csv \
ROW_START=15 \
ROW_LIMIT=15 \
sbatch experiments/scripts/run_iclr26_manifest_job.sh
```

```bash
# Returned job 155412: rows 30-44, dclm seed 2027.
CONFIRM_ICLR26_MANIFEST=1 \
MANIFEST=experiments/manifests/iclr26_main_manifest.csv \
ROW_START=30 \
ROW_LIMIT=15 \
sbatch experiments/scripts/run_iclr26_manifest_job.sh
```

```bash
# Returned job 158114: rows 45-59, dclm seed 3407.
CONFIRM_ICLR26_MANIFEST=1 \
MANIFEST=experiments/manifests/iclr26_main_manifest.csv \
ROW_START=45 \
ROW_LIMIT=15 \
sbatch --dependency=afterok:155411:155412 experiments/scripts/run_iclr26_manifest_job.sh
```

```bash
# Returned job 158115: rows 60-74, fineweb_edu seed 1337.
CONFIRM_ICLR26_MANIFEST=1 \
MANIFEST=experiments/manifests/iclr26_main_manifest.csv \
ROW_START=60 \
ROW_LIMIT=15 \
sbatch --dependency=afterok:155411:155412 experiments/scripts/run_iclr26_manifest_job.sh
```

```bash
# Returned job 158117: rows 75-89, fineweb_edu seed 2027.
CONFIRM_ICLR26_MANIFEST=1 \
MANIFEST=experiments/manifests/iclr26_main_manifest.csv \
ROW_START=75 \
ROW_LIMIT=15 \
sbatch --dependency=afterok:158114:158115 experiments/scripts/run_iclr26_manifest_job.sh
```

```bash
# Returned job 158118: rows 90-104, fineweb_edu seed 3407.
CONFIRM_ICLR26_MANIFEST=1 \
MANIFEST=experiments/manifests/iclr26_main_manifest.csv \
ROW_START=90 \
ROW_LIMIT=15 \
sbatch --dependency=afterok:158114:158115 experiments/scripts/run_iclr26_manifest_job.sh
```

```bash
# Returned job 158155: rows 105-119, fineweb seed 1337.
CONFIRM_ICLR26_MANIFEST=1 \
MANIFEST=experiments/manifests/iclr26_main_manifest.csv \
ROW_START=105 \
ROW_LIMIT=15 \
sbatch --dependency=afterok:158117:158118 experiments/scripts/run_iclr26_manifest_job.sh
```

```bash
# Returned job 158156: rows 120-134, fineweb seed 2027.
CONFIRM_ICLR26_MANIFEST=1 \
MANIFEST=experiments/manifests/iclr26_main_manifest.csv \
ROW_START=120 \
ROW_LIMIT=15 \
sbatch --dependency=afterok:158117:158118 experiments/scripts/run_iclr26_manifest_job.sh
```

```bash
# Returned job 158163: rows 135-149, fineweb seed 3407.
CONFIRM_ICLR26_MANIFEST=1 \
MANIFEST=experiments/manifests/iclr26_main_manifest.csv \
ROW_START=135 \
ROW_LIMIT=15 \
sbatch --dependency=afterok:158155:158156 experiments/scripts/run_iclr26_manifest_job.sh
```

```bash
# Returned job 158164: rows 150-164, dolma_sample seed 1337.
CONFIRM_ICLR26_MANIFEST=1 \
MANIFEST=experiments/manifests/iclr26_main_manifest.csv \
ROW_START=150 \
ROW_LIMIT=15 \
sbatch --dependency=afterok:158155:158156 experiments/scripts/run_iclr26_manifest_job.sh
```

```bash
# Returned job 158166: rows 165-179, dolma_sample seed 2027.
CONFIRM_ICLR26_MANIFEST=1 \
MANIFEST=experiments/manifests/iclr26_main_manifest.csv \
ROW_START=165 \
ROW_LIMIT=15 \
sbatch --dependency=afterok:158163:158164 experiments/scripts/run_iclr26_manifest_job.sh
```

```bash
# Returned job 158165: rows 180-194, dolma_sample seed 3407.
CONFIRM_ICLR26_MANIFEST=1 \
MANIFEST=experiments/manifests/iclr26_main_manifest.csv \
ROW_START=180 \
ROW_LIMIT=15 \
sbatch --dependency=afterok:158163:158164 experiments/scripts/run_iclr26_manifest_job.sh
```

```bash
# Returned job 158168: rows 195-209, c4_en seed 1337.
CONFIRM_ICLR26_MANIFEST=1 \
MANIFEST=experiments/manifests/iclr26_main_manifest.csv \
ROW_START=195 \
ROW_LIMIT=15 \
sbatch --dependency=afterok:158166:158165 experiments/scripts/run_iclr26_manifest_job.sh
```

```bash
# Returned job 158167: rows 210-224, c4_en seed 2027.
CONFIRM_ICLR26_MANIFEST=1 \
MANIFEST=experiments/manifests/iclr26_main_manifest.csv \
ROW_START=210 \
ROW_LIMIT=15 \
sbatch --dependency=afterok:158166:158165 experiments/scripts/run_iclr26_manifest_job.sh
```

```bash
# Returned job 158169: rows 225-239, c4_en seed 3407.
CONFIRM_ICLR26_MANIFEST=1 \
MANIFEST=experiments/manifests/iclr26_main_manifest.csv \
ROW_START=225 \
ROW_LIMIT=15 \
sbatch --dependency=afterok:158168:158167 experiments/scripts/run_iclr26_manifest_job.sh
```


E1 completion note: all jobs `155411`, `155412`, `158114`, `158115`, `158117`, `158118`, `158155`, `158156`, `158163`, `158164`, `158166`, `158165`, `158168`, `158167`, and `158169` completed. The final job `158169` covered rows 225-239 for `c4_en` seed 3407 and completed with exit `0:0` in 06:09:38.

## Rejected MatrixPolicy Variant Artifact Note

As of 2026-06-22, rejected V3/V4/V5 proposal files and standalone manifests have been deleted from the active repo surface. The paths in the historical command log below document what was submitted at the time; they are not live launch artifacts.

## matrixpolicyV3 E1 100M Submission

This is a separate E1-only rerun for the replacement RLB optimizer proposal. It uses `experiments/manifests/iclr26_matrixpolicyV3_e1_manifest.csv`, phase `E1_matrixpolicyV3_100m`, method `rlb_matrixpolicyV3`, activation `rlb_fused_fixed_strong_ffn`, optimizer `matrixpolicyV3`, and one manifest row per Slurm job.

The rows were submitted as two parity dependency chains so at most two 4-A6000 jobs from this set can run at once and preemption loss is limited to one row. `BUILD_EXT=0` was used because the extension had already been built and V3 is Python-side optimizer logic.

Submission template:

```bash
sbatch --parsable \
  --job-name=mpV3-e1-<row> \
  --export=ALL,CONFIRM_ICLR26_MANIFEST=1,MANIFEST=experiments/manifests/iclr26_matrixpolicyV3_e1_manifest.csv,ROW_START=<row>,ROW_LIMIT=1,BUILD_EXT=0 \
  [--dependency=afterok:<previous-chain-job>] \
  experiments/scripts/run_iclr26_manifest_job.sh
```

Submitted jobs:

| Row | Job | Dependency |
| ---: | ---: | --- |
| 0 | `690946` | none |
| 1 | `690947` | none |
| 2 | `690948` | afterok:`690946` |
| 3 | `690949` | afterok:`690947` |
| 4 | `690950` | afterok:`690948` |
| 5 | `690951` | afterok:`690949` |
| 6 | `690952` | afterok:`690950` |
| 7 | `690953` | afterok:`690951` |
| 8 | `690954` | afterok:`690952` |
| 9 | `690955` | afterok:`690953` |
| 10 | `690956` | afterok:`690954` |
| 11 | `690957` | afterok:`690955` |
| 12 | `690958` | afterok:`690956` |
| 13 | `690959` | afterok:`690957` |
| 14 | `690960` | afterok:`690958` |

Completion note: all jobs `690946`-`690960` completed with exit `0:0`. Job `690953` reported `Restarts=1`; its JSONL restarted cleanly and produced a final summary, but its full-step throughput is restart contaminated. The completed E1 aggregate is slightly worse than original MatrixPolicy on every dataset mean, so V3 is rejected/superseded and no V3 E2 jobs should be queued.

V3 is rejected. Its proposal/manifest artifacts were later removed from the active repo surface after V5 also failed.

## matrixpolicyV4 E1 100M Submission

This is a separate E1-only rerun for the functional-balance RLB optimizer proposal. It uses `experiments/manifests/iclr26_matrixpolicyV4_e1_manifest.csv`, phase `E1_matrixpolicyV4_100m`, method `rlb_matrixpolicyV4`, activation `rlb_fused_fixed_strong_ffn`, optimizer `matrixpolicyV4`, and one manifest row per Slurm job.

The rows were submitted as two parity dependency chains so at most two 4-A6000 jobs from this set can run at once and preemption loss is limited to one row. `BUILD_EXT=0` was used because the extension had already been built and V4 is Python-side optimizer logic.

Submission template:

```bash
sbatch --parsable \
  --job-name=mpV4-e1-<row> \
  --export=ALL,CONFIRM_ICLR26_MANIFEST=1,MANIFEST=experiments/manifests/iclr26_matrixpolicyV4_e1_manifest.csv,ROW_START=<row>,ROW_LIMIT=1,BUILD_EXT=0 \
  [--dependency=afterok:<previous-chain-job>] \
  experiments/scripts/run_iclr26_manifest_job.sh
```

Invalid first submitted jobs:

| Row | Job | Dependency |
| ---: | ---: | --- |
| 0 | `715013` | none |
| 1 | `715014` | none |
| 2 | `715015` | afterok:`715013` |
| 3 | `715016` | afterok:`715014` |
| 4 | `715017` | afterok:`715015` |
| 5 | `715018` | afterok:`715016` |
| 6 | `715019` | afterok:`715017` |
| 7 | `715020` | afterok:`715018` |
| 8 | `715021` | afterok:`715019` |
| 9 | `715022` | afterok:`715020` |
| 10 | `715023` | afterok:`715021` |
| 11 | `715024` | afterok:`715022` |
| 12 | `715025` | afterok:`715023` |
| 13 | `715026` | afterok:`715024` |
| 14 | `715027` | afterok:`715025` |

The first submission is invalid: rows `0-8` exited before training because `training/run_lm_optimizer_sweep.sbatch` did not yet include `matrixpolicyV4` in its hard-coded optimizer allowlist; rows `9-14` were cancelled. No JSONL outputs were produced. The wrapper fix was committed as `94d1352`.

Replacement submitted jobs:

| Row | Job | Dependency |
| ---: | ---: | --- |
| 0 | `715054` | none |
| 1 | `715055` | none |
| 2 | `715056` | afterok:`715054` |
| 3 | `715057` | afterok:`715055` |
| 4 | `715058` | afterok:`715056` |
| 5 | `715059` | afterok:`715057` |
| 6 | `715060` | afterok:`715058` |
| 7 | `715061` | afterok:`715059` |
| 8 | `715062` | afterok:`715060` |
| 9 | `715063` | afterok:`715061` |
| 10 | `715064` | afterok:`715062` |
| 11 | `715065` | afterok:`715063` |
| 12 | `715066` | afterok:`715064` |
| 13 | `715067` | afterok:`715065` |
| 14 | `715068` | afterok:`715066` |

Completion note: replacement jobs `715054`-`715068` all completed with exit `0:0` by `2026-06-21 16:53:52 EDT`. Jobs `715054` and `715055` had `Restarts=1` with incomplete pre-restart JSONLs archived; jobs `715056`-`715068` had `Restarts=0`. V4 near-tied original MatrixPolicy but is rejected/superseded because all `4590` recorded functional-balance log-ratio telemetry values clipped to `+0.47`, making the role-wise signal effectively constant and centered away. V4 is rejected. Its proposal/manifest artifacts were later removed from the active repo surface after V5 also failed.

## matrixpolicyV5 E1 100M Submission

Submitted: 2026-06-21 17:06:55 EDT. Commit: `70233f9`. Manifest at submission time: `experiments/manifests/iclr26_matrixpolicyV5_e1_manifest.csv`. This was an E1-only optimizer-geometry test. It later failed E1, no V5 E2 jobs were queued, and the proposal/manifest artifacts were removed from the active repo surface. The proposal explicitly disallowed engineering/fusion/kernel/cache speedups as optimizer evidence.

Submitted jobs:

| Row | Job | Dependency |
| ---: | ---: | --- |
| 0 | `716298` | none |
| 1 | `716299` | none |
| 2 | `716300` | afterok:`716298` |
| 3 | `716301` | afterok:`716299` |
| 4 | `716302` | afterok:`716300` |
| 5 | `716303` | afterok:`716301` |
| 6 | `716304` | afterok:`716302` |
| 7 | `716305` | afterok:`716303` |
| 8 | `716306` | afterok:`716304` |
| 9 | `716307` | afterok:`716305` |
| 10 | `716308` | afterok:`716306` |
| 11 | `716309` | afterok:`716307` |
| 12 | `716310` | afterok:`716308` |
| 13 | `716311` | afterok:`716309` |
| 14 | `716312` | afterok:`716310` |

Initial scheduler state after submission: jobs `716298` and `716299` were pending on priority; jobs `716300`-`716312` were dependency-held behind their parity-chain predecessors.

Launch-health update at `2026-06-21 17:12:02 EDT`: row `0` job `716298` was preempted once before training at `2026-06-21T17:07:38` on `monakhova-compute-01`, then requeued and restarted cleanly at `2026-06-21T17:10:38` with `Restarts=1`. Row `1` job `716299` is running on `sun-compute-03` with `Restarts=0`. The row `1` JSONL reached step `450` and logged nontrivial V5 role scaling (`in ~= 0.866`, `out ~= 1.155`), confirming the `matrixpolicyV5` optimizer path and functional-metric telemetry are active.

Completion note: all V5 E1 jobs `716298`-`716312` completed with exit `0:0` by `2026-06-21T22:37:04`. Jobs `716298` and `716304` had `Restarts=1` and clean final JSONL summaries; all other V5 jobs had `Restarts=0`. V5 failed the E1 acceptance gate: it improved only FineWeb-Edu (`-0.001404` vs original MatrixPolicy) and was neutral/slightly worse on DCLM, FineWeb, Dolma-sample, and C4. No V5 E2 jobs were queued.

## matrixpolicyV7 P0 500-Step Pilot Submission

Corrected submission: 2026-06-22 18:52:42 EDT. Manifest: `experiments/manifests/iclr26_matrixpolicyV7_p0_manifest.csv`. This is a P0 mechanism/loss smoke, not a full E1 run. It compares a fresh V1 control against `matrixpolicyV7` on DCLM seed `1337` for `500` steps. First submission jobs `727119` and `727120` were cancelled before run files were produced because the manifest used `val_tokens=1000000`; the corrected manifest uses the existing `val_tokens=4000000` DCLM validation cache.

```bash
sbatch --parsable \
  --job-name=mpV7-p0-v1 \
  --time=02:00:00 \
  --export=ALL,CONFIRM_ICLR26_MANIFEST=1,MANIFEST=experiments/manifests/iclr26_matrixpolicyV7_p0_manifest.csv,ROW_START=0,ROW_LIMIT=1,BUILD_EXT=0 \
  experiments/scripts/run_iclr26_manifest_job.sh
# returned 727161

sbatch --parsable \
  --job-name=mpV7-p0-cand \
  --time=02:00:00 \
  --export=ALL,CONFIRM_ICLR26_MANIFEST=1,MANIFEST=experiments/manifests/iclr26_matrixpolicyV7_p0_manifest.csv,ROW_START=1,ROW_LIMIT=1,BUILD_EXT=0 \
  experiments/scripts/run_iclr26_manifest_job.sh
# returned 727162
```

Corrected scheduler state at submission: both jobs were pending on resources immediately after submission.

Completion note: jobs `727161` and `727162` completed with exit `0:0` and `Restarts=0`. V7 was rejected after P0 because its final/AUC loss improvement was small while runtime was about `1.085x` paired V1; the V7 source hook, proposal file, and standalone manifest were pruned after the result was recorded in `ICLR_RUN_STATUS.md`.


## matrixpolicyV8 Fast-Pulse P0 Submission

Submitted: 2026-06-22 19:07:51 EDT. Manifest: `experiments/manifests/iclr26_matrixpolicyV8_fastpulse_p0_manifest.csv`. This is a three-row P0 pilot using the existing `rational_matrix_policy_onpolicy` optimizer only: V1 control, role-staged fast pulse, and lower-peak fast pulse.

```bash
sbatch --parsable \
  --job-name=mpV8-p0-v1 \
  --time=02:00:00 \
  --export=ALL,CONFIRM_ICLR26_MANIFEST=1,MANIFEST=experiments/manifests/iclr26_matrixpolicyV8_fastpulse_p0_manifest.csv,ROW_START=0,ROW_LIMIT=1,BUILD_EXT=0 \
  experiments/scripts/run_iclr26_manifest_job.sh
# returned 727338

sbatch --parsable \
  --job-name=mpV8-p0-fast \
  --time=02:00:00 \
  --export=ALL,CONFIRM_ICLR26_MANIFEST=1,MANIFEST=experiments/manifests/iclr26_matrixpolicyV8_fastpulse_p0_manifest.csv,ROW_START=1,ROW_LIMIT=1,BUILD_EXT=0 \
  experiments/scripts/run_iclr26_manifest_job.sh
# returned 727339

sbatch --parsable \
  --job-name=mpV8-p0-low \
  --time=02:00:00 \
  --export=ALL,CONFIRM_ICLR26_MANIFEST=1,MANIFEST=experiments/manifests/iclr26_matrixpolicyV8_fastpulse_p0_manifest.csv,ROW_START=2,ROW_LIMIT=1,BUILD_EXT=0 \
  experiments/scripts/run_iclr26_manifest_job.sh
# returned 727340
```

Scheduler state at 2026-06-22 19:15:48 EDT: all three jobs were pending on priority with no logs yet.

Completion note: jobs `727338`, `727339`, and `727340` completed with exit `0:0` and `Restarts=0`. V8 was rejected after P0: fast pulse worsened final loss/AUC with no same-node speedup, and lower-peak fast pulse was faster only on a different node while also worsening loss/AUC. The temporary V8 manifest was pruned after the result was recorded in `ICLR_RUN_STATUS.md`.


## matrixpolicyV9 Approximate-Muon P0 Submission

Submitted: 2026-06-22 20:28:38 EDT. Manifest: `experiments/manifests/iclr26_matrixpolicyV9_approx_muon_p0_manifest.csv`. This is a three-row P0 pilot using the existing `rational_matrix_policy_onpolicy` optimizer only: V1 control, Muon NS=3, and Muon NS=2.

```bash
sbatch --parsable \
  --job-name=mpV9-p0-v1 \
  --time=02:00:00 \
  --export=ALL,CONFIRM_ICLR26_MANIFEST=1,MANIFEST=experiments/manifests/iclr26_matrixpolicyV9_approx_muon_p0_manifest.csv,ROW_START=0,ROW_LIMIT=1,BUILD_EXT=0 \
  experiments/scripts/run_iclr26_manifest_job.sh
# returned 727913

sbatch --parsable \
  --job-name=mpV9-p0-ns3 \
  --time=02:00:00 \
  --export=ALL,CONFIRM_ICLR26_MANIFEST=1,MANIFEST=experiments/manifests/iclr26_matrixpolicyV9_approx_muon_p0_manifest.csv,ROW_START=1,ROW_LIMIT=1,BUILD_EXT=0 \
  experiments/scripts/run_iclr26_manifest_job.sh
# returned 727914

sbatch --parsable \
  --job-name=mpV9-p0-ns2 \
  --time=02:00:00 \
  --export=ALL,CONFIRM_ICLR26_MANIFEST=1,MANIFEST=experiments/manifests/iclr26_matrixpolicyV9_approx_muon_p0_manifest.csv,ROW_START=2,ROW_LIMIT=1,BUILD_EXT=0 \
  experiments/scripts/run_iclr26_manifest_job.sh
# returned 727915
```

Scheduler state at 2026-06-22 20:28:38 EDT: all three jobs were pending on priority. Estimated starts were `727913` at `20:33:55`, `727914` at `22:34:00`, and `727915` at `2026-06-23 00:34:00` EDT.

Completion note: jobs `727913`, `727914`, and `727915` completed with exit `0:0` and `Restarts=0` by `2026-06-22 20:43:48 EDT`. V9 was rejected after P0: NS=3 gave only a `0.45%` same-node total-time reduction while worsening final loss/AUC, and NS=2 worsened final loss/AUC further with non-comparable slower-node wall time. The temporary V9 manifest was pruned after the result was recorded in `ICLR_RUN_STATUS.md`.


## MatrixPolicy Safe Muon-Off Speed P0 Submission

Submitted: 2026-06-22 20:54:36 EDT. Manifest at submission time: `experiments/manifests/iclr26_matrixpolicy_safe_speed_p0_manifest.csv`. This was a three-row implementation-speed pilot using DCLM seed `1337` for `500` steps: SiLU+AdamW, RLB+AdamW, and original RLB+MatrixPolicy after commit `02b85d9` skips permanently inactive Muon steps. It was not a new Vx optimizer method.

Submitted jobs:

| Row | Job | Method |
| ---: | ---: | --- |
| 0 | `727991` | `silu_adamw` |
| 1 | `727990` | `rlb_adamw` |
| 2 | `727992` | `rlb_matrixpolicy_original` speed-fixed |

Completion note: jobs `727990`, `727991`, and `727992` completed with exit `0:0` and `Restarts=0` by `2026-06-22 22:32:09 EDT`. The MatrixPolicy speed-fixed row retained the expected 500-step quality band and improved mean optimizer-step time from the earlier same-node pre-speed control by about `17.5%`. The temporary P0 manifest was pruned after the result was recorded in `ICLR_RUN_STATUS.md`.

## matrixpolicyV10 Switch-Clean P0 Submission

Submitted: 2026-06-22 20:58:38 EDT. Manifest at submission time: `experiments/manifests/iclr26_matrixpolicyV10_switchclean_p0_manifest.csv`. This was a two-row DCLM seed `1337`, `500`-step P0 method pilot: original MatrixPolicy control and a switch-clean candidate that added only `--rational-matrix-policy-muon-reset-adam-state` to the original MatrixPolicy flags. It used the existing optimizer path; no new source alias was added.

Submitted jobs:

| Row | Job | Method |
| ---: | ---: | --- |
| 0 | `728006` | `rlb_matrixpolicy_original` |
| 1 | `728007` | `rlb_matrixpolicyV10_switchclean` |

Completion note: jobs `728006` and `728007` completed with exit `0:0` and `Restarts=0`. V10 was rejected after P0 because final loss worsened from `5.391717` to `5.424358` and AUC worsened from `6.367025` to `6.401291`. The temporary V10 manifest was pruned after the result was recorded in `ICLR_RUN_STATUS.md`.

## matrixpolicyV11 State-Adaptive Beta2 P0 Submission

Submitted: 2026-06-22 21:02:34 EDT. Manifest at submission time: `experiments/manifests/iclr26_matrixpolicyV11_stateadapt_p0_manifest.csv`. This was a two-row DCLM seed `1337`, `500`-step P0 method pilot: original MatrixPolicy control and a state-adaptive beta2 candidate that added only `--rational-matrix-policy-adam-beta2-final 0.98 --rational-matrix-policy-adam-beta2-decay-start 0.36 --rational-matrix-policy-adam-beta2-decay-end 0.50` to the original MatrixPolicy flags. It used the existing optimizer path; no new source alias was added.

Submitted jobs:

| Row | Job | Method |
| ---: | ---: | --- |
| 0 | `728025` | `rlb_matrixpolicy_original` |
| 1 | `728026` | `rlb_matrixpolicyV11_stateadapt_b98` |

Completion note: jobs `728025` and `728026` completed with exit `0:0` and `Restarts=0`. V11 was rejected after P0 because final loss worsened from `5.391785` to `5.394934` and AUC worsened from `6.367712` to `6.374737`; it also did not show a speed win. The temporary V11 manifest was pruned after the result was recorded in `ICLR_RUN_STATUS.md`.

## matrixpolicyV12 Selector-Beta2 P0 Submission

Submitted: 2026-06-22 21:07:09 EDT. Manifest at submission time: `experiments/manifests/iclr26_matrixpolicyV12_selector_beta2_p0_manifest.csv`. This was a one-row DCLM seed `1337`, `500`-step candidate-only P0 method pilot. It added only `--rational-matrix-policy-adam-beta2-input-final 0.98 --rational-matrix-policy-adam-beta2-decay-start 0.36 --rational-matrix-policy-adam-beta2-decay-end 0.50` to the original MatrixPolicy flags. It used fresh MatrixPolicy controls from the same pilot batch for comparison; no new source alias was added.

Submitted job:

| Row | Job | Method |
| ---: | ---: | --- |
| 0 | `728038` | `rlb_matrixpolicyV12_selector_beta2_b98` |

Completion note: job `728038` completed with exit `0:0` and `Restarts=0`. V12 was rejected after P0 because it worsened final loss and AUC against the clean fresh-control mean. The temporary V12 manifest was pruned after the result was recorded in `ICLR_RUN_STATUS.md`.

## E1 MatrixPolicy Safe-Speed Full Rerun Submission

Submitted: 2026-06-23 14:23:33 EDT. Manifest: `experiments/manifests/iclr26_matrixpolicy_safe_speed_e1_manifest.csv`. This is a full E1 rerun for original `rlb_matrixpolicy_original` under the method-preserving safe Muon-off speed fix from commit `02b85d9`. It is one row per job in two parity dependency chains, so at most two 4-A6000 jobs run concurrently and preemption loss is bounded to one row.

Submitted jobs:

| Row | Job | Dependency |
| ---: | ---: | --- |
| 0 | `767136` | none |
| 1 | `767137` | none |
| 2 | `767138` | afterok:`767136` |
| 3 | `767139` | afterok:`767137` |
| 4 | `767140` | afterok:`767138` |
| 5 | `767141` | afterok:`767139` |
| 6 | `767142` | afterok:`767140` |
| 7 | `767143` | afterok:`767141` |
| 8 | `767144` | afterok:`767142` |
| 9 | `767145` | afterok:`767143` |
| 10 | `767146` | afterok:`767144` |
| 11 | `767147` | afterok:`767145` |
| 12 | `767148` | afterok:`767146` |
| 13 | `767149` | afterok:`767147` |
| 14 | `767150` | afterok:`767148` |

Initial scheduler state after submission: jobs `767136` and `767137` were running on `ma-compute-01` and `monakhova-compute-01`; jobs `767138`-`767150` were dependency-held.

## E2 Main 300M Submissions

Rows `240-464` are E2. E2 initially started with whole 15-row matched cells,
but after confirming the 300M whole-cell runtime was too long for the
preemption risk, the first two DCLM cells were cancelled and resubmitted as
one-row jobs in two dependency chains. This keeps at most two 4-A6000 jobs
active while limiting preemption loss to one manifest row.

```bash
# Returned job 294600: rows 240-254, dclm seed 1337.
CONFIRM_ICLR26_MANIFEST=1 \
MANIFEST=experiments/manifests/iclr26_main_manifest.csv \
ROW_START=240 \
ROW_LIMIT=15 \
sbatch experiments/scripts/run_iclr26_manifest_job.sh
```

```bash
# Returned job 294599: rows 255-269, dclm seed 2027.
CONFIRM_ICLR26_MANIFEST=1 \
MANIFEST=experiments/manifests/iclr26_main_manifest.csv \
ROW_START=255 \
ROW_LIMIT=15 \
sbatch experiments/scripts/run_iclr26_manifest_job.sh
```

Status after launch check: both jobs started around `2026-06-08T15:38:00`; `294600` ran on `fang-compute-02` and `294599` ran on `lancer-compute-01`. Expected runtime after launch is about 30-36 hours for each DCLM E2 whole-cell job.

Cancellation/resubmission note: jobs `294600` and `294599` were cancelled at
about `2026-06-08 16:37 EDT` while still in their first rows. Rows `240-269`
were then resubmitted as one-row jobs:

| Row | Job | Dependency |
| ---: | ---: | --- |
| 240 | `294899` | none |
| 241 | `294900` | afterok:`294899` |
| 242 | `294901` | afterok:`294900` |
| 243 | `294902` | afterok:`294901` |
| 244 | `294903` | afterok:`294902` |
| 245 | `294904` | afterok:`294903` |
| 246 | `294905` | afterok:`294904` |
| 247 | `294906` | afterok:`294905` |
| 248 | `294907` | afterok:`294906` |
| 249 | `294908` | afterok:`294907` |
| 250 | `294909` | afterok:`294908` |
| 251 | `294910` | afterok:`294909` |
| 252 | `294911` | afterok:`294910` |
| 253 | `294912` | afterok:`294911` |
| 254 | `294913` | afterok:`294912` |
| 255 | `294914` | none |
| 256 | `294915` | afterok:`294914` |
| 257 | `294916` | afterok:`294915` |
| 258 | `294917` | afterok:`294916` |
| 259 | `294918` | afterok:`294917` |
| 260 | `294919` | afterok:`294918` |
| 261 | `294920` | afterok:`294919` |
| 262 | `294921` | afterok:`294920` |
| 263 | `294922` | afterok:`294921` |
| 264 | `294923` | afterok:`294922` |
| 265 | `294924` | afterok:`294923` |
| 266 | `294925` | afterok:`294924` |
| 267 | `294926` | afterok:`294925` |
| 268 | `294927` | afterok:`294926` |
| 269 | `294928` | afterok:`294927` |

Additional queued split submission on `2026-06-09 15:11 EDT`: rows `270-284`
were submitted as the DCLM seed-3407 one-row chain. Rows `285-299` were also
submitted at first, but this crossed the intended dataset boundary; those
FineWeb-Edu jobs were cancelled at `2026-06-09 15:24 EDT` before starting. The
active/queued E2 work is DCLM-only until rows `240-284` finish.

| Row | Job | Dependency |
| ---: | ---: | --- |
| 270 | `301071` | none |
| 271 | `301072` | afterok:`301071` |
| 272 | `301073` | afterok:`301072` |
| 273 | `301074` | afterok:`301073` |
| 274 | `301075` | afterok:`301074` |
| 275 | `301076` | afterok:`301075` |
| 276 | `301077` | afterok:`301076` |
| 277 | `301078` | afterok:`301077` |
| 278 | `301079` | afterok:`301078` |
| 279 | `301080` | afterok:`301079` |
| 280 | `301081` | afterok:`301080` |
| 281 | `301082` | afterok:`301081` |
| 282 | `301083` | afterok:`301082` |
| 283 | `301084` | afterok:`301083` |
| 284 | `301085` | afterok:`301084` |
| 285 | `301086` | afterok:`294928`; cancelled 2026-06-09 15:24 EDT |
| 286 | `301087` | afterok:`301086`; cancelled 2026-06-09 15:24 EDT |
| 287 | `301088` | afterok:`301087`; cancelled 2026-06-09 15:24 EDT |
| 288 | `301089` | afterok:`301088`; cancelled 2026-06-09 15:24 EDT |
| 289 | `301090` | afterok:`301089`; cancelled 2026-06-09 15:24 EDT |
| 290 | `301091` | afterok:`301090`; cancelled 2026-06-09 15:24 EDT |
| 291 | `301092` | afterok:`301091`; cancelled 2026-06-09 15:24 EDT |
| 292 | `301093` | afterok:`301092`; cancelled 2026-06-09 15:24 EDT |
| 293 | `301094` | afterok:`301093`; cancelled 2026-06-09 15:24 EDT |
| 294 | `301095` | afterok:`301094`; cancelled 2026-06-09 15:24 EDT |
| 295 | `301096` | afterok:`301095`; cancelled 2026-06-09 15:24 EDT |
| 296 | `301097` | afterok:`301096`; cancelled 2026-06-09 15:24 EDT |
| 297 | `301098` | afterok:`301097`; cancelled 2026-06-09 15:24 EDT |
| 298 | `301099` | afterok:`301098`; cancelled 2026-06-09 15:24 EDT |
| 299 | `301100` | afterok:`301099`; cancelled 2026-06-09 15:24 EDT |

DCLM seed-3407 resplit correction on `2026-06-09 17:39 EDT`: after rows
`269` and `270` completed, only one DCLM chain was active. Pending jobs
`301073`-`301085` were cancelled before starting, leaving running row `271`
job `301072` untouched. The remaining rows `272-284` were resubmitted as two
DCLM-only chains:

| Row | Job | Dependency |
| ---: | ---: | --- |
| 273 | `306116` | afterok:`301072` |
| 275 | `306117` | afterok:`306116` |
| 277 | `306118` | afterok:`306117` |
| 279 | `306119` | afterok:`306118` |
| 281 | `306120` | afterok:`306119` |
| 283 | `306121` | afterok:`306120` |
| 272 | `306122` | none |
| 274 | `306123` | afterok:`306122` |
| 276 | `306124` | afterok:`306123` |
| 278 | `306137` | afterok:`306124` |
| 280 | `306138` | afterok:`306137` |
| 282 | `306139` | afterok:`306138` |
| 284 | `306140` | afterok:`306139` |

Completion note: by `2026-06-10 13:44 EDT`, all DCLM E2 rows `240-284` had completed at final eval step `9150`. The final result package is `experiments/results/iclr26_e2_dclm_2026_06_10/`. No non-DCLM E2 jobs are active or queued; the accidental FineWeb-Edu rows `285-299` remained cancelled before start.

FineWeb-Edu E2 split submission on `2026-06-10 14:37 EDT`: after DCLM E2 rows `240-284` completed and the queue was empty, the next dataset window was submitted. This submission is FineWeb-Edu only: rows `285-329`, one manifest row per job, split into two dependency chains to keep at most two 4-A6000 jobs active.

Odd-row chain:

| Row | Job | Dependency |
| ---: | ---: | --- |
| 285 | `316996` | none |
| 287 | `316997` | afterok:`316996` |
| 289 | `316998` | afterok:`316997` |
| 291 | `316999` | afterok:`316998` |
| 293 | `317000` | afterok:`316999` |
| 295 | `317006` | afterok:`317000` |
| 297 | `317007` | afterok:`317006` |
| 299 | `317008` | afterok:`317007` |
| 301 | `317009` | afterok:`317008` |
| 303 | `317010` | afterok:`317009` |
| 305 | `317011` | afterok:`317010` |
| 307 | `317012` | afterok:`317011` |
| 309 | `317013` | afterok:`317012` |
| 311 | `317014` | afterok:`317013` |
| 313 | `317015` | afterok:`317014` |
| 315 | `317016` | afterok:`317015` |
| 317 | `317017` | afterok:`317016` |
| 319 | `317018` | afterok:`317017` |
| 321 | `317019` | afterok:`317018` |
| 323 | `317020` | afterok:`317019` |
| 325 | `317021` | afterok:`317020` |
| 327 | `317022` | afterok:`317021` |
| 329 | `317023` | afterok:`317022` |

Even-row chain:

| Row | Job | Dependency |
| ---: | ---: | --- |
| 286 | `317024` | none |
| 288 | `317025` | afterok:`317024` |
| 290 | `317026` | afterok:`317025` |
| 292 | `317027` | afterok:`317026` |
| 294 | `317028` | afterok:`317027` |
| 296 | `317029` | afterok:`317028` |
| 298 | `317030` | afterok:`317029` |
| 300 | `317031` | afterok:`317030` |
| 302 | `317032` | afterok:`317031` |
| 304 | `317033` | afterok:`317032` |
| 306 | `317034` | afterok:`317033` |
| 308 | `317039` | afterok:`317034` |
| 310 | `317040` | afterok:`317039` |
| 312 | `317041` | afterok:`317040` |
| 314 | `317042` | afterok:`317041` |
| 316 | `317043` | afterok:`317042` |
| 318 | `317044` | afterok:`317043` |
| 320 | `317045` | afterok:`317044` |
| 322 | `317046` | afterok:`317045` |
| 324 | `317047` | afterok:`317046` |
| 326 | `317048` | afterok:`317047` |
| 328 | `317049` | afterok:`317048` |

Rows `330+` were intentionally held at this point until FineWeb-Edu E2 rows `285-329` completed and the tracked result summaries/status files were updated. The later FineWeb-only submission is recorded below.

## E2 FineWeb 300M Split Submission

FineWeb E2 split submission on `2026-06-12 16:25 EDT`: after DCLM rows `240-284` and FineWeb-Edu rows `285-329` completed, and after the tracked result summaries/status files were updated, the next dataset window was submitted. This submission is FineWeb only: rows `330-374`, one manifest row per job, split into two dependency chains to keep at most two 4-A6000 jobs active. Rows `375+` were not submitted.

Even-row chain:

| Row | Job | Dependency |
| ---: | ---: | --- |
| 330 | `349422` | none |
| 332 | `349424` | afterok:`349422` |
| 334 | `349448` | afterok:`349424` |
| 336 | `349449` | afterok:`349448` |
| 338 | `349450` | afterok:`349449` |
| 340 | `349451` | afterok:`349450` |
| 342 | `349452` | afterok:`349451` |
| 344 | `349453` | afterok:`349452` |
| 346 | `349454` | afterok:`349453` |
| 348 | `349455` | afterok:`349454` |
| 350 | `349456` | afterok:`349455` |
| 352 | `349457` | afterok:`349456` |
| 354 | `349458` | afterok:`349457` |
| 356 | `349459` | afterok:`349458` |
| 358 | `349460` | afterok:`349459` |
| 360 | `349461` | afterok:`349460` |
| 362 | `349462` | afterok:`349461` |
| 364 | `349463` | afterok:`349462` |
| 366 | `349464` | afterok:`349463` |
| 368 | `349465` | afterok:`349464` |
| 370 | `349466` | afterok:`349465` |
| 372 | `349467` | afterok:`349466` |
| 374 | `349470` | afterok:`349467` |

Odd-row chain:

| Row | Job | Dependency |
| ---: | ---: | --- |
| 331 | `349471` | none |
| 333 | `349472` | afterok:`349471` |
| 335 | `349473` | afterok:`349472` |
| 337 | `349474` | afterok:`349473` |
| 339 | `349475` | afterok:`349474` |
| 341 | `349476` | afterok:`349475` |
| 343 | `349477` | afterok:`349476` |
| 345 | `349478` | afterok:`349477` |
| 347 | `349479` | afterok:`349478` |
| 349 | `349480` | afterok:`349479` |
| 351 | `349481` | afterok:`349480` |
| 353 | `349482` | afterok:`349481` |
| 355 | `349483` | afterok:`349482` |
| 357 | `349484` | afterok:`349483` |
| 359 | `349485` | afterok:`349484` |
| 361 | `349486` | afterok:`349485` |
| 363 | `349487` | afterok:`349486` |
| 365 | `349488` | afterok:`349487` |
| 367 | `349489` | afterok:`349488` |
| 369 | `349490` | afterok:`349489` |
| 371 | `349491` | afterok:`349490` |
| 373 | `349492` | afterok:`349491` |

Current scheduler state immediately after submission: jobs `349422` and `349471` were pending on priority; all later FineWeb rows were dependency-pending.

FineWeb E2 completion note: all jobs `349422`-`349492` completed with exit `0:0` and `Restarts=0`; the last FineWeb job ended on `2026-06-14T05:47:34`. Rows `375-419` were then submitted as the Dolma-sample-only E2 dataset window, recorded below.

## E2 Dolma-sample 300M Split Submission

Dolma-sample E2 split submission on `2026-06-15 13:42 EDT`: after FineWeb rows `330-374` completed and after the tracked result summaries/status files were updated, the next dataset window was submitted. This submission is Dolma-sample only: rows `375-419`, one manifest row per job. Row `375` is the cache/front row because the Dolma-sample E2 300M train and 610M+8M validation token caches were not present at submission time. Rows `376-419` are split into two dependency chains behind row `375`; rows `420+` were not submitted.

Front row:

| Row | Job | Dependency |
| ---: | ---: | --- |
| 375 | `393488` | none |

Even-row chain after row `375`:

| Row | Job | Dependency |
| ---: | ---: | --- |
| 376 | `393489` | afterok:`393488` |
| 378 | `393490` | afterok:`393489` |
| 380 | `393491` | afterok:`393490` |
| 382 | `393492` | afterok:`393491` |
| 384 | `393493` | afterok:`393492` |
| 386 | `393494` | afterok:`393493` |
| 388 | `393495` | afterok:`393494` |
| 390 | `393496` | afterok:`393495` |
| 392 | `393497` | afterok:`393496` |
| 394 | `393498` | afterok:`393497` |
| 396 | `393499` | afterok:`393498` |
| 398 | `393500` | afterok:`393499` |
| 400 | `393501` | afterok:`393500` |
| 402 | `393502` | afterok:`393501` |
| 404 | `393503` | afterok:`393502` |
| 406 | `393504` | afterok:`393503` |
| 408 | `393505` | afterok:`393504` |
| 410 | `393506` | afterok:`393505` |
| 412 | `393507` | afterok:`393506` |
| 414 | `393508` | afterok:`393507` |
| 416 | `393509` | afterok:`393508` |
| 418 | `393510` | afterok:`393509` |

Odd-row chain after row `375`:

| Row | Job | Dependency |
| ---: | ---: | --- |
| 377 | `393511` | afterok:`393488` |
| 379 | `393512` | afterok:`393511` |
| 381 | `393513` | afterok:`393512` |
| 383 | `393514` | afterok:`393513` |
| 385 | `393515` | afterok:`393514` |
| 387 | `393516` | afterok:`393515` |
| 389 | `393517` | afterok:`393516` |
| 391 | `393518` | afterok:`393517` |
| 393 | `393519` | afterok:`393518` |
| 395 | `393520` | afterok:`393519` |
| 397 | `393521` | afterok:`393520` |
| 399 | `393522` | afterok:`393521` |
| 401 | `393524` | afterok:`393522` |
| 403 | `393525` | afterok:`393524` |
| 405 | `393526` | afterok:`393525` |
| 407 | `393527` | afterok:`393526` |
| 409 | `393528` | afterok:`393527` |
| 411 | `393529` | afterok:`393528` |
| 413 | `393530` | afterok:`393529` |
| 415 | `393531` | afterok:`393530` |
| 417 | `393532` | afterok:`393531` |
| 419 | `393533` | afterok:`393532` |

Current scheduler state immediately after submission: job `393488` was running on `ellis-compute-02`; all later Dolma-sample rows were dependency-pending.

Dolma-sample E2 completion note: all jobs `393488`-`393533` completed with exit `0:0`; the last Dolma-sample job ended on `2026-06-17T02:00:27`. Jobs `393493`, `393501`, and `393521` show nonzero Slurm `Restarts`, but each corresponding JSONL has exactly one complete summary record at step `9150`. Dolma ADeMaMix variants diverged/non-finite on all three seeds and are recorded that way in `experiments/results/iclr26_e2_dolma_sample_2026_06_17/`.

## E2 C4 300M Split Submission

C4 E2 split submission on `2026-06-17 13:48 EDT`: after Dolma-sample rows `375-419` completed and after the tracked result summaries/status files were updated, the final E2 dataset window was submitted. This submission is C4 only: rows `420-464`, one manifest row per job. Row `420` is the cache/front row because the C4 E2 300M train and 8M validation token caches were not present at submission time. Rows `421-464` are split into two dependency chains behind row `420`; rows `465+` were not submitted because they are E3.

Front row:

| Row | Job | Dependency |
| ---: | ---: | --- |
| 420 | `476451` | none |

Odd-row chain after row `420`:

| Row | Job | Dependency |
| ---: | ---: | --- |
| 421 | `476452` | afterok:`476451` |
| 423 | `476453` | afterok:`476452` |
| 425 | `476454` | afterok:`476453` |
| 427 | `476455` | afterok:`476454` |
| 429 | `476456` | afterok:`476455` |
| 431 | `476457` | afterok:`476456` |
| 433 | `476458` | afterok:`476457` |
| 435 | `476459` | afterok:`476458` |
| 437 | `476460` | afterok:`476459` |
| 439 | `476461` | afterok:`476460` |
| 441 | `476462` | afterok:`476461` |
| 443 | `476463` | afterok:`476462` |
| 445 | `476464` | afterok:`476463` |
| 447 | `476465` | afterok:`476464` |
| 449 | `476466` | afterok:`476465` |
| 451 | `476467` | afterok:`476466` |
| 453 | `476468` | afterok:`476467` |
| 455 | `476469` | afterok:`476468` |
| 457 | `476470` | afterok:`476469` |
| 459 | `476471` | afterok:`476470` |
| 461 | `476472` | afterok:`476471` |
| 463 | `476473` | afterok:`476472` |

Even-row chain after row `420`:

| Row | Job | Dependency |
| ---: | ---: | --- |
| 422 | `476474` | afterok:`476451` |
| 424 | `476475` | afterok:`476474` |
| 426 | `476476` | afterok:`476475` |
| 428 | `476477` | afterok:`476476` |
| 430 | `476478` | afterok:`476477` |
| 432 | `476479` | afterok:`476478` |
| 434 | `476480` | afterok:`476479` |
| 436 | `476481` | afterok:`476480` |
| 438 | `476482` | afterok:`476481` |
| 440 | `476483` | afterok:`476482` |
| 442 | `476484` | afterok:`476483` |
| 444 | `476485` | afterok:`476484` |
| 446 | `476486` | afterok:`476485` |
| 448 | `476487` | afterok:`476486` |
| 450 | `476488` | afterok:`476487` |
| 452 | `476489` | afterok:`476488` |
| 454 | `476490` | afterok:`476489` |
| 456 | `476491` | afterok:`476490` |
| 458 | `476492` | afterok:`476491` |
| 460 | `476493` | afterok:`476492` |
| 462 | `476494` | afterok:`476493` |
| 464 | `476495` | afterok:`476494` |

Current scheduler state immediately after submission: job `476451` was pending on priority; all later C4 rows were dependency-pending.

C4 E2 completion note on `2026-06-19`: all jobs `476451`-`476495` completed with exit `0:0`; the last job was `476495`, ending at `2026-06-19T03:12:51`. Jobs `476453`, `476455`, `476476`, and `476481` show `Restarts=1`, but every C4 JSONL has exactly one complete summary record and final eval at step `9150`. The final E2 C4 package is tracked at `experiments/results/iclr26_e2_c4_2026_06_19/`.


## Internal Per-Row Command Shape

The manifest launcher converts each CSV row into environment variables and executes this
wrapper call:

```bash
RUN_NAME="${ROW_ROW_ID}" \
STEPS="${ROW_STEPS}" \
SEEDS="${ROW_SEED}" \
OPTIMIZERS="${ROW_OPTIMIZER}" \
ACTIVATIONS="${ROW_ACTIVATION}" \
EVAL_INTERVAL="${ROW_EVAL_INTERVAL}" \
EVAL_BATCHES="${ROW_EVAL_BATCHES}" \
LOG_INTERVAL="10" \
NPROC_PER_NODE="4" \
SKIP_BUILD_EXT="1" \
EXTRA_ARGS="--dataset-name ${ROW_DATASET_NAME} --dataset-config ${ROW_DATASET_CONFIG} --dataset-streaming --dataset-text-column ${ROW_TEXT_COLUMN} --train-split ${ROW_TRAIN_SPLIT} --validation-split ${ROW_VAL_SPLIT} --validation-skip-tokens ${ROW_VAL_SKIP_TOKENS} --cache-dir ${TOKEN_CACHE_DIR}/${ROW_DATASET} --output-dir ${OUTPUT_ROOT}/${ROW_PHASE}/${ROW_DATASET} --max-train-tokens ${ROW_TRAIN_TOKENS} --max-val-tokens ${ROW_VAL_TOKENS} --batch-size ${ROW_BATCH_SIZE} --grad-accum ${ROW_GRAD_ACCUM} --layers ${ROW_LAYERS} --d-model ${ROW_D_MODEL} --heads ${ROW_HEADS} --ffn-dim ${ROW_FFN_DIM} --lr ${ROW_LR} --min-lr ${ROW_MIN_LR} --weight-decay ${ROW_WEIGHT_DECAY} --probe-batch-size 1 --matrix-spectrum-interval 250 ${ROW_EXTRA_ARGS} ${COMMON_EXTRA_ARGS}" \
bash training/run_lm_optimizer_sweep.sbatch
```

## Result Regeneration Commands

E1 figures and checkpoint tables:

```bash
python3 experiments/scripts/plot_iclr26_e1_curves.py \
  --status-md experiments/ICLR_RUN_STATUS.md
```

E1 token-to-target savings:

```bash
python3 experiments/scripts/summarize_iclr26_e1_token_savings.py
```

E2 completed-cell summaries:

```bash
python3 experiments/scripts/summarize_iclr26_e2_dataset.py \
  --dataset dclm \
  --output-dir experiments/results/iclr26_e2_dclm_2026_06_10 \
  --completed-date 2026-06-10

python3 experiments/scripts/summarize_iclr26_e2_dataset.py \
  --dataset fineweb_edu \
  --output-dir experiments/results/iclr26_e2_fineweb_edu_2026_06_12 \
  --completed-date 2026-06-12

python3 experiments/scripts/summarize_iclr26_e2_dataset.py \
  --dataset fineweb \
  --output-dir experiments/results/iclr26_e2_fineweb_2026_06_15 \
  --completed-date 2026-06-15

python3 experiments/scripts/summarize_iclr26_e2_dataset.py \
  --dataset dolma_sample \
  --output-dir experiments/results/iclr26_e2_dolma_sample_2026_06_17 \
  --completed-date 2026-06-17

python3 experiments/scripts/summarize_iclr26_e2_dataset.py \
  --dataset c4_en \
  --output-dir experiments/results/iclr26_e2_c4_2026_06_19 \
  --completed-date 2026-06-19
```

E2 dense curve figures and checkpoint tables:

```bash
python3 experiments/scripts/plot_iclr26_e2_curves.py
```

Clean runtime tables for completed paper cells:

```bash
python3 experiments/scripts/summarize_iclr26_runtimes.py
```
