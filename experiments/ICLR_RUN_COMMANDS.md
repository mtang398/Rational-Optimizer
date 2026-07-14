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

Rejected proposal launch artifacts have been pruned from the active repo surface and raw run tree. The single retained negative-result state is `optimizer_design/proposals/matrixpolicy_variant_failures.md`; detailed rejected-variant submission blocks are intentionally not retained here.

## RLB Ablation Full Queue

Corrected submission: 2026-06-24 16:02 EDT. Manifest: `experiments/manifests/iclr26_rational_only_ablation_manifest.csv`. Scope: E1 + E2 across five datasets x three seeds per phase using `rlb_fused_global_rational` and the original `rational_matrix_policy_onpolicy` optimizer settings.

Definition: this is the intended RLB control. It keeps the RLB single-branch MLP wrapper, group RMS normalization, trainable grouped SiLU-fitted P5/Q4 rational, telemetry, gauge/stat hooks, and MatrixPolicy settings. The activation has no `coeff_logits` parameter; trainable rational parameters are only `numerator` and `denominator`.

Correction: the 2026-06-24 15:36 queue used `rlb_fused_rational_only`, which still created a zero-sized `coeff_logits` parameter. That queue is rejected for this ablation. Jobs `830651`-`830681` completed under the flawed definition, jobs `830683` and `830685` were cancelled while active, and jobs `830688`-`830717` were cancelled before starting.

Submission shape: one manifest row per Slurm job, two parity dependency chains, `--constraint=nvlink`, and at most two active 4-A6000 jobs. The launcher has an NVLink timing guard for `E1_rational_only_100m` and `E2_rational_only_300m`.

Submitted jobs:

| Rows | Jobs | Chain |
| --- | --- | --- |
| even rows `0,2,...,28` | `835104`, `835105`, `835106`, `835107`, `835108`, `835109`, `835110`, `835111`, `835112`, `835121`, `835122`, `835124`, `835125`, `835126`, `835127` | terminal `835127` |
| odd rows `1,3,...,29` | `835128`, `835129`, `835130`, `835131`, `835132`, `835133`, `835134`, `835135`, `835137`, `835138`, `835139`, `835140`, `835141`, `835142`, `835143` | terminal `835143` |

Initial state at 2026-06-24 16:02 EDT: chain heads `835104` and `835128` were pending; by 2026-06-24 16:04 EDT, `835104` was running on `rush-compute-03` and `835128` was running on `ellis-compute-02`; all later rows were dependency-held.

The submission used this pattern, once per row with the previous same-parity job passed as `--dependency=afterok:<job>`:

```bash
CONFIRM_ICLR26_MANIFEST=1 \
MANIFEST=experiments/manifests/iclr26_rational_only_ablation_manifest.csv \
ROW_START=<row> \
ROW_LIMIT=1 \
sbatch --parsable --constraint=nvlink [--dependency=afterok:<previous_same_parity_job>] \
  experiments/scripts/run_iclr26_manifest_job.sh
```

## RLB Optimizer-Control Sweep

Submitted: 2026-06-25 15:10 EDT. Manifest: `experiments/manifests/iclr26_global_rational_optimizer_controls_manifest.csv`. Scope: all non-MatrixPolicy RLB optimizer controls under the corrected RLB activation.

Definition: this keeps each existing non-MatrixPolicy optimizer recipe fixed (`adamw`, `muon`, `lion`, `soap_adamw`, `ademamix`, `adafactor_came`, `schedule_free_adamw`) and swaps only the RLB activation to `rlb_fused_global_rational`. It has trainable rational parameters only in `numerator` and `denominator`. MatrixPolicy rows are not in this manifest because the completed `E1_rational_only_100m` and `E2_rational_only_300m` rows already provide the RLB MatrixPolicy overlay.

Submission shape: one manifest row per Slurm job, two parity dependency chains with `afterany` dependencies, `--constraint=nvlink`, and at most two active 4-A6000 jobs. The launcher NVLink timing guard includes `E1_global_rational_optimizers_100m` and `E2_global_rational_optimizers_300m`.

Submitted jobs:

| Rows | Jobs | Chain |
| --- | --- | --- |
| even rows `0,2,...,208` | `982026`-`982236` scheduler range, excluding non-sweep gaps | terminal `982236` |
| odd rows `1,3,...,209` | `982027`-`982237` scheduler range, excluding non-sweep gaps | terminal `982237` |

Initial state at 2026-06-25 15:10 EDT: jobs `982026` and `982027` started on NVLink nodes; later rows were dependency-held. Early logs wrote RLB JSONLs and confirmed legacy basis telemetry is null. Completion note on 2026-06-29: the sweep finished with `180` full completions and `30` RLB+ADeMaMix early stops from non-finite loss; no JSONLs were missing, partial, or bad. The regenerated E1/E2 paper-facing packages now overlay these rows for every non-MatrixPolicy RLB optimizer control.

The submission used this pattern, once per row with the previous same-parity job passed as `--dependency=afterany:<job>`:

```bash
sbatch --parsable --constraint=nvlink [--dependency=afterany:<previous_same_parity_job>]   --job-name=grlb-<row>   --export=ALL,CONFIRM_ICLR26_MANIFEST=1,MANIFEST=experiments/manifests/iclr26_global_rational_optimizer_controls_manifest.csv,ROW_START=<row>,ROW_LIMIT=1,BUILD_EXT=0   experiments/scripts/run_iclr26_manifest_job.sh
```

## MatrixPolicy Safe Muon-Off Speed P0 Submission

Submitted: 2026-06-22 20:54:36 EDT. Manifest at submission time: `experiments/manifests/iclr26_matrixpolicy_safe_speed_p0_manifest.csv`. This was a three-row implementation-speed pilot using DCLM seed `1337` for `500` steps: SiLU+AdamW, RLB+AdamW, and original RLB+MatrixPolicy after commit `02b85d9` skips permanently inactive Muon steps. It was not a new optimizer method.

Submitted jobs:

| Row | Job | Method |
| ---: | ---: | --- |
| 0 | `727991` | `silu_adamw` |
| 1 | `727990` | `rlb_adamw` |
| 2 | `727992` | `rlb_matrixpolicy_original` speed-fixed |

Completion note: jobs `727990`, `727991`, and `727992` completed with exit `0:0` and `Restarts=0` by `2026-06-22 22:32:09 EDT`. The MatrixPolicy speed-fixed row retained the expected 500-step quality band and improved mean optimizer-step time from the earlier same-node pre-speed control by about `17.5%`. The temporary P0 manifest was pruned after the result was recorded in `ICLR_RUN_STATUS.md`.

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

Completion note: all jobs `767136`-`767150` completed with exit `0:0` by `2026-06-23 18:16:55 EDT`. Job `767137` had `Restarts=1`; its preempted partial JSONL was archived as `.incomplete_767137_1_20260623150154`, and the final clean rerun is the only row included in aggregates. The completed rerun passes the E1 acceptance gate: final losses match the original MatrixPolicy E1 table within seed/dataset noise. This established the historical 15-row safe-speed MatrixPolicy aggregate (`27.3` min, `0.5102` s/step, `67,078.3` tokens/s); the current paper-facing runtime summary is later superseded by the RLB MatrixPolicy overlay.

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



## E2 MatrixPolicy Safe-Speed Timing Rerun Submission

Submitted: 2026-06-23 20:06:40 EDT. Manifest: `experiments/manifests/iclr26_matrixpolicy_safe_speed_e2_manifest.csv`. This is the E2 timing rerun for original `rlb_matrixpolicy_original` under the accepted method-preserving safe Muon-off implementation. It mirrors the completed E2 MatrixPolicy rows only: five datasets x three seeds, `300M` train tokens per row, one manifest row per Slurm job.

The rows were submitted as two parity dependency chains, so at most two 4-A6000 jobs from this set run at once and preemption loss is limited to one row.

Submission template:

```bash
sbatch --parsable \
  --job-name=mp-safe-e2-<row> \
  --export=ALL,CONFIRM_ICLR26_MANIFEST=1,MANIFEST=experiments/manifests/iclr26_matrixpolicy_safe_speed_e2_manifest.csv,ROW_START=<row>,ROW_LIMIT=1,BUILD_EXT=0 \
  [--dependency=afterok:<previous-chain-job>] \
  experiments/scripts/run_iclr26_manifest_job.sh
```

Submitted jobs:

| Row | Job | Dependency |
| ---: | ---: | --- |
| 0 | `810092` | none |
| 1 | `810093` | none |
| 2 | `810094` | afterok:`810092` |
| 3 | `810095` | afterok:`810093` |
| 4 | `810096` | afterok:`810094` |
| 5 | `810097` | afterok:`810095` |
| 6 | `810098` | afterok:`810096` |
| 7 | `810099` | afterok:`810097` |
| 8 | `810100` | afterok:`810098` |
| 9 | `810101` | afterok:`810099` |
| 10 | `810102` | afterok:`810100` |
| 11 | `810103` | afterok:`810101` |
| 12 | `810104` | afterok:`810102` |
| 13 | `810105` | afterok:`810103` |
| 14 | `810106` | afterok:`810104` |

Initial scheduler state after submission: jobs `810092` and `810093` were running on `lancer-compute-01` and `monakhova-compute-01`; jobs `810094`-`810106` were dependency-held.

Correction on 2026-06-23 21:36 EDT: job `810093` showed real per-step runtime around `1.8-2.0` s/step on `monakhova-compute-01`, while `810092`/`810094` on NVLink nodes were around `0.44-0.46` s/step. Slurm reported `Restarts=0`, so this was not a restart-accounting artifact. The row was rejected as a timing-contaminated allocation, and jobs `810093`, `810095`, `810097`, `810099`, `810101`, `810103`, and `810105` were cancelled. Replacement odd-chain rows were submitted with `--constraint=nvlink`; replacement job `812522` archived the partial JSONL as `rlb_fused_fixed_strong_ffn.jsonl.incomplete_812522_0_20260623213620`.

Replacement odd-chain jobs:

| Row | Job | Dependency |
| ---: | ---: | --- |
| 1 | `812522` | none |
| 3 | `812523` | afterok:`812522` |
| 5 | `812524` | afterok:`812523` |
| 7 | `812525` | afterok:`812524` |
| 9 | `812526` | afterok:`812525` |
| 11 | `812527` | afterok:`812526` |
| 13 | `812528` | afterok:`812527` |

The E2 safe-speed terminal jobs are now `810106` for the original even chain and `812528` for the replacement odd chain. At 2026-06-23 21:41 EDT, the still-pending original even-chain jobs `810096`, `810098`, `810100`, `810102`, `810104`, and `810106` were updated with `Features=nvlink`; `scontrol show job` verified the constraint on all six jobs before they started. The manifest launcher also has a timing-row NVLink guard: phases `E1_matrixpolicy_safe_speed_100m`, `E2_matrixpolicy_safe_speed_300m`, and `E1_fineweb_edu_seed2027_runtime_repair_100m` exit before JSONL archive/write if the allocated node lacks the `nvlink` feature, unless `ALLOW_NON_NVLINK_TIMING=1` is explicitly set.

Completion note on 2026-06-24: clean E2 safe-speed jobs `810092`, `810094`, `810096`, `810098`, `810100`, `810102`, `810104`, `810106`, and replacement odd-chain jobs `812522`-`812528` all completed with exit `0:0` and `Restarts=0`. These JSONL `summary.total_seconds` rows validated clean E2 RLB+MatrixPolicy timing and exclude cancelled non-NVLink job `810093` plus its cancelled odd-chain descendants; the current paper-facing runtime summary is later superseded by the RLB MatrixPolicy overlay.

## E1 FineWeb-Edu Seed 2027 Runtime Repair Submission

Submitted: 2026-06-23 20:43:47 EDT. Manifest: `experiments/manifests/iclr26_e1_fineweb_edu_seed2027_runtime_repair_manifest.csv`. This is a clean timing repair for original E1 rows `81-88`; the original job `158117` was preempted six times, so those original row artifacts are not used as trusted runtime measurements.

The repair rows were initially submitted as two parity dependency chains after E2 safe-speed terminal jobs `810105` and `810106`; after `810105` was cancelled with the contaminated odd chain, the replacement repair rows depended on clean terminals `810106` and `812528`.

Submission template:

```bash
sbatch --parsable \
  --job-name=e1-time-repair-<original-row> \
  --dependency=afterok:<previous-or-e2-terminal-jobs> \
  --export=ALL,CONFIRM_ICLR26_MANIFEST=1,MANIFEST=experiments/manifests/iclr26_e1_fineweb_edu_seed2027_runtime_repair_manifest.csv,ROW_START=<repair-offset>,ROW_LIMIT=1,BUILD_EXT=0 \
  experiments/scripts/run_iclr26_manifest_job.sh
```

Submitted jobs:

| Repair offset | Original row | Job | Dependency |
| ---: | ---: | ---: | --- |
| 0 | 81 | `811802` | afterok:`810105`:`810106` |
| 1 | 82 | `811803` | afterok:`810105`:`810106` |
| 2 | 83 | `811804` | afterok:`811802` |
| 3 | 84 | `811805` | afterok:`811803` |
| 4 | 85 | `811806` | afterok:`811804` |
| 5 | 86 | `811807` | afterok:`811805` |
| 6 | 87 | `811808` | afterok:`811806` |
| 7 | 88 | `811809` | afterok:`811807` |

Initial scheduler state after submission: jobs `811802`-`811809` were dependency-held behind the E2 safe-speed chain; jobs `810092` and `810093` were running, while `810094`-`810106` were still dependency-held.

Correction on 2026-06-23 21:37 EDT: because old E2 terminal job `810105` was cancelled with the timing-contaminated odd chain, repair jobs `811802`-`811809` were cancelled and replaced with NVLink-constrained jobs depending on the clean E2 terminals `810106` and `812528`. Internal repair-chain dependencies were verified as explicit `afterok`.

Replacement repair jobs:

| Repair offset | Original row | Job | Dependency |
| ---: | ---: | ---: | --- |
| 0 | 81 | `812529` | afterok:`810106`:`812528` |
| 1 | 82 | `812530` | afterok:`810106`:`812528` |
| 2 | 83 | `812531` | afterok:`812529` |
| 3 | 84 | `812532` | afterok:`812530` |
| 4 | 85 | `812533` | afterok:`812531` |
| 5 | 86 | `812534` | afterok:`812532` |
| 6 | 87 | `812535` | afterok:`812533` |
| 7 | 88 | `812536` | afterok:`812534` |

Completion note on 2026-06-24: replacement repair jobs `812529`-`812536` all completed with exit `0:0` and `Restarts=0`. The regenerated runtime summary overlays these eight rows and restores SOAP, ADeMaMix, CAME, and ScheduleFree to 15 E1 timing runs.

## RLB MatrixPolicy Timing Node Repair

Policy correction on 2026-06-25 15:54 EDT: repair is based on bad-node provenance, not a universal seconds-per-step cutoff. The old pending strict-cutoff repair jobs `984723`, `984724`, and `984725` were cancelled before starting. Downstream RLB optimizer-control heads `982030` and `982031` were held during the replacement, then updated to depend on the new terminal repair job and released.

Audited legacy-node mapping is recorded in `experiments/manifests/iclr26_timing_node_overrides.csv`; it is only used when a legacy JSONL lacks `slurm_node`. New repaired JSONLs carry their own Slurm metadata and bypass that override.

Replacement submissions used `FORCE_RERUN_COMPLETE_JSONL=1`, `TIMING_NODE_DENYLIST=sablab-gpu-12`, `TIMING_GUARD_MAX_SECONDS_PER_STEP=0`, and `TIMING_GUARD_MAX_REQUEUES=6`:

```bash
env CONFIRM_ICLR26_MANIFEST=1 MANIFEST=experiments/manifests/iclr26_global_rational_matrixpolicy_manifest.csv ROW_START=21 ROW_LIMIT=1 FORCE_RERUN_COMPLETE_JSONL=1 TIMING_NODE_DENYLIST=sablab-gpu-12 TIMING_GUARD_MAX_SECONDS_PER_STEP=0 TIMING_GUARD_MAX_REQUEUES=6 \
  sbatch --parsable --job-name=mprepair-node-21 --constraint=nvlink --dependency=afterany:982028:982029 experiments/scripts/run_iclr26_manifest_job.sh
# returned 986793

env CONFIRM_ICLR26_MANIFEST=1 MANIFEST=experiments/manifests/iclr26_global_rational_matrixpolicy_manifest.csv ROW_START=26 ROW_LIMIT=1 FORCE_RERUN_COMPLETE_JSONL=1 TIMING_NODE_DENYLIST=sablab-gpu-12 TIMING_GUARD_MAX_SECONDS_PER_STEP=0 TIMING_GUARD_MAX_REQUEUES=6 \
  sbatch --parsable --job-name=mprepair-node-26 --constraint=nvlink --dependency=afterany:982028:982029 experiments/scripts/run_iclr26_manifest_job.sh
# returned 986794

env CONFIRM_ICLR26_MANIFEST=1 MANIFEST=experiments/manifests/iclr26_global_rational_matrixpolicy_manifest.csv ROW_START=28 ROW_LIMIT=1 FORCE_RERUN_COMPLETE_JSONL=1 TIMING_NODE_DENYLIST=sablab-gpu-12 TIMING_GUARD_MAX_SECONDS_PER_STEP=0 TIMING_GUARD_MAX_REQUEUES=6 \
  sbatch --parsable --job-name=mprepair-node-28 --constraint=nvlink --dependency=afterany:986793:986794 experiments/scripts/run_iclr26_manifest_job.sh
# returned 986795

scontrol update JobId=982030 Dependency=afterany:986795
scontrol update JobId=982031 Dependency=afterany:986795
scontrol release 982030 982031
```

Final dependency check after replacement:

| Job | State | Reason | Name | Dependency |
| ---: | --- | --- | --- | --- |
| `986793` | PENDING | Dependency | `mprepair-node-21` | afterany:`982029` (`982028` already fulfilled) |
| `986794` | PENDING | Dependency | `mprepair-node-26` | afterany:`982029` (`982028` already fulfilled) |
| `986795` | PENDING | Dependency | `mprepair-node-28` | afterany:`986793`:`986794` |
| `982030` | PENDING | Dependency | `grlb-004` | afterany:`986795` |
| `982031` | PENDING | Dependency | `grlb-005` | afterany:`986795` |

Verification commands now fail on the known legacy bad rows by node, not by speed threshold, until the repair jobs replace their JSONLs:

```bash
python3 experiments/scripts/summarize_iclr26_runtimes.py \
  --safe-e1-matrixpolicy-manifest experiments/manifests/iclr26_global_rational_matrixpolicy_manifest.csv \
  --safe-e2-matrixpolicy-manifest experiments/manifests/iclr26_global_rational_matrixpolicy_manifest.csv
# RuntimeError: ... denylisted_slurm_node=sablab-gpu-12 ... Rerun/repair this row; do not exclude it from aggregates.

python3 experiments/scripts/summarize_iclr26_e2_dataset.py \
  --dataset fineweb \
  --output-dir /tmp/iclr26_e2_fineweb_node_guard_check \
  --matrixpolicy-manifest experiments/manifests/iclr26_global_rational_matrixpolicy_manifest.csv \
  --matrixpolicy-phase E2_rational_only_300m
# RuntimeError: ... denylisted_slurm_node=sablab-gpu-12 ... Rerun/repair this row; do not exclude it from aggregates.
```

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

The paper-facing MatrixPolicy source is the validated live-statistic correction
campaign:

```bash
CORRECTED_MP_MANIFEST=experiments/corrections/matrixpolicy_live_stats_20260712/manifests/matrixpolicy_live_stats_20260712_main.csv
CORRECTED_MP_ROOT=experiments/corrections/matrixpolicy_live_stats_20260712/runs/main
RLB_CONTROL_MANIFEST=experiments/manifests/iclr26_global_rational_optimizer_controls_manifest.csv
```

E1 figures and checkpoint tables:

```bash
python3 experiments/scripts/plot_iclr26_e1_curves.py \
  --matrixpolicy-manifest "$CORRECTED_MP_MANIFEST" \
  --matrixpolicy-run-root "$CORRECTED_MP_ROOT/E1_rational_only_100m" \
  --matrixpolicy-phase E1_rational_only_100m \
  --replacement-manifest "$RLB_CONTROL_MANIFEST" \
  --replacement-run-root experiments/runs/iclr26_main/E1_global_rational_optimizers_100m \
  --replacement-phase E1_global_rational_optimizers_100m \
  --status-md experiments/ICLR_RUN_STATUS.md
```

E1 token-to-target savings:

```bash
python3 experiments/scripts/summarize_iclr26_e1_token_savings.py \
  --matrixpolicy-manifest "$CORRECTED_MP_MANIFEST" \
  --matrixpolicy-run-root "$CORRECTED_MP_ROOT" \
  --matrixpolicy-phase E1_rational_only_100m \
  --replacement-manifest "$RLB_CONTROL_MANIFEST" \
  --replacement-phase E1_global_rational_optimizers_100m
```

E2 completed-cell summaries:

```bash
python3 experiments/scripts/summarize_iclr26_e2_dataset.py \
  --dataset DATASET \
  --output-dir OUTPUT_DIRECTORY \
  --completed-date ORIGINAL_COMPLETION_DATE \
  --matrixpolicy-manifest "$CORRECTED_MP_MANIFEST" \
  --matrixpolicy-run-root "$CORRECTED_MP_ROOT" \
  --matrixpolicy-phase E2_rational_only_300m \
  --replacement-manifest "$RLB_CONTROL_MANIFEST" \
  --replacement-phase E2_global_rational_optimizers_300m
```

Run the command for `dclm`, `fineweb_edu`, `fineweb`, `dolma_sample`, and
`c4_en`, retaining their tracked output directories and original completion
dates.

E2 dense curve figures and checkpoint tables:

```bash
python3 experiments/scripts/plot_iclr26_e2_curves.py \
  --matrixpolicy-manifest "$CORRECTED_MP_MANIFEST" \
  --matrixpolicy-run-root "$CORRECTED_MP_ROOT/E2_rational_only_300m" \
  --matrixpolicy-phase E2_rational_only_300m \
  --replacement-manifest "$RLB_CONTROL_MANIFEST" \
  --replacement-run-root experiments/runs/iclr26_main/E2_global_rational_optimizers_300m \
  --replacement-phase E2_global_rational_optimizers_300m
```

Clean runtime tables for completed paper cells:

```bash
python3 experiments/scripts/summarize_iclr26_runtimes.py \
  --safe-e1-matrixpolicy-manifest "$CORRECTED_MP_MANIFEST" \
  --safe-e2-matrixpolicy-manifest "$CORRECTED_MP_MANIFEST" \
  --matrixpolicy-run-root "$CORRECTED_MP_ROOT" \
  --global-rational-optimizer-manifest "$RLB_CONTROL_MANIFEST"
```

Synchronize the generated packages into the active Markdown mirrors:

```bash
python3 experiments/scripts/sync_iclr26_result_readmes.py
```
