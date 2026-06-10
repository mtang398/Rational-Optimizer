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
