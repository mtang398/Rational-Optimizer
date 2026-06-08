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

Rows `240-464` are E2. E2 is submitted in whole 15-row matched cells. The first running window starts two DCLM cells, using at most 8 requested A6000 if both run at once.

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
