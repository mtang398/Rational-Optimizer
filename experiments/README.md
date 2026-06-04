# Experiments

This directory keeps the existing FineWeb/FineWeb-Edu and WikiText results. New paper experiments use the manifest workflow below.

## Existing Results

```text
results/real_lm_multiseed_2026_05_31/
results/real_lm_screen_2026_05_30/
results/rlb_matrix_policy_muon_switch_2026_05_28/
runs/real_lm_multiseed_20260531/
runs/wikitext103/
```

New paper runs follow the manifest rule below.

## Forward Contract

Use `ICLR_EXACT_RUN_PLAN.md` and the generated manifest. Each matched cell must contain AdamW and MatrixPolicy rows with the same outer optimizer config:

```text
same dataset
same model
same train-token budget
same seed
same validation slice
same sequence length
same global tokens per step
same eval interval
same lr
same min_lr
same weight_decay
```

Do not launch one-sided AdamW grids. Do not launch MatrixPolicy sensitivity rows unless the corresponding AdamW/RLB control rows with the same outer config are in the same manifest cell. Sensitivity maps and method ablations come after main evidence.

## Commands

Generate the main manifest:

```bash
python3 experiments/scripts/build_iclr26_main_manifest.py \
  --output experiments/manifests/iclr26_main_manifest.csv \
  --print-summary
```

Run one manifest row:

```bash
CONFIRM_ICLR26_MANIFEST=1 \
MANIFEST=experiments/manifests/iclr26_main_manifest.csv \
ROW_START=0 \
ROW_LIMIT=1 \
sbatch experiments/scripts/run_iclr26_manifest_job.sh
```

Run a bounded shard:

```bash
CONFIRM_ICLR26_MANIFEST=1 \
MANIFEST=experiments/manifests/iclr26_main_manifest.csv \
ROW_START=40 \
ROW_LIMIT=4 \
sbatch experiments/scripts/run_iclr26_manifest_job.sh
```

Do not run more than two 4-GPU jobs at once.

## Output Policy

Raw new outputs belong under ignored `experiments/runs/`. Token caches belong under ignored `experiments/cache/`. New tracked summaries should be created only after matched cells complete and must include dense curves, AUC, timing, divergence markers, and exact manifest row IDs.
