# Experiments

This directory keeps the existing FineWeb/FineWeb-Edu and WikiText results. New paper experiments use the manifest workflow below.

## Existing Results

```text
results/real_lm_multiseed_2026_05_31/
results/real_lm_screen_2026_05_30/
results/rlb_matrix_policy_muon_switch_2026_05_28/
runs/real_lm_multiseed_20260531/
```

New paper runs follow the manifest rule below.


## FineWeb And FineWeb-Edu Results

| task | MatrixPolicy mean | SiLU+AdamW mean | best non-MatrixPolicy mean | gap vs SiLU+AdamW | gap vs best control |
| --- | ---: | ---: | ---: | ---: | ---: |
| FineWeb | 4.369701 loss / 79.04 PPL | 4.528963 loss / 92.69 PPL | 4.522311 loss / 92.08 PPL | 0.159263 | 0.152302 |
| FineWeb-Edu | 4.069422 loss / 58.52 PPL | 4.223572 loss / 68.28 PPL | 4.223572 loss / 68.28 PPL | 0.154149 | 0.153402 |

Full CSVs and summaries live in `results/real_lm_multiseed_2026_05_31/`.

FineWeb curves:

![FineWeb mean validation loss](results/real_lm_multiseed_2026_05_31/fineweb_validation_loss_mean.png)

![FineWeb mean validation loss zoom](results/real_lm_multiseed_2026_05_31/fineweb_validation_loss_mean_zoom_step1000.png)

![FineWeb mean validation PPL](results/real_lm_multiseed_2026_05_31/fineweb_validation_ppl_mean.png)

![FineWeb training loss](results/real_lm_multiseed_2026_05_31/fineweb_training_loss_mean.png)

FineWeb-Edu curves:

![FineWeb-Edu mean validation loss](results/real_lm_multiseed_2026_05_31/fineweb_edu_validation_loss_mean.png)

![FineWeb-Edu mean validation loss zoom](results/real_lm_multiseed_2026_05_31/fineweb_edu_validation_loss_mean_zoom_step1000.png)

![FineWeb-Edu mean validation PPL](results/real_lm_multiseed_2026_05_31/fineweb_edu_validation_ppl_mean.png)

![FineWeb-Edu training loss](results/real_lm_multiseed_2026_05_31/fineweb_edu_training_loss_mean.png)

## WikiText Result

| method | final loss | final PPL |
| --- | ---: | ---: |
| RLB MatrixPolicy-Muon | 3.476232 | 32.34 |
| RLB Smooth-MatrixPolicy | 3.493210 | 32.89 |
| SiLU/SwiGLU+AdamW beta2=0.999 | 3.549346 | 34.79 |
| RLB+AdamW beta2=0.999 | 3.550018 | 34.81 |
| RLB+AdamW | 3.617501 | 37.24 |
| SiLU/SwiGLU+AdamW | 3.621982 | 37.41 |
| SiLU/SwiGLU+Muon | 3.644921 | 38.28 |
| RLB+Muon | 3.657877 | 38.78 |

![WikiText validation loss](results/rlb_matrix_policy_muon_switch_2026_05_28/same_lr_validation_loss.png)

![WikiText validation PPL](results/rlb_matrix_policy_muon_switch_2026_05_28/same_lr_validation_ppl.png)

![WikiText training loss](results/rlb_matrix_policy_muon_switch_2026_05_28/same_lr_training_loss_from_step1.png)

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

## Reproducibility Map

Current paper runs are reproduced from these files:

```text
experiments/scripts/build_iclr26_main_manifest.py
experiments/manifests/iclr26_main_manifest.csv
experiments/scripts/run_iclr26_manifest_job.sh
training/run_wikitext103_optimizer_sweep.sbatch
training/transformer_wikitext103_compare.py
optimizer_design/matrix_policy_optimizer.py
optimizer_design/transport_onpolicy_optimizer.py
optimizer_design/baseline_optimizers.py
activation/rational_opt/rational.py
```

Regenerate paper figures and tables from raw JSONL with:

```bash
python3 experiments/scripts/plot_iclr26_e1_curves.py
python3 experiments/scripts/summarize_real_lm_multiseed.py \
  --run-root experiments/runs/real_lm_multiseed_20260531 \
  --result-dir experiments/results/real_lm_multiseed_2026_05_31
```

The curated WikiText anchor is tracked under `results/rlb_matrix_policy_muon_switch_2026_05_28/`; raw WikiText launcher output is local run data and is ignored.

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
