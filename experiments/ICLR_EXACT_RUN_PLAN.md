# Exact ICLR Experiment Plan

This plan is the forward experiment contract for a publishable MatrixPolicy optimizer paper. Current paper-facing evidence starts from the completed E1 fixed-config, manifest-declared, matched language-model pretraining suite; WikiText remains only a small demo anchor.

## Principle

The main claim must be tested before any ablation or sensitivity map:

```text
At equal model, data, token budget, seed, batch, validation slice, and evaluation cadence, the original MatrixPolicy optimizer recipe improves loss-vs-compute relative to strong fixed optimizer baselines.
```

The main suite is not a tuning stage. It is a fixed comparison copied from the experimental style of accepted optimizer papers: loss vs tokens, loss vs wall-clock/GPU-hours, model scale, paired sensitivity, and only then mechanism ablation appendices.

## Non-Negotiable Gates

```text
Gate 0: No GPU launch without a manifest row.
Gate 1: No reported comparison unless every matched cell row is complete.
Gate 2: AdamW and MatrixPolicy must share the exact same outer lr/min_lr/weight_decay set inside each matched cell.
Gate 3: No baseline grid before the MatrixPolicy main comparison.
Gate 4: No MatrixPolicy component ablation before main M0 curves exist.
Gate 5: No sensitivity landscape before main M0 curves exist.
Gate 6: Eval interval must be <= 50 for paper/protocol curves.
Gate 7: One Slurm job uses <= 4 A6000; total active use <= 8 A6000.
Gate 8: Raw outputs and caches remain ignored; tracked summaries are generated only from complete matched cells.
```


## Matched-Config Rule

A reportable comparison cell is identified by:

```text
phase
dataset
model
train-token budget
seed
validation slice
sequence length
global tokens per step
eval interval
```

Inside each reportable cell:

```text
AdamW outer configs = {(lr, min_lr, weight_decay) used by silu_adamw and rlb_adamw}
MatrixPolicy outer configs = {(lr, min_lr, weight_decay) used by rlb_matrixpolicy_original}
```

The required relation is equality:

```text
AdamW outer configs == MatrixPolicy outer configs
```

If AdamW uses `lr=5e-4, min_lr=5e-5, weight_decay=0.20` in a cell, MatrixPolicy must also use `lr=5e-4, min_lr=5e-5, weight_decay=0.20` in that same cell. If MatrixPolicy uses an extra outer config, AdamW must also use it. Otherwise the cell is incomplete and cannot appear as evidence.

The same parity rule applies later to sensitivity maps. A sensitivity map is a paired map, not an AdamW-only landscape and not a MatrixPolicy-only landscape.

## Dataset Suite

| task key | dataset | role |
| --- | --- | --- |
| `dclm` | `mlfoundations/dclm-baseline-1.0` | modern curated pretraining corpus |
| `fineweb_edu` | `HuggingFaceFW/fineweb-edu`, `sample-10BT` | modern educational web |
| `fineweb` | `HuggingFaceFW/fineweb`, `sample-10BT` | modern broad web |
| `dolma_sample` | `allenai/dolma`, `v1_6-sample` | diverse open corpus |
| `c4_en` | `allenai/c4`, `en` | comparability anchor |

FineWeb and FineWeb-Edu remain in the suite because they are modern corpora.

## Model Suite

| model | layers | d_model | heads | ffn_dim | seq_len | global tokens/step | role |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| M0 | 12 | 768 | 12 | 2048 | 256 | 32768 | main 123M-scale benchmark |
| M1 | 18 | 1024 | 16 | 3072 | 256 | 32768 | required scale check |
| M2 | 24 | 1280 | 16 | 4096 | 256 | 32768 | stretch smoke if M1 succeeds |

## Fixed Main Methods

The main manifest uses fixed method recipes. These are not LR/WD landscapes.

| method | activation | optimizer | LR | min LR | WD | extra recipe |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `silu_adamw` | `silu` | `adamw` | 3e-4 | 3e-5 | 0.10 | baseline |
| `rlb_adamw` | `rlb_fused_fixed_strong_ffn` | `adamw` | 3e-4 | 3e-5 | 0.10 | architecture control |
| `silu_muon` | `silu` | `muon` | 3e-4 | 3e-5 | 0.10 | `momentum=0.95`, `ns_steps=5` |
| `rlb_muon` | `rlb_fused_fixed_strong_ffn` | `muon` | 3e-4 | 3e-5 | 0.10 | architecture control |
| `silu_lion` | `silu` | `lion` | 1e-4 | 1e-5 | 0.10 | sign optimizer baseline |
| `rlb_lion` | `rlb_fused_fixed_strong_ffn` | `lion` | 1e-4 | 1e-5 | 0.10 | architecture control |
| `silu_soap` | `silu` | `soap_adamw` | 3e-4 | 3e-5 | 0.10 | precondition frequency 50 |
| `rlb_soap` | `rlb_fused_fixed_strong_ffn` | `soap_adamw` | 3e-4 | 3e-5 | 0.10 | architecture control |
| `silu_ademamix` | `silu` | `ademamix` | 3e-4 | 3e-5 | 0.10 | alpha 5, beta3 0.9999 |
| `rlb_ademamix` | `rlb_fused_fixed_strong_ffn` | `ademamix` | 3e-4 | 3e-5 | 0.10 | architecture control |
| `silu_came` | `silu` | `adafactor_came` | 3e-4 | 3e-5 | 0.10 | confidence scale 1 |
| `rlb_came` | `rlb_fused_fixed_strong_ffn` | `adafactor_came` | 3e-4 | 3e-5 | 0.10 | architecture control |
| `silu_schedulefree` | `silu` | `schedule_free_adamw` | 3e-4 | 3e-5 | 0.10 | beta1 0.9 |
| `rlb_schedulefree` | `rlb_fused_fixed_strong_ffn` | `schedule_free_adamw` | 3e-4 | 3e-5 | 0.10 | architecture control |
| `rlb_matrixpolicy_original` | `rlb_fused_fixed_strong_ffn` | `rational_matrix_policy_onpolicy` | 3e-4 | 3e-5 | 0.10 | original AdamW-backed group-stat MatrixPolicy recipe |

The MatrixPolicy row is a fixed method row with the same outer LR/min-LR/WD as the AdamW fixed rows. It is not allowed to be compared against an AdamW grid unless the MatrixPolicy rows contain the same outer grid in the same cells and the section is explicitly a later paired sensitivity appendix.

Current result-package note: the table above records the original main-manifest recipes. Paper-facing MatrixPolicy summaries overlay the 30 validated live-statistic-corrected `rlb_fused_global_rational` rows under `experiments/corrections/matrixpolicy_live_stats_20260712/`; non-MatrixPolicy RLB controls overlay `experiments/manifests/iclr26_global_rational_optimizer_controls_manifest.csv`. SiLU controls remain the completed main-manifest rows, and RLB+ADeMaMix is retained as a divergent/early-stop negative row.

## Main Experiment Units

### E0: Manifest And Loader Preflight

Purpose: prevent broken loaders and broken flags. This is not evidence and cannot choose settings.

```text
datasets: dclm, fineweb_edu, fineweb, dolma_sample, c4_en
methods: silu_adamw, rlb_adamw, rlb_matrixpolicy_original
model: M0
steps: 80
seed: 1337
eval_interval: 40
```

### E1: M0 100M Fixed Main Suite

Status: complete for all five datasets, three seeds, and 15 fixed methods. Full tables and curves are in `ICLR_RUN_STATUS.md`.

```text
datasets: dclm, fineweb_edu, fineweb, dolma_sample, c4_en
methods: all fixed main methods
seeds: 1337, 2027, 3407
train tokens: 100M
validation tokens: 4M
steps: 3050
eval interval: 50
```

Required deliverables:

```text
validation loss curves
training loss curves
full-run validation AUC
early/mid/late AUC: 25%, 50%, 75%, 100%
loss vs tokens
loss vs wall-clock and GPU-hours
tokens-to-target and GPU-hours-to-target
per-dataset and average rank
paired seed gaps
failure-adjusted rank
```

### E2: M0 300M Fixed Main Suite

Same rows as E1.

```text
train tokens: 300M
validation tokens: 8M
steps: 9150
eval interval: 50
```

This is the main ICLR table/curve source. E1 is a shorter-budget ranking-flip and compute-efficiency source; E2 is the stronger final-budget evidence.

### E3: M1 Scale Check

```text
datasets: dclm, fineweb_edu, c4_en
methods: silu_adamw, rlb_adamw, silu_soap, rlb_soap, rlb_matrixpolicy_original
seeds: 1337, 2027, 3407
train tokens: 300M
validation tokens: 8M
model: M1
eval interval: 50
```

The old checkpoint-dependent transfer diagnostics are removed from the local plan. They do not directly test the optimizer/activation claim, and the completed E1/E2 suite already tests the fixed method across five datasets and three seeds.

### E8: Paired LR/WD Sensitivity Maps, After Main Evidence Only

This is not used to choose the main result. The fixed E1/E2 setting stays frozen; E8 answers the reviewer-facing hyperparameter question: does MatrixPolicy remain competitive when every compared method receives the same outer learning-rate and weight-decay grid?

Use the current RLB source activation, `rlb_fused_global_rational`, for all RLB rows. SiLU rows use `silu`.

Primary compact map:

```text
model: M0
train tokens: 100M
validation tokens: 4M
eval interval: 50
datasets: dclm, fineweb_edu, c4_en
seed: 1337
methods:
  silu_adamw
  silu_muon
  rlb_matrixpolicy_original
LR grid: 1e-4, 2e-4, 3e-4, 5e-4
min LR: 0.1 * LR in each cell
WD grid: 0.00, 0.05, 0.10, 0.20
rows: 3 datasets * 1 seed * 3 methods * 16 grid cells = 144 runs
```

The initial grid prioritizes SiLU+AdamW, SiLU+Muon, and RLB+MatrixPolicy because they give the fixed AdamW baseline, the strongest familiar non-AdamW optimizer baseline, and the proposed method under the same outer LR/WD cells. RLB+AdamW is not part of the primary queue. SOAP, Lion, and CAME can be added later as narrow appendix extensions if the primary map leaves a clear reviewer-facing gap; AdEMAMix is not grid-expanded because the fixed-suite rows are already unstable/divergent.

Report:

```text
final validation loss heatmaps by dataset and method
target-arrival token heatmaps at shared target losses
wall-clock minutes to shared target losses
failure/non-finite cells
paired rank over the LR/WD grid
best-cell and fixed-cell comparison, with the fixed E1/E2 cell highlighted
```

Stopping rule:

```text
do not launch extra seeds unless a primary conclusion depends on one isolated grid cell
do not promote a new LR/WD cell into the main result
do not compare a tuned MatrixPolicy cell against an untuned AdamW/Muon/SOAP cell
```

Optional 300M confirmation, only after the 100M map is complete:

```text
datasets: dclm, fineweb_edu
seed: 1337
methods: silu_adamw, rlb_adamw, rlb_soap, rlb_matrixpolicy_original
grid cells:
  fixed cell: LR 3e-4, WD 0.10
  low LR: LR 1e-4, WD 0.10
  high LR: LR 5e-4, WD 0.10
  no WD: LR 3e-4, WD 0.00
  high WD: LR 3e-4, WD 0.20
rows: 2 datasets * 1 seed * 4 methods * 5 grid cells = 40 runs
```

### E9: Method Ablations, Last

Run only after the main evidence and E8 diagnostics are summarized. E9 uses the
frozen paper-facing method and cannot select a new setting. The complete
pre-registered design, intervention flags, estimands, timing contract, and
compute envelope are in `experiments/E9_100M_ABLATION_PLAN.md`.

```text
model: M0, 123.55M parameters
budget: 3,050 steps, exactly 99,942,400 training tokens
datasets: dclm, fineweb_edu, fineweb, dolma_sample, c4_en
fresh paired seeds: 2479, 5052, 8913
arms:
  A0 SiLU + AdamW
  A1 RLB + AdamW
  A2 RLB + static MatrixPolicy optimizer-recipe shell
  A3 full RLB + MatrixPolicy
  A4 without group-stat gradient gating
  A5 without role/depth factors
  A6 with the transient Muon branch suppressed at a fixed schedule
  A7 without role/depth factors and with the Muon branch suppressed at a fixed schedule
  A8 without reciprocal pair rescaling
  A9 without the role/depth-Muon action block
rows: 10 arms * 5 datasets * 3 seeds = 150 fresh runs
```

The full five-dataset matrix is fixed before launch. Historical `no_role_depth`,
`bypass_muon`, and V2-V6 rows remain exploratory and are not substituted into
E9. E9 is a 100M-token ablation study; no outcome-dependent 300M promotion is
part of this design.

## Experiment Code Map

The exact experiment code path is:

```text
experiments/scripts/build_iclr26_main_manifest.py
  -> experiments/manifests/iclr26_main_manifest.csv
  -> experiments/ICLR_RUN_COMMANDS.md
  -> experiments/scripts/run_iclr26_manifest_job.sh
  -> training/run_lm_optimizer_sweep.sbatch
  -> training/transformer_lm_compare.py
```

The optimizer and activation implementation files used by those rows are:

```text
optimizer_design/matrix_policy_optimizer.py
optimizer_design/transport_onpolicy_optimizer.py
optimizer_design/function_space_rational_optimizer.py
optimizer_design/baseline_optimizers.py
activation/rational_opt/rational.py
activation/csrc/rational_ext.cpp
activation/csrc/rational_cuda_kernel.cu
```

Tracked E1 figures and checkpoint tables are regenerated from completed JSONL files by:

```text
experiments/scripts/plot_iclr26_e1_curves.py
```

## Manifest Workflow

Generate the manifest:

```bash
python3 experiments/scripts/build_iclr26_main_manifest.py \
  --output experiments/manifests/iclr26_main_manifest.csv \
  --print-summary
```

Launch bounded chunks:

```bash
CONFIRM_ICLR26_MANIFEST=1 \
MANIFEST=experiments/manifests/iclr26_main_manifest.csv \
ROW_START=0 \
ROW_LIMIT=1 \
sbatch experiments/scripts/run_iclr26_manifest_job.sh
```

No job should be submitted if `squeue -u mt872` already shows two 4-GPU jobs active.

## Reporting Standard

Every tracked summary must include:

```text
manifest row IDs
complete/incomplete status by matched cell
dense validation and training curves
mean +/- std curves where seeds are aggregated
AUC and target-reaching metrics
loss-vs-GPU-hour curves when timing exists
nonfinite/divergence markers
hardware, wall-clock, and optimizer-state accounting
```

Final validation loss is one table column, not the whole result.
