# Exact 2026 ICLR Experiment Plan

This is the concrete experiment plan for a publishable MatrixPolicy optimizer paper. It is not a minimal continuation of the current FineWeb pilot. It follows accepted optimizer-paper experiment types: LM pretraining loss-vs-compute, token-budget scaling, model scaling, batch/overhead behavior, memory/throughput accounting, cross-corpus transfer, long-horizon behavior, optimizer sensitivity maps, and late ablations.

The plan is complicated by design: the paper is won only if MatrixPolicy holds up across several different optimizer experiments, not because one existing setting is dressed up.

## Evidence Standard Being Copied

```text
Sophia / SOAP: loss vs tokens, steps, GPU-hours, and wall-clock to target loss.
Fantastic Pretraining Optimizers: final-budget comparisons across model scale and data/token ratios; report ranking flips.
AdamW: LR/WD landscapes so optimizer effects are not schedule/regularization artifacts.
Lion / Schedule-Free: transfer and horizon robustness.
Adam-mini / GaLore / CAME: memory, optimizer-state footprint, throughput, and overhead.
AdEMAMix: long-horizon behavior under longer token budgets and corpus shift.
Cautious / Fantastic Pretraining Optimizers: config transfer, ranking-flip checks, and final-budget comparisons across scales.
```

## Dataset Suite

Use a suite so the claim is not tied to the existing FineWeb pilot.

| role | task key | dataset | purpose |
| --- | --- | --- | --- |
| modern curated pretraining | `dclm` | `mlfoundations/dclm-baseline-1.0` | modern DataComp-LM corpus; strong 2026 relevance. |
| modern educational web | `fineweb_edu` | `HuggingFaceFW/fineweb-edu`, `sample-10BT` | modern web/educational corpus; connects to pilot without defining the paper. |
| modern broad web | `fineweb` | `HuggingFaceFW/fineweb`, `sample-10BT` | tests general web data outside educational filtering. |
| diverse open corpus | `dolma_sample` | `allenai/dolma`, `v1_6-sample` | open OLMo-style corpus for transfer/corpus-shift evidence. |
| comparability anchor | `c4_en` | `allenai/c4`, `en` | C4/C4-EN style anchor for optimizer-paper comparability. |

Optional loader checks only:

```text
openwebtext, pile
```

## Methods

```text
silu_adamw
rlb_adamw
silu_muon
rlb_muon
silu_soap
rlb_soap
silu_lion
rlb_lion
silu_ademamix
rlb_ademamix
silu_came
rlb_came
rlb_matrixpolicy_original_groupstat
```

Minimum expensive-run row set:

```text
silu_adamw
rlb_adamw
rlb_soap
rlb_matrixpolicy_original_groupstat
best non-MatrixPolicy generic from the main suite
```

## Models

| model | layers | d_model | heads | ffn_dim | seq_len | batch plan | role |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| M0 | 12 | 768 | 12 | 2048 | 256 | batch 16/GPU, grad_accum 2 | main 123M-scale benchmark |
| M1 | 18 | 1024 | 16 | 3072 | 256 | batch 8/GPU, grad_accum 4 | required scale confirmation |
| M2 | 24 | 1280 | 16 | 4096 | 256 | smoke first | stretch scale if feasible |

All GPU jobs use 4 A6000s. At most two jobs may be active.


## Cross-Phase Analysis Rules

Every experiment writes the same core fields so the paper can compare methods across phases:

```text
final validation loss
validation loss AUC over full run
validation loss AUC over first 25%, 50%, 75% of tokens
best validation loss reached
step/tokens/GPU-hours/wall-clock to each target threshold
optimizer-step seconds
forward/backward seconds
tokens/sec
peak CUDA allocated/reserved
estimated optimizer-state memory
clipping rate
grad norm before clipping
nonfinite/divergence status
```

Target thresholds are defined per dataset from the frozen `silu_adamw` 300M run:

```text
target_easy:  silu_adamw_300M_final_loss + 0.05
target_match: silu_adamw_300M_final_loss
target_hard:  silu_adamw_300M_final_loss - 0.05 if any method reaches it
```

Main paper statistics:

```text
paired seed gaps by dataset
bootstrap CI over seeds within each dataset
bootstrap CI over dataset-seed pairs for average rank
average rank across dclm, fineweb_edu, fineweb, dolma_sample, c4_en
loss-per-GPU-hour Pareto frontier
memory-throughput-loss Pareto frontier
failure-adjusted score where divergent runs are assigned worst rank
```

## Gate Rules

The plan is large, so expensive phases have gates. Gates do not change the scientific target; they prevent wasting cluster time on broken rows.

```text
Gate A after Phase 0: dataset/model smokes pass or failed loader is marked infrastructure-blocked.
Gate B after Phase 1: one frozen config per optimizer family is selected.
Gate C after Phase 2 100M: remove only rows that diverge on both dclm and fineweb_edu for two seeds.
Gate D after Phase 2 300M: start M1 and 600M only for rows needed to establish the frontier and strongest baselines.
Gate E after Phase 5: if MatrixPolicy overhead is too high, report both equal-token and equal-GPU-hour comparisons.
```

## Distinct Experiment Units

These are the experiments the paper should be able to show. Some share launches, but they answer different reviewer questions.

| id | experiment | concrete matrix | paper question |
| --- | --- | --- | --- |
| E0 | loader/model feasibility | Phase 0 smokes on `dclm`, `fineweb_edu`, `dolma_sample`, `c4_en`, plus M1 | can the benchmark suite run without hidden loader/model failures? |
| E1 | baseline protocol lock | Phase 1 50M grids on `dclm` and `fineweb_edu` | are baselines selected before the main comparison, rather than after seeing the headline setting? |
| E2 | optimizer-family factorial | Phase 2 full rows on 4 corpora | does MatrixPolicy beat modern optimizer families, not just AdamW/Muon? |
| E3 | architecture-control factorial | SiLU vs RLB variants for AdamW/Muon/Lion/SOAP/AdEMAMix/CAME | is the gain optimizer-specific rather than plain RLB? |
| E4 | main loss-vs-compute frontier | M0, 100M and 300M, 5 corpora, 3 seeds | does MatrixPolicy improve validation loss at equal tokens and equal GPU-hours? |
| E5 | token-budget ranking flips | 25/50/75/100 percent checkpoints plus 100M/300M comparison | does the ranking survive stopping-time changes? |
| E6 | long-horizon frontier | 600M on decisive rows over `dclm`, `fineweb_edu`, `fineweb` | do gains widen, shrink, or reverse with longer training? |
| E7 | model-scale transfer | M1 300M and M2 smoke | is the result not only a 123M-model artifact? |
| E8 | batch-size behavior | global tokens/step 16k, 32k, 65k | is the optimizer robust to batch regime? |
| E9 | throughput and optimizer-state cost | 1000-step profiling at M0/M1 | is the loss gain worth the overhead and memory footprint? |
| E10 | cross-corpus evaluation | train on one corpus, evaluate on held-out corpora | does the gain transfer out of the training distribution? |
| E11 | corpus-shift continued training | `dclm -> fineweb_edu`, `fineweb_edu -> dolma_sample` | does MatrixPolicy retain an advantage after data distribution shifts? |
| E12 | sensitivity maps | compact LR/WD/family maps on 3 corpora | is the result robust to reasonable optimizer settings? |
| E13 | mechanism telemetry | diagnostics harvested from main runs | do role/update statistics explain when the method works or fails? |
| E14 | method ablations | MatrixPolicy component removals after main evidence | which components matter once a real advantage exists? |

Approximate planned method-run units before gates:

```text
Phase 2 main M0 suite:        342 method-runs
Phase 3 600M frontier:         45 method-runs
Phase 4 M1/M2 scale:           47 method-runs
Phase 5 batch/memory:          21 profiling method-runs
Phase 7 corpus shift:          24 continued-training method-runs
Phase 10 ablations:            84 method-runs if all rows are promoted
Phase 8 sensitivity maps:     large compact-grid appendix; not used to choose datasets or token budgets
```

## Phase 0: Loader And Model Smokes

Purpose: make the suite runnable before expensive runs.

### 0A: modern corpus smoke

```bash
REAL_LM_TASKS="dclm fineweb_edu" \
SEEDS="1337" \
RUN_SUFFIX="iclr26_smoke_modern_m0" \
OUTPUT_ROOT="experiments/runs/iclr26_smoke" \
TOKEN_CACHE_DIR="experiments/cache/tokens_iclr26_smoke" \
MAX_TRAIN_TOKENS=2000000 \
MAX_VAL_TOKENS=200000 \
STEPS=80 \
EVAL_INTERVAL=40 \
EVAL_BATCHES=2 \
LOG_INTERVAL=10 \
INCLUDE_MUON=1 \
EXTRA_OPTIMIZERS="soap_adamw" \
VAL_SKIP_TOKENS=10000 \
DEFAULT_VAL_SKIP_TOKENS=10000 \
sbatch experiments/scripts/run_real_lm_screen_20260530.sh
```

### 0B: transfer/comparability smoke

```bash
REAL_LM_TASKS="dolma_sample c4_en" \
SEEDS="1337" \
RUN_SUFFIX="iclr26_smoke_transfer_anchor_m0" \
OUTPUT_ROOT="experiments/runs/iclr26_smoke" \
TOKEN_CACHE_DIR="experiments/cache/tokens_iclr26_smoke" \
MAX_TRAIN_TOKENS=2000000 \
MAX_VAL_TOKENS=200000 \
STEPS=80 \
EVAL_INTERVAL=40 \
EVAL_BATCHES=2 \
LOG_INTERVAL=10 \
INCLUDE_MUON=1 \
EXTRA_OPTIMIZERS="soap_adamw" \
VAL_SKIP_TOKENS=10000 \
DEFAULT_VAL_SKIP_TOKENS=10000 \
sbatch experiments/scripts/run_real_lm_screen_20260530.sh
```

### 0C: M1 memory smoke

```bash
REAL_LM_TASKS="dclm" \
SEEDS="1337" \
RUN_SUFFIX="iclr26_smoke_m1_dclm" \
OUTPUT_ROOT="experiments/runs/iclr26_smoke" \
TOKEN_CACHE_DIR="experiments/cache/tokens_iclr26_smoke" \
MAX_TRAIN_TOKENS=4000000 \
MAX_VAL_TOKENS=200000 \
STEPS=120 \
EVAL_INTERVAL=60 \
EVAL_BATCHES=2 \
LOG_INTERVAL=10 \
BATCH_SIZE=8 \
GRAD_ACCUM=4 \
INCLUDE_MUON=0 \
COMMON_EXTRA_ARGS="--layers 18 --d-model 1024 --heads 16 --ffn-dim 3072" \
sbatch experiments/scripts/run_real_lm_screen_20260530.sh
```

## Phase 1: Baseline Protocol Lock

Purpose: lock fair optimizer settings before final claims. This is not the main evidence and is not used to choose the dataset suite, token budgets, model sizes, or MatrixPolicy ablations.

Run 50M-token protocol-lock grids on:

```text
dclm
fineweb_edu
```

Grids:

```text
AdamW: LR {1e-4, 3e-4, 5e-4}, WD {0.03, 0.10, 0.20}
Muon: LR {1e-4, 3e-4, 5e-4}, WD {0.03, 0.10, 0.20}, momentum {0.90, 0.95}
Lion: LR {3e-5, 1e-4, 2e-4}, WD {0.10, 0.30, 0.60}
SOAP-style: LR {1e-4, 3e-4, 5e-4}, WD {0.03, 0.10}, frequency {10, 50}
AdEMAMix: LR {1e-4, 2e-4, 3e-4}, WD {0.03, 0.10}, alpha {2, 5}, beta3 {0.999, 0.9999}
CAME-style: LR {1e-4, 3e-4, 5e-4}, WD {0.03, 0.10, 0.20}, confidence {0.5, 1.0}
MatrixPolicy: LR {2e-4, 3e-4, 5e-4}, WD {0.03, 0.10}, adam-scale {2, 3, 4}, group-gain {0.20, 0.35}
```

Selection rule: freeze one config per optimizer family by mean rank over `dclm` and `fineweb_edu`, using final loss, AUC, divergence, then overhead. The frozen configs transfer to the main suite unless Phase 8 shows that a baseline was materially under-served.

## Phase 2: Main M0 Loss-Vs-Compute Suite

Purpose: main paper result.

| dataset | budgets | seeds | rows |
| --- | --- | --- | --- |
| dclm | 100M, 300M | 1337, 2027, 3407 | full methods |
| fineweb_edu | 100M, 300M | 1337, 2027, 3407 | full methods |
| fineweb | 100M, 300M | 1337, 2027, 3407 | full methods |
| dolma_sample | 100M, 300M | 1337, 2027, 3407 | full methods |
| c4_en | 100M, 300M | 1337, 2027, 3407 | minimum row set plus best generic |

Template for one DCLM 100M seed:

```bash
REAL_LM_TASKS="dclm" \
SEEDS="1337" \
RUN_SUFFIX="iclr26_main_dclm_m0_100m_seed1337" \
OUTPUT_ROOT="experiments/runs/iclr26_main_m0" \
TOKEN_CACHE_DIR="experiments/cache/tokens_iclr26_main_m0" \
MAX_TRAIN_TOKENS=100000000 \
MAX_VAL_TOKENS=4000000 \
DCLM_VAL_SKIP_TOKENS=210000000 \
STEPS=3050 \
EVAL_INTERVAL=50 \
EVAL_BATCHES=10 \
LOG_INTERVAL=10 \
INCLUDE_MUON=1 \
EXTRA_OPTIMIZERS="soap_adamw lion ademamix schedule_free_adamw adafactor_came" \
sbatch experiments/scripts/run_real_lm_screen_20260530.sh
```

For 300M:

```text
MAX_TRAIN_TOKENS=300000000
MAX_VAL_TOKENS=8000000
*_VAL_SKIP_TOKENS=610000000
STEPS=9150
```

Deliverables:

```text
mean +/- std validation curves
loss vs tokens
loss vs GPU-hours
final-budget average rank across datasets
tokens-to-target and GPU-hours-to-target
ranking flip table from 100M to 300M
nonfinite/divergence table
per-dataset winner table
per-seed paired gap table
frontier dominance table: lower loss at equal GPU-hours or fewer GPU-hours at equal loss
```

Main-suite figure set:

```text
Figure 1a: DCLM loss vs GPU-hours
Figure 1b: FineWeb-Edu loss vs GPU-hours
Figure 1c: FineWeb loss vs GPU-hours
Figure 1d: Dolma loss vs GPU-hours
Figure 2: average rank over datasets at 100M and 300M
Figure 3: tokens-to-target and GPU-hours-to-target for target_easy/target_match
Figure 4: ranking flips from 25%, 50%, 75%, 100% of training
```

## Phase 3: Long-Horizon Frontier

Purpose: accepted optimizer papers need horizon checks.

Run 600M-token extensions on decisive rows:

```text
datasets: dclm, fineweb_edu, fineweb
seeds: 1337, 2027, 3407
rows: silu_adamw, rlb_adamw, rlb_soap, rlb_matrixpolicy, best generic
```

Deliverables:

```text
loss-vs-GPU-hour frontier through 600M
whether MatrixPolicy gains widen, shrink, or reverse
long-horizon tokens-to-target table
```

## Phase 4: Model Scale

Purpose: show the effect is not 123M-only.

| model | datasets | budget | seeds | rows |
| --- | --- | ---: | --- | --- |
| M1 | dclm, fineweb_edu, c4_en | 300M | 1337, 2027, 3407 | minimum row set plus best generic |
| M2 | dclm or fineweb_edu | 100M | 1337 | silu_adamw, rlb_matrixpolicy |

M1 launch changes:

```text
BATCH_SIZE=8
GRAD_ACCUM=4
COMMON_EXTRA_ARGS="--layers 18 --d-model 1024 --heads 16 --ffn-dim 3072"
```

## Phase 5: Batch, Throughput, And Memory

Purpose: efficiency accounting.

| model | dataset | steps | global tokens/step | rows |
| --- | --- | ---: | ---: | --- |
| M0 | dclm | 1000 | 16k, 32k, 65k | silu_adamw, rlb_soap, rlb_matrixpolicy |
| M0 | fineweb_edu | 1000 | 16k, 32k, 65k | silu_adamw, rlb_soap, rlb_matrixpolicy |
| M1 | dclm | 1000 | 32k | silu_adamw, rlb_soap, rlb_matrixpolicy |

Deliverables:

```text
optimizer-step seconds
forward/backward seconds
tokens/sec
peak CUDA allocated/reserved
optimizer-state memory estimate
loss improvement per extra GPU-hour
batch-size sensitivity curves
```

## Phase 6: Cross-Corpus Transfer

Purpose: transfer and robustness.

Use 300M checkpoints. Add an evaluation-only script if needed.

| train corpus | eval corpora | model | seeds | rows |
| --- | --- | --- | --- | --- |
| dclm | fineweb_edu, fineweb, dolma_sample, c4_en | M0 | 1337, 2027, 3407 | silu_adamw, rlb_soap, rlb_matrixpolicy |
| fineweb_edu | dclm, fineweb, dolma_sample, c4_en | M0 | 1337, 2027, 3407 | silu_adamw, rlb_soap, rlb_matrixpolicy |
| dolma_sample | dclm, fineweb_edu, c4_en | M0 | 1337, 2027, 3407 | silu_adamw, rlb_soap, rlb_matrixpolicy |

Deliverables:

```text
in-domain vs out-of-domain validation scatter
transfer degradation table
MatrixPolicy gap retained/lost across corpora
```

## Phase 7: Corpus-Shift Continued Training

Purpose: long-horizon behavior under corpus shift.

```text
dclm 300M -> fineweb_edu 300M
fineweb_edu 300M -> dolma_sample 300M
rows: silu_adamw, rlb_soap, rlb_ademamix, rlb_matrixpolicy
seeds: 1337, 2027, 3407
```

Deliverables:

```text
new-domain learning curve
old-domain forgetting curve
loss-vs-GPU-hour after corpus shift
```

## Phase 8: Sensitivity Maps

Purpose: reviewer defense and optimizer robustness.

Run compact 50M maps on:

```text
dclm
fineweb_edu
c4_en
```

Rows:

```text
AdamW, Muon, Lion, SOAP-style, AdEMAMix, CAME-style, MatrixPolicy
```

Grid execution detail:

```text
seed 1337 for full grid
seed 2027 for top 3 configs per family
seed 3407 for final selected config per family if the 1337/2027 ranking disagrees
early-stop only for nonfinite or clearly failed rows after warmup
all stopped rows remain in the stability map
```

Deliverables:

```text
LR/WD heatmaps
stability maps
best-config appendix table
LR/WD sensitivity width for each family
config-transfer table from the dclm protocol lock to fineweb_edu and c4_en
```

## Phase 9: Mechanism Diagnostics From Main Runs

Use logs from the main runs.

Deliverables:

```text
role update/weight RMS
matrix spectrum/SVD entropy
rational output RMS
rational derivative RMS
denominator margin
gradient clipping frequency
grad norm before clipping
fixed-probe logit KL/RMS movement
optimizer-step overhead by role
```

## Phase 10: Method Ablations

Run after Phases 2-5 show a real advantage.

```text
dataset: dclm and fineweb_edu
budget: 100M, then 300M for promoted rows
seeds: 1337, 2027, 3407
base: original MatrixPolicy group-stat
ablations: no group-stat scaling, no role-depth policy, no matrix branch, no rebalance if relevant, gain-only, pressure-only, activity-only
```

## Paper Table And Figure Plan

Main paper:

```text
Table 1: main M0 final loss, PPL, GPU-hours, divergence across dataset suite
Table 2: tokens-to-target and GPU-hours-to-target by dataset
Table 3: M1 scale confirmation and M2 stretch smoke
Table 4: memory/throughput/optimizer-state footprint
Figure 1: loss-vs-GPU-hour frontier across DCLM/FineWeb-Edu/FineWeb/Dolma
Figure 2: average rank over token budget and model size
Figure 3: batch-size and overhead sensitivity
Figure 4: transfer matrix
Figure 5: long-horizon 600M and corpus-shift curves
```

Appendix:

```text
all LR/WD sensitivity heatmaps
all per-seed curves
all failed/diverged rows
all exact commands and configs
all telemetry diagnostics
all ablation curves
```

## Immediate Queue

```text
1. Phase 0A smoke: dclm + fineweb_edu.
2. Phase 0B smoke: dolma_sample + c4_en.
3. Phase 0C M1 smoke on dclm.
4. Phase 1 baseline protocol lock on dclm and fineweb_edu.
5. Phase 2 M0 100M: dclm seed 1337 and fineweb_edu seed 1337 as the first real pair.
6. Phase 2 M0 100M: fineweb seed 1337 and dolma_sample seed 1337 as the second pair.
7. Repeat Phase 2 100M for seeds 2027 and 3407.
8. Run c4_en 100M minimum row set as comparability anchor.
9. Move to 300M for dclm, fineweb_edu, fineweb, dolma_sample, c4_en.
10. Start M1 300M only after M1 smoke passes and 100M M0 results are summarized.
11. Start 600M only after 300M loss-per-GPU-hour curves justify it.
12. Run transfer, corpus-shift, diagnostics, sensitivity maps, and ablations after main curves exist.
```
