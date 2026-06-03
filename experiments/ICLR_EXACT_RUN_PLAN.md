# Exact 2026 ICLR Experiment Plan

This is the concrete run plan for turning MatrixPolicy into a 2026-level optimizer paper. FineWeb and FineWeb-Edu remain in the plan. They are not copied from the accepted optimizer papers, but they are modern 2024 web pretraining corpora and should be used as a modern stress test. Accepted optimizer papers supply the experimental templates and anchor datasets: Sophia used OpenWebText and the Pile, SOAP used C4, and Fantastic Pretraining Optimizers evaluates C4-EN while training on a modern OLMo-like mixture with DCLM.

Therefore the paper uses two dataset tiers:

```text
accepted-paper anchors: C4-EN, OpenWebText, Pile validation where feasible
modern 2026-grade corpora: FineWeb-Edu, FineWeb, DCLM, Dolma sample
```

The existing FineWeb/FineWeb-Edu 100M-token 3-seed result stays as pilot evidence. The experiments below are the new paper evidence.

## Fixed Methods

Run these methods unless a row is explicitly marked as reduced:

```text
silu_adamw:        SiLU + AdamW
rlb_adamw:         RLB + AdamW
silu_muon:         SiLU + Muon
rlb_muon:          RLB + Muon
silu_soap:         SiLU + SOAP-style AdamW
rlb_soap:          RLB + SOAP-style AdamW
silu_lion:         SiLU + Lion
rlb_lion:          RLB + Lion
silu_ademamix:     SiLU + AdEMAMix
rlb_ademamix:      RLB + AdEMAMix
silu_came:         SiLU + CAME-style
rlb_came:          RLB + CAME-style
rlb_matrixpolicy:  RLB + original MatrixPolicy group-stat
```

Minimum main-table row set if runtime forces staged execution:

```text
silu_adamw, rlb_adamw, rlb_muon, rlb_soap, rlb_matrixpolicy
```

No MatrixPolicy v2 result belongs in the original MatrixPolicy claim.

## Fixed Models

| model | layers | d_model | heads | ffn_dim | seq_len | batch plan | role |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| M0 | 12 | 768 | 12 | 2048 | 256 | batch 16/GPU, grad_accum 2 | main 123M setting |
| M1 | 18 | 1024 | 16 | 3072 | 256 | batch 8/GPU, grad_accum 4 | required scale setting |
| M2 | 24 | 1280 | 16 | 4096 | 256 | memory-smoke first | stretch setting |

All jobs use 4 A6000 GPUs. At most two jobs may be active.

## Phase 0: Loader And Model Smoke Tests

Purpose: make accepted-paper anchors and modern corpora runnable before expensive runs.

Run these first. Each job is small and uses the minimum row set.

### Job 0A: accepted anchors C4-EN and OpenWebText

```bash
REAL_LM_TASKS="c4_en openwebtext" \
SEEDS="1337" \
RUN_SUFFIX="iclr26_smoke_anchor_m0" \
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
sbatch experiments/scripts/run_real_lm_screen_20260530.sh
```

### Job 0B: modern corpora DCLM and FineWeb-Edu

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
sbatch experiments/scripts/run_real_lm_screen_20260530.sh
```

### Job 0C: M1 memory smoke on FineWeb-Edu

```bash
REAL_LM_TASKS="fineweb_edu" \
SEEDS="1337" \
RUN_SUFFIX="iclr26_smoke_m1" \
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

Pass condition: every task writes complete JSONL for `silu_adamw`, `rlb_adamw`, `rlb_soap`, and `rlb_matrixpolicy`; M1 does not OOM.

## Phase 1: Accepted-Anchor Reproduction Runs

Purpose: make the paper comparable to accepted optimizer papers.

Datasets:

```text
c4_en          # SOAP and Fantastic-style C4/C4-EN anchor
openwebtext    # Sophia-style GPT-2 anchor
```

Run matrix:

| model | dataset | token budgets | seeds | row set |
| --- | --- | --- | --- | --- |
| M0 | c4_en | 100M, 300M | 1337, 2027, 3407 | full methods |
| M0 | openwebtext | 100M, 300M | 1337, 2027, 3407 | full methods |
| M1 | c4_en | 300M | 1337, 2027, 3407 | minimum row set |

Concrete launch template for one M0 100M C4-EN seed:

```bash
REAL_LM_TASKS="c4_en" \
SEEDS="1337" \
RUN_SUFFIX="iclr26_anchor_c4_m0_100m_seed1337" \
OUTPUT_ROOT="experiments/runs/iclr26_anchor" \
TOKEN_CACHE_DIR="experiments/cache/tokens_iclr26_anchor" \
MAX_TRAIN_TOKENS=100000000 \
MAX_VAL_TOKENS=4000000 \
STEPS=3050 \
EVAL_INTERVAL=50 \
EVAL_BATCHES=10 \
LOG_INTERVAL=10 \
INCLUDE_MUON=1 \
EXTRA_OPTIMIZERS="soap_adamw lion ademamix schedule_free_adamw adafactor_came" \
sbatch experiments/scripts/run_real_lm_screen_20260530.sh
```

For 300M, change:

```text
MAX_TRAIN_TOKENS=300000000
MAX_VAL_TOKENS=8000000
STEPS=9150
RUN_SUFFIX includes 300m
```

For OpenWebText, change `REAL_LM_TASKS="openwebtext"`. For seed changes, change `SEEDS` and `RUN_SUFFIX`.

Deliverables:

```text
accepted-anchor loss-vs-token curves
accepted-anchor loss-vs-GPU-hour curves
tokens-to-target table using tuned/selected AdamW target loss
optimizer overhead table
ranking-flip table from 100M to 300M
```

## Phase 2: FineWeb Modern Web Integration

Purpose: keep FineWeb/FineWeb-Edu and make them a strength rather than an unsupported accepted-paper claim.

Datasets:

```text
fineweb_edu
fineweb
```

Run matrix:

| model | dataset | token budgets | seeds | row set |
| --- | --- | --- | --- | --- |
| M0 | fineweb_edu | 100M, 300M, 600M | 1337, 2027, 3407 | full methods |
| M0 | fineweb | 100M, 300M, 600M | 1337, 2027, 3407 | full methods |
| M1 | fineweb_edu | 300M | 1337, 2027, 3407 | minimum row set |

Concrete launch template for one FineWeb-Edu 300M seed:

```bash
REAL_LM_TASKS="fineweb_edu" \
SEEDS="1337" \
RUN_SUFFIX="iclr26_modern_fwedu_m0_300m_seed1337" \
OUTPUT_ROOT="experiments/runs/iclr26_modern_fineweb" \
TOKEN_CACHE_DIR="experiments/cache/tokens_iclr26_modern_fineweb" \
MAX_TRAIN_TOKENS=300000000 \
MAX_VAL_TOKENS=8000000 \
FINEWEB_EDU_VAL_SKIP_TOKENS=610000000 \
STEPS=9150 \
EVAL_INTERVAL=50 \
EVAL_BATCHES=10 \
LOG_INTERVAL=10 \
INCLUDE_MUON=1 \
EXTRA_OPTIMIZERS="soap_adamw lion ademamix schedule_free_adamw adafactor_came" \
sbatch experiments/scripts/run_real_lm_screen_20260530.sh
```

For FineWeb, use `REAL_LM_TASKS="fineweb"` and `FINEWEB_VAL_SKIP_TOKENS=610000000`.

Deliverables:

```text
modern web speed-to-target table
FineWeb/FineWeb-Edu final-budget mean +/- std curves
comparison of accepted anchors vs modern web results
statement: FineWeb is not copied from accepted optimizer papers; it is the modern web generalization test
```

## Phase 3: DCLM 2026-Grade Pretraining Corpus

Purpose: align with modern DCLM/DataComp and Fantastic-style data mixtures.

Run matrix:

| model | dataset | token budgets | seeds | row set |
| --- | --- | --- | --- | --- |
| M0 | dclm | 100M, 300M, 600M | 1337, 2027, 3407 | full methods |
| M1 | dclm | 300M | 1337, 2027, 3407 | minimum row set |

Concrete launch template for one DCLM 300M seed:

```bash
REAL_LM_TASKS="dclm" \
SEEDS="1337" \
RUN_SUFFIX="iclr26_dclm_m0_300m_seed1337" \
OUTPUT_ROOT="experiments/runs/iclr26_dclm" \
TOKEN_CACHE_DIR="experiments/cache/tokens_iclr26_dclm" \
MAX_TRAIN_TOKENS=300000000 \
MAX_VAL_TOKENS=8000000 \
DCLM_VAL_SKIP_TOKENS=610000000 \
STEPS=9150 \
EVAL_INTERVAL=50 \
EVAL_BATCHES=10 \
LOG_INTERVAL=10 \
INCLUDE_MUON=1 \
EXTRA_OPTIMIZERS="soap_adamw lion ademamix schedule_free_adamw adafactor_came" \
sbatch experiments/scripts/run_real_lm_screen_20260530.sh
```

Deliverables:

```text
DCLM loss-vs-compute frontier
MatrixPolicy gap on a modern DataComp-style corpus
comparison with FineWeb and C4/OpenWebText anchors
```

## Phase 4: Scale And Data-Ratio Study

Purpose: copy Fantastic Pretraining Optimizers' scale/data-ratio lesson at academic scale.

Run matrix:

| model | datasets | token budgets | seeds | rows |
| --- | --- | --- | --- | --- |
| M0 | c4_en, fineweb_edu, dclm | 100M, 300M, 600M | 1337, 2027, 3407 | adamw, rlb_adamw, rlb_soap, rlb_matrixpolicy |
| M1 | c4_en, fineweb_edu, dclm | 300M | 1337, 2027, 3407 | adamw, rlb_adamw, rlb_soap, rlb_matrixpolicy |
| M2 | fineweb_edu | 100M | 1337 | adamw, rlb_matrixpolicy |

M1 launch changes from the M0 templates:

```text
BATCH_SIZE=8
GRAD_ACCUM=4
COMMON_EXTRA_ARGS="--layers 18 --d-model 1024 --heads 16 --ffn-dim 3072"
```

M2 is launched only after smoke success:

```text
BATCH_SIZE=4
GRAD_ACCUM=8
COMMON_EXTRA_ARGS="--layers 24 --d-model 1280 --heads 16 --ffn-dim 4096"
```

Deliverables:

```text
speedup vs model size
speedup vs token budget
rank changes over training horizon
compute multiplier needed by AdamW/other baselines to match MatrixPolicy loss
```

## Phase 5: Cross-Corpus Transfer

Purpose: show the result is not one-corpus overfitting.

Use checkpoints from selected 300M runs. Evaluate each checkpoint on validation caches from:

```text
c4_en
openwebtext
fineweb_edu
fineweb
dclm
dolma_sample
pile, if loader smoke passed
```

Train/eval grid:

| train corpus | eval corpora | model | seeds | rows |
| --- | --- | --- | --- | --- |
| c4_en | openwebtext, fineweb_edu, dclm | M0 | 1337, 2027, 3407 | adamw, rlb_soap, rlb_matrixpolicy |
| fineweb_edu | c4_en, openwebtext, dclm, dolma_sample | M0 | 1337, 2027, 3407 | adamw, rlb_soap, rlb_matrixpolicy |
| dclm | c4_en, fineweb_edu, dolma_sample | M0 | 1337, 2027, 3407 | adamw, rlb_soap, rlb_matrixpolicy |

Needed code: add an evaluation-only script that loads a checkpoint and multiple token caches. Do not run more pretraining just to get transfer numbers if the checkpoints already exist.

Deliverables:

```text
in-domain vs out-of-domain validation scatter
transfer degradation table
MatrixPolicy gap retained/lost across corpora
```

## Phase 6: Memory, Throughput, And Batch Regime

Purpose: copy SOAP, Adam-mini, GaLore, and CAME efficiency accounting.

Run profiling jobs:

| model | dataset | steps | global tokens/step | rows |
| --- | --- | ---: | ---: | --- |
| M0 | c4_en | 1000 | 16k, 32k, 65k | adamw, rlb_soap, rlb_matrixpolicy |
| M0 | fineweb_edu | 1000 | 16k, 32k, 65k | adamw, rlb_soap, rlb_matrixpolicy |
| M1 | fineweb_edu | 1000 | 32k | adamw, rlb_soap, rlb_matrixpolicy |

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

## Phase 7: Long-Horizon Corpus Shift

Purpose: copy AdEMAMix-style long-horizon/forgetting evaluation.

Run continued training:

```text
FineWeb-Edu 300M -> DCLM 300M
C4-EN 300M -> FineWeb-Edu 300M
```

Rows:

```text
adamw
rlb_soap
rlb_ademamix
rlb_matrixpolicy
```

Seeds:

```text
1337, 2027, 3407
```

Evaluate at switch and final on:

```text
source corpus validation
target corpus validation
Dolma validation
```

Deliverables:

```text
new-domain learning curve
old-domain forgetting curve
MatrixPolicy loss-vs-GPU-hour after corpus shift
```

## Phase 8: Reviewer-Defense Sensitivity Maps

Purpose: prevent weak-baseline criticism. This is a support experiment, not the main evidence.

Run 50M-token maps on:

```text
c4_en
fineweb_edu
dclm
```

Rows and grids:

```text
AdamW: LR {1e-4, 3e-4, 5e-4}, WD {0.03, 0.10, 0.20}
Muon: LR {1e-4, 3e-4, 5e-4}, WD {0.03, 0.10, 0.20}, momentum {0.90, 0.95}
Lion: LR {3e-5, 1e-4, 2e-4}, WD {0.10, 0.30, 0.60}
SOAP-style: LR {1e-4, 3e-4, 5e-4}, WD {0.03, 0.10}, frequency {10, 50}
AdEMAMix: LR {1e-4, 2e-4, 3e-4}, WD {0.03, 0.10}, alpha {2, 5}, beta3 {0.999, 0.9999}
MatrixPolicy: LR {2e-4, 3e-4, 5e-4}, WD {0.03, 0.10}, adam-scale {2, 3, 4}, group-gain {0.20, 0.35}
```

If this finds a stronger baseline, rerun the affected main rows. Do not change the dataset, token budget, model size, or seed plan.

## Immediate Queue

Run in this order:

```text
1. Job 0A smoke: c4_en + openwebtext.
2. Job 0B smoke: dclm + fineweb_edu.
3. Job 0C M1 memory smoke on fineweb_edu.
4. Phase 1 C4-EN M0 100M seed 1337 and Phase 2 FineWeb-Edu M0 100M seed 1337, two jobs active.
5. Phase 3 DCLM M0 100M seed 1337 and Phase 1 OpenWebText M0 100M seed 1337, two jobs active.
6. Repeat 100M for seeds 2027 and 3407.
7. Move to 300M on c4_en, fineweb_edu, dclm, openwebtext.
8. Start M1 300M only after M1 smoke passes.
9. Run 600M only after 300M curves show MatrixPolicy is still competitive in loss per GPU-hour.
10. Run sensitivity maps and ablations after the main curves exist.
```

This plan integrates FineWeb correctly: FineWeb/FineWeb-Edu remain central modern web stress tests, while C4/OpenWebText/Pile/DCLM connect the paper to accepted optimizer-paper evaluation practice.
