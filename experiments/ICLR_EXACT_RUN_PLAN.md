# Exact New Experiment Plan For An ICLR MatrixPolicy Paper

This is the experiment plan to make RationalOPT look like a serious accepted optimizer paper. It copies the experimental shapes used by accepted optimizer papers: Sophia, SOAP, AdamW, Lion, Adam-mini, GaLore, CAME, Schedule-Free, AdEMAMix, and Fantastic Pretraining Optimizers.

The main paper should be won by standard optimizer evidence: speed-to-target, final-budget loss, scaling, data-ratio robustness, transfer, memory/throughput, and broad baselines.

## Main Claim To Prove

```text
MatrixPolicy is a stronger optimizer for RLB Transformer training than generic AdamW/Muon and modern optimizer-family baselines, giving a better loss-vs-compute frontier at academic LM-pretraining scale while remaining stable, transferable, and efficient enough to justify its overhead.
```

The existing 100M-token FineWeb/FineWeb-Edu result is only a pilot. The paper needs the new experiments below.

## Accepted-Paper Templates To Copy

| accepted paper | what to copy exactly |
| --- | --- |
| Sophia, ICLR 2024 | GPT-style LM pretraining at multiple model sizes; report steps, tokens, compute, and wall-clock to the same validation loss. |
| SOAP, ICLR 2025 | Compare AdamW vs matrix/preconditioned optimizers on LM pretraining; report wall-clock, preconditioner overhead, batch-size sensitivity, and long runs. |
| Fantastic Pretraining Optimizers, ICLR 2026 | Tune/check baselines fairly, but main comparisons are final-budget, across model scales and data-to-model ratios; detect ranking flips over training horizon. |
| AdamW, ICLR 2019 | Include LR/WD heatmaps to show a method is not winning from an unfair regularization/schedule choice. |
| Lion, NeurIPS 2023 | Include broad transfer, batch-size behavior, and limitations, not only one LM curve. |
| Adam-mini, ICLR 2025 | Report memory, throughput, role/block behavior, and optimizer-state footprint. |
| GaLore, ICML 2024 oral | Include optimizer-state memory and feasibility/efficiency tables at meaningful model scale. |
| CAME, ACL 2023 | Report convergence, stability, and memory for memory-efficient adaptive baselines. |
| Schedule-Free, NeurIPS 2024 | Check stopping-horizon effects; compare without relying on one lucky schedule endpoint. |
| AdEMAMix, ICLR 2025 | Include long-horizon token-efficiency and forgetting/distribution-shift behavior. |

## Model And Data Grid

These are the model/data settings for all main experiments.

| id | purpose | layers | d_model | heads | ffn_dim | seq_len | global tokens/step | note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| M0 | main small scale | 12 | 768 | 12 | 2048 | 256 | 32768 | existing 123M setting |
| M1 | medium scale | 18 | 1024 | 16 | 3072 | 256 | 32768 | required after smoke |
| M2 | stretch scale | 24 | 1280 | 16 | 4096 | 256 | 32768 | run if 4xA6000 memory smoke passes |

Datasets:

```text
FineWeb-Edu sample-10BT
FineWeb sample-10BT
DCLM baseline
Dolma sample
WikiText-103 only as a small anchor, not main evidence
```

Validation policy for final runs:

```text
validation skip: 610M tokens
validation budget: 8M tokens
same tokenizer: GPT-2
same validation slice for every optimizer row within a dataset
```

## Experiment 1: Sophia/SOAP-Style LM Speed-To-Target

This is the main experiment.

### Run Matrix

| model | datasets | train budgets | seeds | rows |
| --- | --- | --- | --- | --- |
| M0 | FineWeb-Edu, FineWeb, DCLM | 100M, 300M, 600M | 1337, 2027, 3407 | all rows below |
| M1 | FineWeb-Edu, DCLM | 300M, 600M | 1337, 2027, 3407 | all rows below |
| M2 | FineWeb-Edu | 300M | 1337 | reduced rows below |

Full rows:

```text
SiLU+AdamW
RLB+AdamW
SiLU+Muon
RLB+Muon
SiLU+SOAP-style or reference SOAP
RLB+SOAP-style or reference SOAP
SiLU+Lion
RLB+Lion
SiLU+AdEMAMix
RLB+AdEMAMix
SiLU+Schedule-Free AdamW
RLB+Schedule-Free AdamW
SiLU+CAME-style
RLB+CAME-style
RLB+MatrixPolicy original group-stat
```

M2 reduced rows:

```text
SiLU+AdamW
RLB+AdamW
RLB+SOAP-style or best matrix baseline
RLB+MatrixPolicy original group-stat
```

### Metrics

```text
final validation loss
validation loss AUC
tokens to reach SiLU+AdamW final loss
steps to reach target
GPU-hours to reach target
wall-clock to reach target
optimizer-step time
forward/backward time
tokens/sec
peak CUDA memory
optimizer-state memory
nonfinite/divergence count
```

### Required Figures

```text
loss vs tokens
loss vs GPU-hours
speedup-to-target bar chart
final loss table with mean +/- std
ranking over budget, showing any ranking flips
```

This copies Sophia/SOAP/Fantastic directly: speed is not asserted from a single equal-length run.

## Experiment 2: Fantastic-Style Model/Data Scaling

Purpose: show whether MatrixPolicy's advantage changes with model size and data-to-model ratio.

### Run Matrix

| model | token ratios to run | datasets | seeds | rows |
| --- | --- | --- | --- | --- |
| M0 | 100M, 300M, 600M | FineWeb-Edu, DCLM | 1337, 2027, 3407 | AdamW, best generic matrix/scalar, MatrixPolicy |
| M1 | 300M, 600M | FineWeb-Edu, DCLM | 1337, 2027, 3407 | AdamW, best generic matrix/scalar, MatrixPolicy |
| M2 | 300M | FineWeb-Edu | 1337 | AdamW, MatrixPolicy |

Best generic matrix/scalar means the strongest non-MatrixPolicy row from Experiment 1, frozen before this experiment is summarized.

### Analysis

Fit an optimizer comparison table:

```text
compute multiplier needed by AdamW to match MatrixPolicy loss
compute multiplier needed by best generic optimizer to match MatrixPolicy loss
MatrixPolicy gap vs model size
MatrixPolicy gap vs token budget
```

### Required Figures

```text
loss-vs-compute frontier by model size
speedup vs model size
speedup vs token budget
```

This copies Fantastic Pretraining Optimizers: do not rely on one size or one data ratio.

## Experiment 3: SOAP-Style Batch-Size And Overhead Study

Purpose: optimizer rankings change with batch regime and overhead. MatrixPolicy must be measured like SOAP.

### Run Matrix

| model | dataset | tokens | seeds | global batch tokens | rows |
| --- | --- | ---: | --- | --- | --- |
| M0 | FineWeb-Edu | 100M, 300M | 1337, 2027, 3407 | 16k, 32k, 65k | AdamW, RLB+AdamW, RLB+SOAP, RLB+MatrixPolicy |

For 16k/65k, change per-GPU batch and grad accumulation only; keep total train tokens fixed.

### Metrics

```text
final loss
loss AUC
tokens-to-target
steps-to-target
wall-clock-to-target
optimizer-step overhead
tokens/sec
peak memory
clipping frequency
grad norm before clipping
```

### Required Figures

```text
speedup vs batch size
loss vs wall-clock by batch size
optimizer-step overhead table
```

## Experiment 4: Adam-mini/GaLore/CAME-Style Memory And Throughput

Purpose: show whether MatrixPolicy's loss gain survives overhead and memory accounting.

### Run Matrix

Use M0 and M1, one seed each for profiling, then aggregate from full runs.

```text
models: M0, M1
datasets: FineWeb-Edu
steps: first 1000 steps and full-run averages
rows: AdamW, Muon, SOAP-style, CAME-style, AdEMAMix, MatrixPolicy
```

### Metrics

```text
optimizer state memory, estimated and measured where possible
peak CUDA allocated/reserved
forward/backward seconds
optimizer-step seconds
tokens/sec
extra memory vs AdamW
extra step time vs AdamW
loss improvement per extra GPU-hour
```

### Required Tables

```text
memory table by optimizer
throughput table by optimizer
loss-per-GPU-hour table
```

This copies Adam-mini/GaLore/CAME-style efficiency accounting.

## Experiment 5: Lion/Schedule-Free-Style Broad Transfer

Purpose: show MatrixPolicy is not a one-corpus artifact.

### Run Matrix

| train corpus | evaluation corpus | model | tokens | seeds | rows |
| --- | --- | --- | ---: | --- | --- |
| FineWeb-Edu | FineWeb-Edu, DCLM, Dolma | M0 | 300M | 1337, 2027, 3407 | AdamW, best generic, MatrixPolicy |
| DCLM | DCLM, FineWeb-Edu, Dolma | M0 | 300M | 1337, 2027, 3407 | AdamW, best generic, MatrixPolicy |
| FineWeb | FineWeb, DCLM, Dolma | M0 | 300M | 1337, 2027, 3407 | AdamW, best generic, MatrixPolicy |

Best generic is selected from non-MatrixPolicy rows in Experiment 1. No retuning on transfer corpora.

### Metrics

```text
in-domain loss
cross-corpus loss
transfer degradation
loss AUC
nonfinite/divergence
```

### Required Figure

```text
in-domain vs transfer validation loss scatter
```

This copies Lion/Schedule-Free's broad-transfer expectation but keeps the scope academic.

## Experiment 6: AdEMAMix-Style Long-Horizon And Forgetting

Purpose: accepted optimizer papers test long-horizon behavior, not only early gains.

### Run Matrix

```text
model: M0
dataset order: FineWeb-Edu 300M -> DCLM 300M continued training
seeds: 1337, 2027, 3407
rows: AdamW, best generic, AdEMAMix, MatrixPolicy
```

At the switch and final point, evaluate on:

```text
FineWeb-Edu heldout
DCLM heldout
Dolma heldout
```

### Metrics

```text
continued-training loss on new corpus
forgetting on original corpus
transfer loss on Dolma
wall-clock and GPU-hours to target
```

### Required Figure

```text
old-domain loss and new-domain loss over continued training
```

This copies AdEMAMix's long-horizon/older-gradient framing and the modern concern that optimizers affect forgetting.

## Experiment 7: AdamW-Style Hyperparameter Landscape As Reviewer Defense

Purpose: show the result is not caused by a weak AdamW/baseline setting. This is not the core paper story, but accepted papers include it.

### Run Matrix

```text
model: M0
datasets: FineWeb-Edu, FineWeb
tokens: 50M
seed: 1337
rows: AdamW, Muon, Lion, SOAP-style, AdEMAMix, Schedule-Free AdamW, CAME-style, MatrixPolicy
```

Paper-informed grids:

```text
AdamW: LR {1e-4, 3e-4, 5e-4}, WD {0.03, 0.10, 0.20}
Muon: LR {1e-4, 3e-4, 5e-4}, WD {0.03, 0.10, 0.20}, momentum {0.90, 0.95}
Lion: LR {3e-5, 1e-4, 2e-4}, WD {0.10, 0.30, 0.60}
SOAP-style: LR {1e-4, 3e-4, 5e-4}, WD {0.03, 0.10}, frequency {10, 50}
AdEMAMix: LR {1e-4, 2e-4, 3e-4}, WD {0.03, 0.10}, alpha {2, 5}, beta3 {0.999, 0.9999}
Schedule-Free: LR {1e-4, 3e-4, 5e-4}, WD {0.03, 0.10, 0.20}
CAME-style: LR {1e-4, 3e-4, 5e-4}, WD {0.03, 0.10, 0.20}, confidence {0.5, 1.0}
MatrixPolicy: LR {2e-4, 3e-4, 5e-4}, WD {0.03, 0.10}, adam-scale {2, 3, 4}, group-gain {0.20, 0.35}
```

### Required Figures

```text
LR/WD heatmaps for AdamW, Lion, SOAP, MatrixPolicy
best-row comparison table
stability map showing divergence/nonfinite rows
```

If this finds a stronger baseline, rerun the relevant Experiment 1 rows. Do not change the benchmark.

## Experiment 8: Post-Training Probe

Purpose: show pretrained checkpoints are useful, not only lower pretraining loss.

### Run Matrix

Use checkpoints from M0 300M FineWeb-Edu:

```text
rows: AdamW, RLB+AdamW, best generic, MatrixPolicy
seeds: 1337, 2027, 3407
adaptation budget: 10M tokens
adaptation corpus: small held-out instruction-like or domain mixture if available; otherwise continued-pretraining transfer on WikiText/OpenWebText-style data
```

Metrics:

```text
adaptation loss
heldout loss after adaptation
forgetting on FineWeb-Edu heldout
```

This is secondary, but useful for ICLR reviewers who distrust pretraining loss alone.

## Experiment 9: Mechanism And Diagnostics, Not Main Theory

Purpose: support the optimizer story with measurements, without making gauge the paper center.

Use logs from Experiments 1-6. Produce:

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

Main mechanism claim:

```text
MatrixPolicy allocates updates differently across rational FFN roles and converts optimizer overhead into better loss-per-compute.
```

Gauge-specific diagnostics can go in an appendix only if they clarify results; they are not the main experiment plan.

## Experiment 10: Ablations Last

Only after Experiments 1-6 show MatrixPolicy is truly ahead.

```text
model: M0
dataset: FineWeb-Edu first
tokens: 100M, then 300M for promoted rows
seeds: 1337, 2027, 3407
base: MatrixPolicy original group-stat
ablations: no group-stat scaling, no role-depth policy, no matrix branch, no rebalance if still relevant, gain-only, pressure-only, activity-only
```

Ablations explain the method. They do not choose model size, dataset, token budget, baseline rows, or paper claim.

## Immediate Work Order

```text
1. Validate DCLM and Dolma loaders with tiny 500-step M0 runs.
2. Validate M1 memory with 500-step M1 FineWeb-Edu runs.
3. Start Experiment 1 M0 speed-to-target runs at 100M/300M on FineWeb-Edu and DCLM.
4. Start Experiment 7 hyperparameter landscapes only as reviewer defense, not as the main work.
5. Start Experiment 3 batch-size/overhead runs after first M0 curves confirm runtime.
6. Then run M1 scale, transfer, long-horizon forgetting, post-training probe, diagnostics, and ablations last.
```

This plan contains the new experiments that have not been done yet and follows accepted optimizer-paper experimental style.
