# ICLR Optimizer Experiment Blueprint

This is the claim-first plan for turning RationalOPT into a serious optimizer paper. It is not an ablation-first plan, and it is not centered on the current 3-seed FineWeb/FineWeb-Edu result. That result is preliminary signal. The paper must be won by a stronger evidence stack that would remain convincing even if a reviewer treated the current pilot table only as motivation.

The target claim is:

```text
RLB exposes optimizer-visible rational/gauge geometry, and MatrixPolicy uses that geometry to improve language-model optimization at matched compute over strong generic scalar and matrix optimizers.
```

The plan below is the academic evidence plan before resource partitioning. The scheduler still has hard limits, but those limits do not define the scientific standard.

```text
max 4 A6000 GPUs per job
max 8 A6000 GPUs active at once
repo size below 200G
do not commit checkpoints, caches, or Slurm logs
```

## Accepted Optimizer-Paper Standard

Top optimizer papers do not rely on one favorable row. They establish strong baselines, tuning fairness, efficiency, scaling, and failure-mode accounting.

| paper | venue | standard RationalOPT must match |
| --- | --- | --- |
| AdamW, Loshchilov & Hutter | ICLR 2019 | Separate optimizer effects from regularization and schedule effects; sweep LR and weight decay rather than inheriting defaults. |
| Sophia, Liu et al. | ICLR 2024 | Report steps, tokens, compute, and wall-clock speed-to-target across GPT scales; include optimizer overhead and stability. |
| GaLore, Zhao et al. | ICML 2024 oral | If memory or optimizer-state efficiency is discussed, report optimizer-state memory and full training curves at meaningful LM scale. |
| SOAP, Vyas et al. | ICLR 2025 | Compare against Adam and Shampoo/Adafactor-style methods; report LM pretraining, stability, memory, throughput, and hyperparameter sensitivity. |
| Adam-mini, Zhang et al. | ICLR 2025 | If the optimizer uses parameter roles or blocks, justify the partition and report memory/trajectory behavior, not only final loss. |
| Lion, Chen et al. | NeurIPS 2023 | Evaluate transfer, LR/WD sensitivity, compute/memory behavior, and limitations. |
| Schedule-Free, Defazio et al. | NeurIPS 2024 | Test multiple stopping horizons rather than relying on one schedule point. |
| AdEMAMix, Pagliardini et al. | ICLR 2025 | Demonstrate token efficiency over long horizons and use paper-style alpha/beta3 warmup behavior. |
| CAME, Luo et al. | ACL 2023 | Report convergence and stability together with memory footprint for Adam/Adafactor-like baselines. |
| Cautious Optimizers, Liang et al. | ICLR 2026 | If claiming a simple optimizer improvement, show consistent LLM pretraining/post-training gains with minimal extra tuning. |
| Fantastic Pretraining Optimizers, Wen et al. | ICLR 2026 | Tune each optimizer seriously, compare at final budgets, check model scale and data-to-model ratio, and avoid intermediate-checkpoint ranking traps. |

Primary source links:

```text
AdamW: https://openreview.net/forum?id=Bkg6RiCqY7
Sophia: https://proceedings.iclr.cc/paper_files/paper/2024/hash/06960915ba8674c7a898ec0b472b80ff-Abstract-Conference.html
GaLore: https://openreview.net/forum?id=hYHsrKDiX7
SOAP: https://openreview.net/forum?id=IDxZhXrpNf
Adam-mini: https://proceedings.iclr.cc/paper_files/paper/2025/hash/45ae878717399e6f62d57c65f052cd46-Abstract-Conference.html
Lion: https://papers.neurips.cc/paper_files/paper/2023/hash/9a39b4925e35cf447ccba8757137d84f-Abstract-Conference.html
Schedule-Free: https://neurips.cc/virtual/2024/poster/96925
AdEMAMix: https://iclr.cc/virtual/2025/poster/28625
CAME: https://aclanthology.org/2023.acl-long.243/
Cautious Optimizers: https://openreview.net/forum?id=zBPZeRjfgu
Fantastic Pretraining Optimizers: https://openreview.net/forum?id=2J51qUZ0iG
```

The consequence is simple: RationalOPT needs a decisive optimizer benchmark, speed-to-target, scaling, transfer, overhead accounting, and mechanism evidence tied to RLB geometry. Method ablations are necessary later, but they cannot be the starting point.

## Evidence Stack

The paper should be organized around six evidence blocks, in this order.

### Benchmark Lock Before Tuning

Before launching any optimizer tuning, write and freeze a benchmark card. This prevents the project from using ablations or exploratory screens to choose the favorable setting.

The benchmark card must specify:

```text
model sizes and parameter budgets
datasets and held-out transfer corpus
train-token budgets and validation-token budgets
training/validation slice construction
seed set
global batch, sequence length, warmup, and scheduler family
final validation metric
speed-to-target loss thresholds
optimizer families eligible for final comparison
tuning split and final split
maximum tuning budget per optimizer family
rules for retiring unstable or dominated baselines
exact policy for divergent/nonfinite runs
```

After the card is frozen, tuning can choose optimizer hyperparameters only within that card. It cannot change the benchmark to favor MatrixPolicy, and component ablations cannot be used to pick the main test setting.

### 1. Decisive Headline Benchmark

Goal: show that MatrixPolicy wins as an optimizer, not because of weak baselines or one lucky pilot run.

Required rows:

```text
SiLU/SwiGLU + tuned AdamW
RLB + tuned AdamW
SiLU/SwiGLU + tuned Muon or accepted matrix optimizer baseline
RLB + tuned Muon or accepted matrix optimizer baseline
RLB + MatrixPolicy
RLB + SOAP/Shampoo-style or reference SOAP if available
SiLU/SwiGLU + strongest non-RLB optimizer family that remains stable
```

Required protocol:

```text
same model architecture within each table
same tokenizer
same train/validation split construction
same sequence length and global batch
same token budget
same final evaluation budget
same seeds
tuned optimizer-specific hyperparameters selected on tuning splits
final-budget comparison, not only early checkpoints
```

Primary datasets:

```text
FineWeb-Edu
FineWeb
DCLM or Dolma-style held-out transfer corpus
one additional transfer evaluation not used for HPO
```

Seed standard:

```text
5 seeds at the 123M setting if affordable
3 seeds at larger scale
paired reporting whenever the same seeds are available across methods
```

The current 3-seed FineWeb/FineWeb-Edu result is a pilot supporting this plan. It is not the final headline benchmark.

### 2. Strong Optimizer Tuning

Goal: make a reviewer believe the baselines were not under-tuned.

Tune each optimizer family separately. Do not force AdamW hyperparameters onto methods with different update geometry.

Required tuning surfaces:

```text
learning rate
weight decay
warmup length
minimum LR or schedule endpoint where scheduled
gradient clipping threshold
optimizer-specific beta / momentum / precondition-frequency / alpha parameters
RLB-specific MatrixPolicy group and rebalance strengths
```

Fantastic Pretraining Optimizers highlights that different optimizers have different optima and that rankings can flip during LR decay. Selection must therefore be based on final-budget behavior or shortened schedules explicitly calibrated against final-budget behavior.

This is tuning for the main result, not a broad optimizer zoo. Low-value families should be retired once a fair screen shows they are dominated, but the paper must keep enough accepted optimizer families to make the comparison hard.

### 3. Speed-To-Target And Compute Accounting

Goal: show whether MatrixPolicy is useful after accounting for slower steps.

Report:

```text
tokens-to-target validation loss
steps-to-target validation loss
wall-clock-to-target
GPU-hours-to-target
forward/backward time
optimizer-step time
tokens/sec
peak CUDA memory
optimizer-state memory estimate
nonfinite/divergence rate
gradient-clipping frequency
```

MatrixPolicy has per-step overhead. The paper must show when lower loss repays that overhead and when it does not.

### 4. Scale And Token-Budget Laws

Goal: show the effect is not a small-model or short-budget artifact.

Required axes:

```text
current 123M model
one larger dense model that still fits academic 4-A6000 jobs
at least two token budgets at 123M
at least one longer run beyond the current 100M-token pilot
held-out transfer where HPO-selected configs are reused without retuning
```

Industrial-scale LLM pretraining is not required. The right standard is strong academic scaling evidence: enough model and token variation to show the trend is real and to expose any collapse.

### 5. Mechanism Tests Before Method Ablations

Goal: connect the win to RLB geometry and MatrixPolicy behavior.

Mechanism experiments should be run after headline configs are selected, but before component ablations become the story.

Required mechanism tests:

```text
gauge-equivalent initialization test
mid-training gauge perturbation and recovery test
optimizer-state gauge covariance test
function-space movement on fixed probes
role-specific update/weight RMS for W_in and W_out
gauge drift and norm-product tracking
denominator/pole-margin safety
rational derivative/output activity
matrix spectrum / SVD entropy by role
```

The mechanism claim should be narrow:

```text
MatrixPolicy improves useful function movement while controlling nonfunctional gauge drift and rational-role instability.
```

### 6. Ablations Last

Ablations are necessary, but they are explanatory once the main optimizer result is real.

Late-stage ablations:

```text
remove group-stat scaling
remove gauge rebalance
remove role-depth policy
remove early matrix-normalized branch
gain-only / pressure-only / activity-only group policy
coefficient trust-radius variants
MatrixPolicy with generic RLB telemetry disabled
```

Run ablations on the selected benchmark setting first. Promote only informative ablations to full multi-corpus runs.

## Experiment Sequence

1. Freeze the exact original `rational_matrix_policy_onpolicy` group-stat method that produced the pilot. Any v2 work must be a separate branch and a separate paper claim.
2. Freeze the benchmark card before any tuning job is launched.
3. Build the accepted-paper comparison matrix: reference implementation, deviations, tuned hyperparameters, memory state, failure modes, and whether a baseline is exact or style-matched.
4. Run targeted optimizer tuning for the frozen headline benchmark on tuning splits. This is not a broad ablation queue and not a setting-selection process.
5. Run final matched headline tables with frozen configs, final validation slices, paired seeds, and no final-table retuning.
6. Convert the same final runs into speed-to-target, wall-clock, GPU-hour, memory, throughput, clipping, and divergence tables.
7. Run scaling and transfer across model size, token budget, and held-out corpus.
8. Run mechanism tests: gauge perturbations, fixed probes, role update geometry, denominator safety, and spectrum diagnostics.
9. Run method ablations only after the main claim is established.
10. Write a concise paper: claim, optimizer, decisive benchmark, speed, scaling, mechanism; put full surfaces and failed rows in the appendix.

## What Not To Do

```text
do not center the paper plan on the current 3-seed pilot
do not start with method-component ablations
do not run a broad optimizer zoo without a headline purpose
do not report intermediate-checkpoint wins as final optimizer wins
do not hide RLB+AdamW divergence or failed broad-baseline rows
do not claim MatrixPolicy v2 results as the original MatrixPolicy method
do not weaken the scientific plan because the current scheduler is inconvenient
```

## Current Pilot Evidence

The current pilot remains useful because it shows the claim is plausible:

```text
FineWeb, 3 seeds:
  RLB+MatrixPolicy mean val loss 4.369701
  SiLU+AdamW mean val loss        4.528963
  best non-MatrixPolicy control  4.522311

FineWeb-Edu, 3 seeds:
  RLB+MatrixPolicy mean val loss 4.069422
  SiLU+AdamW mean val loss        4.223572
  best non-MatrixPolicy control  4.223572
```

This evidence motivates the full plan. It does not replace it.

## Operational Partitioning

Only after the scientific plan is fixed should it be mapped onto the cluster:

```text
one Slurm job = 4 A6000 GPUs
at most two jobs active
use dependent chains
chunk long tuning surfaces
archive or skip completed JSONL rows
commit compact summaries and traces only
```

Operational scripts may run bounded slices, but README/TODO files should preserve the full academic evidence target.
