# ICLR Optimizer Experiment Blueprint

This is the experiment program for turning RationalOPT into a serious optimizer paper. It is not a minimal plan and it is not ablation-first. The target is a full optimizer-benchmarking story: fair tuning, scale, efficiency, mechanism, downstream utility, and enough stress testing that the claim does not depend on one favorable setting.

Hard resource rules:

```text
max 4 A6000 GPUs per job
max 8 A6000 GPUs active at once
repo size below 200G
do not commit checkpoints, caches, or Slurm logs
```

## What Related Optimizer Papers Actually Require

The experimental standard comes from these papers:

| paper | relevant standard for this repo |
| --- | --- |
| SOAP, Vyas et al. 2024 | Learning-rate sweep first; then two-dimensional sweeps with LR and one optimizer hyperparameter. Report throughput. Estimate efficiency by running shortened cosine schedules at fractions of the full budget instead of comparing only equal-length runs. Include a matrix-preconditioned baseline because MatrixPolicy is partly a matrix optimizer claim. |
| Sophia, Liu et al. 2024 | Evaluate several model sizes; report wall-clock/compute overhead; report stability statistics such as gradient-clipping frequency; show hyperparameter sensitivity and transfer. |
| Muon/Kimi, Liu et al. 2025 | Matrix optimizers need update-scale accounting, weight-decay treatment, and matrix-spectrum evidence. Claims should be about compute efficiency and scaling, not just one final loss. |
| Lion, Chen et al. 2023 | Show LR/weight-decay sensitivity, batch-size sensitivity, and where gains weaken. Optimizer papers need surfaces, not only best rows. |
| Adam-mini, Zhang et al. 2024 | If the optimizer uses parameter roles/blocks, show why the block structure matters. Compare memory and trajectory behavior, not only loss. |
| GaLore, Zhao et al. 2024 | Memory-efficient optimizer claims need optimizer-state memory estimates and training progression curves at multiple model sizes. |
| CAME, Luo et al. 2023 | Memory-efficient adaptive optimizers should be tested on stability and performance, not just memory. If included, report factorized-state memory and clipping/instability rates. |
| AdEMAMix, Pagliardini et al. 2024 | Token-efficiency claims should report tokens-to-loss and long-horizon behavior, because dual-momentum methods are designed to reuse older gradients. |
| Schedule-Free AdamW / SF-NorMuon, Defazio et al. 2024-2026 | Horizon-free claims require checking multiple stopping horizons. Even if MatrixPolicy keeps a cosine schedule, the paper must show whether wins persist at early, middle, and final horizons. |
| Fantastic Pretraining Optimizers, Wen et al. 2026 | The strongest warning: fixed hyperparameters across optimizers are not fair; rankings can flip during LR decay; evaluate at final budget across model size and data-to-model ratio. |

Therefore the paper cannot be built from one default protocol plus component ablations. It needs a complete optimizer benchmark where MatrixPolicy wins after the baselines have been tuned seriously.

## Full Paper Standard, Not A Small Run Plan

The ICLR-level target is:

```text
7 to 10 optimizer families
2 activation families where relevant: SiLU/SwiGLU and RLB
2 HPO corpora before final selection: FineWeb-Edu and FineWeb
3 final corpora: FineWeb-Edu, FineWeb, DCLM or Dolma
2 to 3 model sizes
2 token budgets per main model size
5 seeds at 123M and 3 seeds at larger scale if cost forces it
speed-to-target curves, not only equal-token tables
mechanism interventions tied to RLB gauge/rational structure
downstream zero-shot sanity checks from selected final checkpoints
```

The resource constraint changes scheduling, not ambition:

```text
run at most two 4-A6000 jobs simultaneously
use dependent Slurm chains for long queues
checkpoint locally but commit only compact traces and summaries
stop a branch only when the evidence says it is dominated after the scheduled confirmation, not after one noisy early run
```

## Paper Claim To Test

The claim is:

```text
RLB exposes rational/gauge structure, and MatrixPolicy uses that structure to produce better pretraining optimization than generic scalar or generic matrix optimizers at matched compute.
```

The required evidence chain is:

```text
fairly tuned optimizer benchmark
=> speed-to-target and final-budget wins
=> scaling over model size and token budget
=> mechanism evidence tied to RLB gauge/rational geometry
=> downstream sanity checks from saved final checkpoints
```

## Required Code Before New GPU Runs

Telemetry and broad baseline optimizer wiring are now implemented. The remaining pre-launch requirement is to validate them under CUDA/DDP and add the Phase A HPO launch/summarization layer. The fields below define what must be checked in that validation.

### 1. Add Optimizer Telemetry

File: `training/transformer_wikitext103_compare.py`

Add JSONL fields/events at every `log_interval`:

```text
grad_global_norm_before_clip
grad_clip_triggered
grad_clip_threshold
optimizer_step_seconds
forward_backward_seconds
cuda_max_memory_allocated
cuda_max_memory_reserved
```

For `rational_matrix_policy_onpolicy`, add a method on `RationalMatrixPolicyOptimizer` that returns per-role telemetry:

```text
matrix_policy_muon_mix_mean_by_role
matrix_policy_adam_lr_scale_mean_by_role
matrix_policy_update_rms_by_role
matrix_policy_weight_rms_by_role
matrix_policy_update_to_weight_rms_by_role
matrix_policy_group_scale_mean/std/min/max
matrix_policy_pressure_mean/std
matrix_policy_activity_mean/std
```

For RLB modules, log from `_rlb_optimizer_stats`:

```text
rlb_output_rms_mean/std_by_layer
rlb_derivative_rms_mean/std_by_layer
rlb_atom_rms_mean/std_by_layer
rlb_abs_moment_mean/std_by_layer
```

Add denominator-risk probes on the rational basis grid:

```text
denominator_abs_min_by_layer
denominator_abs_p01_by_layer
denominator_abs_median_by_layer
```

Add gauge metrics for every RLB FFN:

```text
w_in_rms_by_layer
w_out_rms_by_layer
log_w_in_over_w_out_by_layer
log_norm_product_by_layer
```

These are not optional. They make the paper about the actual method instead of a loss-table artifact.

### 2. Add Fixed-Probe Function Movement

At prepare time, create a deterministic probe batch per dataset/seed from the validation stream and store token IDs in the run directory.

At every eval interval, log:

```text
probe_logit_rms
probe_logit_delta_rms_since_prev_eval
probe_logit_delta_rms_since_step1
probe_kl_since_prev_eval
probe_kl_since_step1
```

This is the function-space analogue of Adam-mini trajectory analysis and is directly relevant to the RLB geometry claim.

### 3. Add Matrix Spectrum Sampling

At eval intervals divisible by 250 or 500 steps, sample matrix SVD entropy for:

```text
attention q/k/v/o weights
FFN W_in
FFN W_out
```

Log macro-averages by role:

```text
svd_entropy_attn_q
svd_entropy_attn_k
svd_entropy_attn_v
svd_entropy_attn_o
svd_entropy_rlb_in
svd_entropy_rlb_out
```

This mirrors Muon's spectrum analysis but uses our roles.

### 4. Matrix-Preconditioned Baseline Status

Muon alone is insufficient. The harness now includes `soap_adamw`, a SOAP/Shampoo-style eigenbasis AdamW baseline for eligible 2D tensors.

Required implementation target:

```text
optimizer name: soap_adamw
2D parameters: SOAP/Shampoo eigenbasis AdamW
1D, embedding, LM head: AdamW
precondition frequencies: 10, 50, 100
large side fallback: identity rotation when dimension is too large
state precision: fp32 first
```

Do not call this a final SOAP reproduction until matched against the original implementation details. In paper tables, label it accurately as `SOAP-style AdamW eigenbasis` unless it exactly matches a known implementation.

### 5. Broad Baseline Family Status

The final benchmark should not look like MatrixPolicy was compared only to weak or convenient controls. The harness now exposes these families for Phase A tuning:

| family | required variants | why it is needed |
| --- | --- | --- |
| AdamW | SiLU/SwiGLU, RLB | standard baseline and current strongest generic baseline |
| Muon | SiLU/SwiGLU, RLB | spectral/matrix optimizer baseline, update-scale treatment required |
| `soap_adamw` | SiLU/SwiGLU, RLB | direct matrix preconditioning competitor; style baseline until reference-matched |
| `lion` | SiLU/SwiGLU, RLB if stable | sign/momentum optimizer baseline with LR/WD surfaces |
| `ademamix` | SiLU/SwiGLU, RLB if stable | token-efficiency competitor with long-gradient-memory behavior |
| `schedule_free_adamw` | SiLU/SwiGLU, RLB if stable | horizon-free competitor and stopping-time stress test |
| `adafactor_came` | SiLU/SwiGLU, RLB if stable | factorized-memory adaptive optimizer baseline; style baseline until reference-matched |
| MatrixPolicy | RLB primary; SiLU/SwiGLU only if a non-RLB fallback is meaningful | proposed method |

Optional only if implementation time allows:

```text
GaLore or APOLLO-style low-rank/scaled optimizer as a memory-efficiency comparison
Sophia-style Hessian estimator if the estimator cost can be measured correctly
SF-NorMuon if a maintained implementation is available and compatible with the repo
```

Optional baselines cannot replace the required matrix, Adam-family, sign/momentum, and schedule-free baselines above.

### 6. Add HPO/Benchmark Summarizers

Create:

```text
experiments/scripts/run_optimizer_hpo_202606xx.sh
experiments/scripts/summarize_optimizer_hpo.py
experiments/scripts/summarize_speed_to_loss.py
experiments/scripts/summarize_mechanism.py
```

The HPO summarizer must output:

```text
hpo_candidates.csv
hpo_best_by_family.csv
hpo_lr_wd_heatmaps/
tokens_to_target.csv
wallclock_to_target.csv
optimizer_overhead.csv
mechanism_summary.csv
```

## Experiment Phase A: Fair HPO, Not Final Results

Purpose: choose fair optimizer settings before final comparisons and produce the hyperparameter-sensitivity figures reviewers expect from optimizer papers.

Datasets:

```text
FineWeb-Edu sample-10BT
FineWeb sample-10BT
validation offset: same 110M-token offset used in current results
seed: 1337 for HPO selection on both corpora
model: current 123M config
sequence length: 256
global tokens/step: 32768
```

Schedule:

```text
pilot length: 1000 steps
confirmation length: 3050 steps for the top configs
eval interval: 50
log interval: 10
```

Optimizer families:

```text
SiLU+AdamW
RLB+AdamW
SiLU+Muon
RLB+Muon
SiLU+SOAP-style
RLB+SOAP-style
SiLU+Lion
RLB+Lion
SiLU+AdEMAMix
RLB+AdEMAMix
SiLU+Schedule-Free AdamW
RLB+Schedule-Free AdamW
SiLU+Adafactor/CAME
RLB+Adafactor/CAME
RLB+MatrixPolicy
```

HPO structure:

1. Run LR by weight-decay surfaces for every required optimizer family on both HPO corpora.
2. For each family, hold the top LR/WD band and sweep the family-specific parameters below.
3. Run beta/min-LR/clip/eps sensitivity for Adam-like methods.
4. Run update-scale/weight-decay/momentum sensitivity for matrix and sign optimizers.
5. Confirm the top 5 configs per family at 3050 steps on both HPO corpora.
6. Carry forward either one shared config per family or two corpus-specific configs if the ranking differs materially between FineWeb and FineWeb-Edu.

Shared grid:

```text
lr:             1e-4, 2e-4, 3e-4, 4.5e-4, 6e-4
weight_decay:   0.03, 0.05, 0.10, 0.20, 0.30
beta2:          0.95, 0.98, 0.999
min_lr_ratio:   0.00, 0.03, 0.10, 0.20
grad_clip:      0.5, 1.0, 2.0
eps:            1e-8, 1e-6 for AdamW-style methods only
```

Muon-specific grid:

```text
muon_adjust_lr_fn: original, match_rms_adamw
muon_momentum:     0.90, 0.95, 0.98
muon_ns_steps:     3, 5
weight_decay:      0.03, 0.10, 0.20, 0.30
```

SOAP-style grid:

```text
precondition_frequency: 10, 50, 100
large_side_identity_threshold: 1024, 2048, 4096
diagonal_preconditioner: AdamW, Adafactor-style
one_sided: false, true
```

Lion grid:

```text
lr:           3e-5, 6e-5, 1e-4, 2e-4, 3e-4
weight_decay: 0.03, 0.10, 0.20, 0.30, 0.50
beta1:        0.90, 0.95
beta2:        0.98, 0.99
```

AdEMAMix grid:

```text
lr:             1e-4, 2e-4, 3e-4, 4.5e-4
weight_decay:   0.03, 0.10, 0.20, 0.30
beta1:          0.90, 0.95
beta2:          0.95, 0.98, 0.999
beta3/slow_ema: implementation defaults plus one slower and one faster setting
alpha/slow_mix: implementation defaults plus half and double
```

Schedule-Free AdamW grid:

```text
lr:             3e-4, 6e-4, 1e-3, 1.5e-3
weight_decay:   0.03, 0.10, 0.20, 0.30
beta1:          implementation default, 0.90, 0.95
warmup_steps:   100, 500, 1000
```

Adafactor/CAME grid:

```text
lr:                    1e-4, 2e-4, 3e-4, 4.5e-4
weight_decay:          0.03, 0.10, 0.20, 0.30
factored_second_moment: true
relative_step:          false
confidence_beta/CAME:   implementation default plus one lower and one higher setting
```

MatrixPolicy grid:

```text
base lr:                         2e-4, 3e-4, 4.5e-4
matrix_policy_adam_lr_scale:      2.0, 3.0, 4.0
matrix_policy_adam_role_strength: 0.8, 1.2, 1.6
matrix_policy_beta2:              0.95, 0.999
matrix_policy_muon_strength:      0.0, 0.375, 0.75
matrix_policy_end:                0.06, 0.12, 0.20
rlb_gauge_strength:               0.25, 0.50, 0.75
group_gain_strength:              0.0, 0.10, 0.20, 0.35
group_pressure_strength:          0.0, 0.05, 0.10, 0.20
group_activity_damping:           0.0, 0.10, 0.20, 0.35
group_min/max_scale:              [0.75,1.35], [0.65,1.55]
```

Selection metric:

```text
primary: final validation loss at the end of the schedule
secondary: area under validation-loss curve after warmup
tie-breakers: lower GPU-hours to same loss, lower instability/clipping frequency
```

Do not pick configs from early-step rankings if they reverse by the end of cosine decay.

Required HPO figures:

```text
LR by weight-decay heatmap for each family and activation
family-specific sensitivity strips for the parameters above
rank over training horizon at steps 250/500/1000/2000/3050
tokens/sec and optimizer-step overhead by family
clip-trigger rate by family
```

## Experiment Phase B: Final Benchmark

Purpose: produce the main paper tables.

Use only configs selected in Phase A.

Datasets:

```text
FineWeb-Edu sample-10BT
FineWeb sample-10BT
DCLM baseline or Dolma sample, whichever is reliable in the local HF/cache environment
```

Seeds:

```text
1337, 2027, 3407, 4517, 5153
```

If cost forces fewer seeds at larger scale, use 5 seeds at 123M and 3 seeds at larger scale.

Model/token settings:

| setting | model | token budgets |
| --- | --- | --- |
| S1 | current 123M, 12L d768 h12 ffn2048 | 100M, 300M |
| S2 | medium target around 300M-355M | 100M, 300M if feasible |
| S3 | largest feasible 4xA6000 config | capacity probe, then at least 100M if stable |

The S2/S3 configs must be chosen by memory probes, not guessed. Capacity probes are not evidence runs; they only choose the largest safe shape under the 4-GPU/job limit.

Main methods:

```text
SiLU+AdamW tuned
RLB+AdamW tuned
SiLU+Muon tuned
RLB+Muon tuned
SiLU+SOAP-style tuned
RLB+SOAP-style tuned
SiLU+Lion tuned
SiLU+AdEMAMix tuned
SiLU+Schedule-Free AdamW tuned
SiLU+Adafactor/CAME tuned
RLB+MatrixPolicy tuned
```

RLB variants of Lion, AdEMAMix, Schedule-Free AdamW, and Adafactor/CAME remain in the final table if their HPO-confirmed runs are stable. If they are unstable, report the instability rate and keep the strongest stable SiLU/SwiGLU version as the required baseline.

Primary paper table columns:

```text
method
activation
optimizer family
model size
dataset
tokens
seeds
mean final val loss
std final val loss
mean PPL
tokens/sec
GPU-hours
GPU-hours to target
max memory
clip-trigger rate
```

## Experiment Phase C: Speed-To-Target

Purpose: match SOAP/Sophia-style efficiency claims.

For each of the strongest methods from Phase B:

```text
SiLU+AdamW tuned
best generic matrix optimizer tuned
RLB+MatrixPolicy tuned
```

Run shortened cosine schedules at:

```text
0.50, 0.625, 0.75, 0.875, 1.00 of full token budget
```

For 100M-token setting:

```text
50M, 62.5M, 75M, 87.5M, 100M tokens
```

For 300M-token setting:

```text
150M, 187.5M, 225M, 262.5M, 300M tokens
```

Fit per method:

```text
loss(N) = a + b * N^(-beta)
```

Report:

```text
tokens_to_match_best_AdamW_final_loss
GPU_hours_to_match_best_AdamW_final_loss
wallclock_speedup_at_target_loss
confidence bands from seed/bootstrap when available
```

This avoids the common mistake of declaring speedup from one equal-length run.

## Experiment Phase D: RLB Geometry Mechanism

Purpose: show the win is from RLB-specific optimizer geometry, not merely another LR schedule.

### D1. Gauge-Equivalent Initialization

Construct function-equivalent RLB initializations with positive gauge changes:

```text
log gauge std: 0.0, 0.25, 0.50, 0.75
same seed
same represented function after compensating W_in and W_out
```

Run:

```text
RLB+AdamW tuned
RLB+Muon tuned
RLB+SOAP-style tuned
RLB+MatrixPolicy tuned
```

Measure:

```text
final loss spread across equivalent gauges
early AUC loss spread
gauge drift
function-probe movement
update_to_weight_rms_by_role
```

Expected strong result:

```text
MatrixPolicy has lower sensitivity to equivalent gauge choices than generic optimizers while maintaining or improving loss.
```

### D2. Mid-Training Gauge Intervention

At step 25% of schedule, apply a function-preserving gauge transform:

```text
W_in[g]  <- exp(delta_g) W_in[g]
W_out[g] <- exp(-delta_g) W_out[g]
delta_g sampled with std 0.50
```

Run each optimizer with:

```text
no intervention
intervention
intervention plus covariant optimizer-state repair where implemented
```

Measure:

```text
loss spike at next eval
recovery steps
final loss delta
change in update_to_weight_rms_by_role
probe KL jump
```

This is the cleanest causal test of the gauge-specific optimizer story.

### D2b. Optimizer-State Gauge Covariance

For MatrixPolicy and every adaptive RLB baseline with optimizer state:

```text
start from the same checkpoint
apply gauge transform at 25% schedule
run with optimizer state unchanged
run with optimizer state transformed/repaired where mathematically defined
run with optimizer state reset for RLB tensors only
```

Measure:

```text
one-step update cosine before/after gauge
one-step function delta before/after gauge
loss spike and recovery
final validation loss
```

This separates "the model representation changed" from "the optimizer state is not covariant to the RLB gauge."

### D3. Rational Function Movement

For each RLB layer/group, evaluate the rational curve on a fixed grid at eval checkpoints.

Measure:

```text
curve L2 movement
curve derivative L2 movement
denominator minimum
atom RMS change
coefficient update norm
```

Relate those to loss improvements and group activity.

### D4. Role-Specific Update Geometry

Compare:

```text
W_in update RMS
W_out update RMS
coefficient update RMS
attention matrix update RMS
embedding update RMS
```

The paper needs to show MatrixPolicy is actually doing something role-specific and that the role-specific behavior correlates with training gains.

## Experiment Phase E: Downstream Evaluation

Save final checkpoints for selected runs only. Do not commit them.

Evaluate with an LM harness on:

```text
LAMBADA
HellaSwag
PIQA
ARC Easy
ARC Challenge
BoolQ
Winogrande
```

Use this as a secondary table. The optimizer paper still centers on pretraining loss and efficiency, but downstream sanity checks prevent the result from looking like validation-slice overfitting.

## Experiment Phase F: Paper Figures

Main paper figures:

1. Final validation loss vs GPU-hours for tuned methods.
2. Validation loss curves with mean +/- std over seeds.
3. Tokens-to-target curves from shortened schedules.
4. LR/WD sensitivity surfaces for every required optimizer family.
5. Family-specific hyperparameter sensitivity strips.
6. Rank-over-horizon plot showing whether rankings flip during LR decay.
7. Scale plot: improvement vs model size and token budget.
8. Gauge-equivalent initialization sensitivity plot.
9. Mid-training gauge intervention recovery plot.
10. Optimizer-state gauge covariance plot.
11. Role-specific update RMS and gauge drift.
12. Rational curve movement and denominator margin over training.
13. Matrix spectrum entropy by role.
14. Downstream zero-shot table for selected checkpoints.

Appendix figures:

```text
all HPO LR/WD/beta2 surfaces
all per-seed curves with mean +/- std
all datasets separately
optimizer overhead microbenchmarks
memory usage curves
failed/diverged runs
exact command provenance
dataset-transfer table for HPO-selected configs
large-model memory-probe table
```

## Launch Discipline

No expensive run should start until:

```text
telemetry fields are implemented and tested
HPO launcher writes exact resolved config per run
summarizers can ingest partial/incomplete runs
run directories include commit hash and source diff status
repo size check passes
squeue check confirms the 8-GPU active cap will not be exceeded
```

At most two active jobs:

```text
job A: 4 A6000
job B: 4 A6000
total: 8 A6000
```

First launches after code is ready:

```text
job A: HPO surfaces for AdamW/Muon/SOAP-style/Lion families, 4 A6000
job B: HPO surfaces for AdEMAMix/Schedule-Free/Adafactor-CAME/MatrixPolicy families, 4 A6000
```

Use dependent Slurm chains for the FineWeb pass after the FineWeb-Edu pass, so there are never more than two active jobs.

Do not run method-component ablations before Phase A selects credible tuned configs. Method-component ablations before a fair tuned benchmark are not evidence.
