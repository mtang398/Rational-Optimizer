# E9: 100M-Token MatrixPolicy Ablation Plan

## Decision

E9 is a fresh, fixed ablation study on the paper-facing method. It uses the
123.55M-parameter M0 Transformer, a nominal 100M-token budget, all five main
corpora, and three new paired seeds. Every treatment is run contemporaneously
from one frozen code revision and manifest. Existing role/depth V2-V6 runs are
exploratory design history and are not E9 evidence.

The study has two purposes:

1. separate the RLB nonlinear-sublayer effect, the fixed MatrixPolicy optimizer
   recipe, and the signal-conditioned MatrixPolicy controls; and
2. test the conditional contribution of group-gradient gating, the role/depth
   plus transient-Muon matrix control, and reciprocal pair rescaling.

E9 is restricted to the 100M-token regime. It does not select a new method,
retune a schedule, or promote an ablation into the headline configuration.

## Frozen Protocol

| Item | Value |
| --- | --- |
| Model | M0: 12 layers, width 768, 12 heads, nonlinear width 2048 |
| Paper-facing RLB parameter count | 123,552,672 |
| Context length | 256 |
| Per-GPU batch | 16 |
| Gradient accumulation | 2 |
| GPUs per run | 4 A6000 GPUs |
| Tokens per optimizer step | 32,768 |
| Optimizer steps | 3,050 |
| Exact training exposure | 99,942,400 tokens |
| Validation cadence | step 1 and every 50 steps through step 3,050 |
| Outer learning rate | 3e-4 |
| Minimum learning rate | 3e-5 |
| Weight decay | 0.10 |
| Datasets | DCLM, FineWeb-Edu, FineWeb, Dolma-sample, C4 |
| E9 seeds | 2479, 5052, 8913 |

Repository-wide manifest and result search confirms that the E9 seeds have not
appeared in any prior run. They were frozen before E9 execution by sorting the
output of `random.Random(20260710).sample(range(1, 10000), 3)`. Every arm uses the same
per-rank sampled batch-index stream and validation-index stream within a
dataset-seed block. Before launch, each block must also match token-cache hash,
validation-index fingerprint, first-batch fingerprint, and initial-state hash
for A1-A9.

The RLB activation is always `rlb_fused_global_rational`. It contains the
groupwise P5/Q4 global rational response and no local-atom parameter group.
MatrixPolicy arms use `rational_matrix_policy_onpolicy` with an AdamW backbone.
SiLU uses the existing matched SwiGLU-like sublayer and is labelled
`SiLU + AdamW` in paper-facing output.

The optimizer contract is also frozen. All arms use AdamW `beta1=0.9`,
`eps=1e-8`, a 200-step linear warmup, cosine decay to `3e-5`, gradient clipping
at global norm 1.0, weight decay 0.10 on decay-eligible parameters, and zero
decay on the existing no-decay groups. A0/A1 use AdamW `beta2=0.95`. A2-A9 use
the MatrixPolicy parameter partition with `beta2=0.999` for both the backbone
AdamW and RLB-matrix AdamW; their nominal RLB-matrix learning-rate scale is 3.0
unless an intervention explicitly changes the realized mixture.

## Treatment Matrix

All entries marked "off" are exact identity interventions. Unmentioned
settings stay equal to the full method.

| ID | Paper-facing arm | Activation and optimizer | Scientific contrast |
| --- | --- | --- | --- |
| A0 | SiLU + AdamW | SiLU, ordinary AdamW | Nonlinear-sublayer anchor |
| A1 | RLB + AdamW | RLB, ordinary AdamW | A1 vs A0: RLB nonlinear-sublayer effect under the same optimizer |
| A2 | RLB + static MatrixPolicy optimizer-recipe shell | MatrixPolicy parameter partition, backbone/matrix AdamW `beta2=0.999`, 3x matrix learning-rate scale; all three policy actions off | A2 vs A1: bundled fixed-recipe effect; A3 vs A2: net adaptive-policy effect |
| A3 | RLB + MatrixPolicy | Complete paper-facing method | Reference treatment |
| A4 | MatrixPolicy without group-stat gradient gating | A3 with all group gain/pressure/activity multipliers off | A3 vs A4: first control |
| A5 | MatrixPolicy without role/depth factors | A3 with neutral Adam role/depth factors and neutral role/depth factors in the Muon gate; transient Muon remains active | Role/depth contribution with Muon present |
| A6 | MatrixPolicy with the transient Muon branch suppressed at a fixed schedule | A3 with the entire Muon optimizer branch skipped while the scheduled Muon fraction still reduces AdamW; role/depth factors remain active | Applied Muon-branch contribution at a fixed AdamW mixture rate |
| A7 | MatrixPolicy without role/depth factors and with the Muon branch suppressed at a fixed schedule | A5 with the Muon optimizer branch skipped while its role/depth-neutral schedule still reduces AdamW; group gating and pair rescaling remain active | Fourth cell of the descriptive role/depth-by-applied-Muon decomposition |
| A8 | MatrixPolicy without reciprocal pair rescaling | A3 with the post-step positive pair move off | A3 vs A8: third control |
| A9 | MatrixPolicy without the role/depth-Muon action block | A3 with neutral role/depth factors and zero Muon fraction, leaving the nominal full-rate 3x matrix AdamW path, group gating, and pair rescaling | A3 vs A9: complete second-control action block |

A3, A5, A6, and A7 form the predeclared role/depth-by-Muon factorial:

| | Muon branch applied | Muon branch suppressed, schedule fixed |
| --- | --- | --- |
| Role/depth on | A3 | A6 |
| Role/depth off | A5 | A7 |

This decomposition estimates the conditional role/depth effect, conditional
applied-Muon-branch effect, and their interaction while holding the AdamW
mixture schedule fixed. The primary E9 contrast for the complete second-control
action block is A3 vs A9. Factorial quantities are descriptive unless a later
analysis plan explicitly budgets a separate inferential family.

## Exact Intervention Flags

The manifest generator must clone the complete A3 row first and append only the
listed last-wins overrides.

### A2: Static MatrixPolicy Optimizer-Recipe Shell

```text
--rational-matrix-policy-group-gain-strength 0.0
--rational-matrix-policy-group-pressure-strength 0.0
--rational-matrix-policy-group-activity-damping 0.0
--rational-matrix-policy-adam-role-strength 0.0
--rational-matrix-policy-adam-role-strength-final 0.0
--rational-matrix-policy-input-depth-gain 0.0
--rational-matrix-policy-output-depth-gain 0.0
--rational-matrix-policy-muon-strength 0.0
--rational-matrix-policy-muon-lr-scale 0.0
--rational-matrix-policy-final-muon 0.0
--rational-matrix-policy-min-muon 0.0
--rational-matrix-policy-max-muon 0.0
--rlb-gauge-strength 0.0
```

A2 deliberately retains `--rational-matrix-policy-adam-lr-scale 3.0`, the
MatrixPolicy parameter partition, its matrix/backbone betas, weight decay, and
detached-statistic collection. A2 vs A1 is therefore a bundled fixed-recipe
contrast, not a matrix-learning-rate-only contrast. It tests whether those fixed
optimizer choices explain the gain before any signal-conditioned action is
credited.

### A4: No Group-Stat Gradient Gating

```text
--rational-matrix-policy-group-gain-strength 0.0
--rational-matrix-policy-group-pressure-strength 0.0
--rational-matrix-policy-group-activity-damping 0.0
```

### A5: No Role/Depth Factors

```text
--rational-matrix-policy-adam-role-strength 0.0
--rational-matrix-policy-adam-role-strength-final 0.0
--rational-matrix-policy-input-depth-gain 0.0
--rational-matrix-policy-output-depth-gain 0.0
```

Pressure/activity attenuation of the Muon pulse remains active. These flags
remove only its input/output role and layer-depth prior.

### A6: Suppress The Transient Muon Branch At A Fixed Schedule

```text
--no-rational-matrix-policy-apply-muon-update
```

The E9 switch keeps the
original Muon schedule and fraction calculation, keep the AdamW learning rate
at `3 * role_factor * (1 - muon_fraction)`, construct the same optimizer objects,
and skip the entire Muon optimizer branch. Muon momentum is not advanced,
Newton-Schulz is not computed, and Muon's decoupled weight-decay contribution
is absent. The resulting step is attenuated AdamW, omitted Muon branch, then
reciprocal pair rescaling. This is a missing-branch intervention, not a
compute-matched parameter-only intervention. `muon_reset_adam_state` remains
false. Setting the existing Muon strengths to zero is invalid for A6 because it
reallocates the missing mixture fraction to a full-rate AdamW step.

### A7: No Role/Depth Factors And Suppressed Muon Branch

A7 combines the A5 and A6 overrides. It retains the nominal 3x matrix AdamW
scale, group-stat modulation, the role/depth-neutral Muon schedule in the AdamW
mixture factor, and reciprocal pair rescaling.

### A8: No Reciprocal Pair Rescaling

```text
--rlb-gauge-strength 0.0
```

`--rational-transport-strength` remains at its existing value of zero and must
not be used for A8. The implemented reciprocal operation rescales the parameter
pair while leaving nested AdamW/Muon states unchanged; E9 evaluates that exact
paper-facing method.

### A9: No Role/Depth-Muon Action Block

A9 combines the A5 role/depth overrides with the following existing flags:

```text
--rational-matrix-policy-muon-strength 0.0
--rational-matrix-policy-muon-lr-scale 0.0
--rational-matrix-policy-final-muon 0.0
--rational-matrix-policy-min-muon 0.0
--rational-matrix-policy-max-muon 0.0
```

Unlike A6/A7, A9 intentionally sets the Muon fraction to zero and therefore
returns the missing mixture share to the nominal full-rate 3x matrix AdamW path.
It is the practical leave-one-control-out arm used for A3 vs A9.

## Run Count And Execution Order

```text
10 arms x 5 datasets x 3 seeds = 150 independent scientific runs
150 x 99,942,400 = 14,991,360,000 scientific training tokens
10 arm preflights x 80 steps x 32,768 = 26,214,400 preflight tokens
160 submitted jobs and 15,017,574,400 total tokens including preflights
```

Each manifest row is one independent Slurm job. No job may contain multiple
dataset-seed-arm rows. This prevents one preemption from invalidating completed
rows. Within every dataset-seed block, arm order uses a frozen balanced random
permutation generated before launch. All jobs use one approved homogeneous
A6000 node class. Node, GPU identifiers, clocks, power limits, restart count,
and contention indicators are recorded. Scientific runtime contrasts remain
descriptive unless the compared arms are node-matched or analyzed with the
predeclared node effect.

Before the scientific queue, run one 80-step engineering preflight for each of
A0-A9. Preflight losses are never analyzed. Every preflight must
verify expanded configuration, finite updates, intervention fidelity, JSONL
schema, cumulative timing, and telemetry fields.

Existing A3, historical `no_role_depth`, `bypass_muon`, and V2-V6 rows are not
substituted into the fixed E9 matrix. Their quality curves remain useful as
exploratory evidence. Their timing is not E9 evidence because node metadata and
phase-specific cumulative time are incomplete.

## Outcomes

### Primary Endpoint

The primary endpoint is normalized validation-loss AUC over the complete regular
evaluation trajectory from step 50 through step 3,050. For arm `a`, dataset `d`,
and seed `s`, define

```text
auc_100(a,d,s) = trapezoid_integral(val_loss, steps 50:50:3050)
                 / (3050 - 50)
delta_auc(a,d,s) = auc_100(a,d,s) - auc_100(A3,d,s)
```

Positive values favor A3. Report every raw paired point, each dataset's mean and
paired standard deviation, and the equal-dataset macro difference within each
seed. Across-seed uncertainty is summarized from the three seed-level macro
differences; the 15 dataset-seed medians and direction counts are descriptive.
The irregular step-1 validation is plotted but excluded from AUC.

The four primary policy contrasts are:

1. A3 vs A2: all signal-conditioned MatrixPolicy actions;
2. A3 vs A4: group-stat gradient gating;
3. A3 vs A9: role/depth plus transient-Muon action block; and
4. A3 vs A8: reciprocal pair rescaling.

With three independent seeds, E9 is estimation-focused and does not report
confirmatory p-values. A1 vs A0 and A2 vs A1 are descriptive attribution
contrasts. Component-contribution language requires a mean seed-macro
`delta_auc` of at least 0.005 NLL, a positive macro effect in all three seeds,
and a positive dataset mean in at least four of five datasets. Smaller or mixed
effects are reported as such without a binary success claim.

The primary outcome is lexicographic: finite completion through step 3,050
precedes AUC, and every completed run ranks ahead of a numerical failure.
`delta_auc` is computed only for paired finite completions, with paired
completion status reported over all blocks. A component-contribution claim
requires 15/15 finite completions for A3 and the compared arm; failures are
therefore never hidden by complete-case AUC averages.

A5-A7 support the descriptive factorial decomposition. For an outcome `Y`,
report

```text
role_effect = 0.5 * [(Y_A5 - Y_A3) + (Y_A7 - Y_A6)]
muon_effect = 0.5 * [(Y_A6 - Y_A3) + (Y_A7 - Y_A5)]
interaction = (Y_A7 - Y_A6) - (Y_A5 - Y_A3)
```

Positive role or Muon effects mean that suppression raises AUC or loss. The
interaction is the change in the role-removal penalty when the Muon branch is
suppressed at a fixed schedule.

### Secondary Curve Summaries

Final validation loss, final perplexity, target arrival, timing, and partial
validation-loss AUC are secondary. Normalized partial AUC through observed steps
750, 1,500, and 2,250 is the trapezoidal integral over regular validation
observations from step 50 to the named horizon, divided by `horizon - 50`.

### Frozen Target-Arrival Ladder

Targets are fixed before any E9 result is observed:

| Dataset | Easier | Main E1 target | Harder |
| --- | ---: | ---: | ---: |
| DCLM | 4.65 | 4.55 | 4.45 |
| FineWeb-Edu | 4.50 | 4.40 | 4.30 |
| FineWeb | 4.70 | 4.60 | 4.50 |
| Dolma-sample | 4.70 | 4.60 | 4.50 |
| C4 | 4.70 | 4.60 | 4.50 |

For every target, report first observed-checkpoint hit step, exact tokens,
`active_seconds_at_val_loss`, and hit coverage. A complete finite run that does
not reach a target is right-censored at 99,942,400 tokens and the final
evaluation timestamp. A numerical failure is a competing failure outcome at its
actual step and last valid timestamp, never a full-horizon censor.
Report all per-run values and paired status counts: full earlier, tie, ablation
earlier, full-only hit, ablation-only hit, and both censored. A hit always beats
a non-hit; two non-hits remain unresolved. Ordinary means and percentages are
reported only for fully observed paired hits with their coverage denominator.
When full hits and an ablation is censored, report the valid token-saving lower
bound `100 * (99,942,400 - full_hit_tokens) / 99,942,400` against the fixed
token horizon. Target hits are never interpolated.

There are 62 validation observations at steps 1, 50, 100, ..., 3,050. Regular
50-step intervals correspond to 1,638,400 tokens; the initial step-1-to-50 gap
is 1,605,632 tokens.

### Efficiency And Reliability

Report instrumented training-loop wall-clock, cumulative active time to each
target, optimizer step time by schedule phase, training-step tokens/s,
training-loop active throughput, peak allocated/reserved CUDA memory,
gradient-clipping frequency, non-finite events, and completion status.
Training-loop active throughput is
`99,942,400 / summary.total_seconds` for completed runs; a failed run reports
partial throughput as `completed_steps * 32,768 / elapsed_seconds`.
Training-step throughput retains the existing warmup-trimmed step-time
definition.

Every attempt records `timing_attempt_id`, node/GPU metadata, and a monotonic
timer with the same origin as `summary.total_seconds`. Each validation record
stores `active_seconds_at_val_loss` immediately after synchronized validation
and all-reduce, before fixed-probe or spectrum diagnostics, plus
`active_seconds_after_event` and `eval_seconds`. The timer resets on restart,
is monotone within an attempt, and must reconcile with the final summary.
Target time includes every scheduled validation needed to observe the hit.

The timer begins immediately before the training loop. Token loading,
model/optimizer construction, DDP initialization, probe setup, configuration
output, compilation, cache construction, dependency setup, and queue time are
outside this metric and listed explicitly. A signal handler records an
attempt-end event when possible; otherwise an archived partial attempt is
reported with restart count and a lower bound from its last timestamp, never an
invented exact wasted runtime. The 150-run values are labelled instrumented
training-loop runtime. A2-A9 use identical telemetry cadence and probe cost.

Before E9, add an all-step gradient-clipping counter and a run-wide CUDA-memory
maximum that spans training, validation, and probes. Logged-step clipping flags
and per-step-reset memory peaks are not used as run-level frequencies or maxima.

## Mechanism Checks

Mechanism trajectories are manipulation checks and explanatory evidence. They
are not separate significance families.

| Control | Required checks |
| --- | --- |
| Group-stat gradient gate | group multiplier mean/SD/range and clipping rate; pre/post-gate group gradient RMS; pressure and activity distributions |
| Role/depth-Muon matrix step | scheduled and applied Muon branch by layer and role; Adam scale; update-to-weight RMS; applied branch absent in A6/A7 and scheduled fraction zero in A9; branch inactive after progress 0.36 in A3/A5; matrix spectral entropy |
| Reciprocal pair rescaling | applied log move, clipping rate, absolute target mismatch, input/output log-norm ratio, norm product, and sampled same-step fixed-probe outputs immediately before and after rescaling |
| Stability | denominator minimum and p01, gradient clipping, non-finite events, and fixed-probe KL/logit movement |

The current logs already expose role-mean Adam/Muon scales, update-to-weight RMS,
pressure/activity summaries, RLB gain summaries, matrix spectra, and pair norm
ratios. Before E9, add the missing cumulative active timestamp, group-level
pre/post-gate summaries, applied reciprocal log move, reciprocal clipping rate,
pair target mismatch, and the sampled same-step pre/post-rescale probe. The new
Muon-update switch must log both scheduled mixture and whether an update was
applied. A scientific run cannot start until each arm passes its intervention
check.

## Failure And Interpretation Rules

- No efficacy-based early stopping is allowed. Every finite row runs to step
  3,050.
- Numerical failure remains an outcome. Only documented infrastructure faults
  are rerun, with the same row, seed, configuration, and approved node class.
- Preemption archives the incomplete attempt and restarts only that row from
  step zero. Completed rows are never repeated because another row failed.
- Adaptive MatrixPolicy controls receive a contribution claim only when A3 vs
  A2 satisfies the predeclared AUC magnitude and consistency rule.
- Group-stat gating, the role/depth-Muon action block, and reciprocal rescaling
  receive component claims only when A3 vs A4, A3 vs A9, and A3 vs A8,
  respectively, satisfy the same rule. Mixed, negligible, or negative effects
  are reported without component-credit language.
- If the A3/A5/A6/A7 factorial supports only the joint cell, report a
  role/depth-Muon interaction rather than independent contributions.
- A2 vs A1 measures the bundled fixed MatrixPolicy optimizer-recipe effect. A1
  vs A0 measures the RLB nonlinear-sublayer effect under ordinary AdamW.
- E9 results do not change the frozen A3 configuration and do not trigger a new
  V-version search.

## Compute Envelope

Scaling the arm-specific completed 100M runtime envelope to 150 fresh scientific
runs gives a 50.2-74.1 serial-node-hour planning range. The corresponding
arm-weighted historical mean is about 239.6 A6000 GPU-hours. New E9 telemetry is
benchmarked in preflight before these values are used for scheduling.

| Quantity | Estimate |
| --- | ---: |
| Serial node-hours | 50.2-74.1 h |
| A6000 GPU-hours | 200.7-296.4 h |
| Arm-weighted mean A6000 GPU-hours | about 239.6 h |
| Theoretical two-job active lower bound | 25.1-37.1 h |
| Mean-based theoretical lower bound | about 30.0 h |

The lower-bound rows divide total active work by two and are not an executable
schedule guarantee for 150 indivisible jobs. All figures exclude queue delay,
preflights, and infrastructure reruns. They are derived
from `experiments/results/iclr26_runtime_summary_2026_06_11/` and the completed
role/depth run records; they are planning values, not E9 results.

## Required Artifacts Before Submission

```text
abalation/manifests/e9_100m_manifest.csv
abalation/manifests/e9_preflight_manifest.csv
abalation/scripts/build_e9_100m_manifest.py
abalation/scripts/run_e9_100m_manifest_job.sh
abalation/scripts/analyze_e9_100m.py
training/transformer_lm_compare.py  # cumulative timing and Muon-update switch
optimizer_design/matrix_policy_optimizer.py  # Muon-update switch and telemetry
abalation/results/e9_100m/coverage.csv
abalation/results/e9_100m/final_loss.csv
abalation/results/e9_100m/paired_effects.csv
abalation/results/e9_100m/target_arrival.csv
abalation/results/e9_100m/runtime.csv
abalation/results/e9_100m/mechanism_checks.csv
```

The manifest, expanded configurations, code revision, analysis script, target
ladder, and node policy are hashed and frozen before the first scientific row is
submitted. Paper tables and figures are generated only from the validated E9
result package.
