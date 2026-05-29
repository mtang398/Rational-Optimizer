# Read First

This repo is about designing an RLB-specific optimizer. Do not claim wins from a different global LR schedule, and do not treat Jacobian or quotient optimizers as baselines.

## Non-Negotiables

```text
main controls:       SiLU/SwiGLU+AdamW, RLB+AdamW, SiLU/SwiGLU+Muon, RLB+Muon
current optimizer:   rational_matrix_policy_onpolicy on rlb_fused_fixed_strong_ffn
GPU type:            A6000 only
normal allocation:   --gres=gpu:nvidia_rtx_a6000:4
concurrency cap:     8 A6000s total
raw run policy:      do not commit experiments/runs/ JSONL folders or Slurm logs
```

Check jobs before launching:

```bash
squeue -u mt872
```

## Active Fair Rerun

The clean synthetic rerun is Slurm job `937608`, launched on May 29, 2026:

```text
script:       experiments/scripts/run_synthetic_fair_full_20260529.sh
job name:     synth-fair
walltime:     24h
GPUs:         4x nvidia_rtx_a6000
output root:  experiments/runs/synthetic_fair_full_20260529/
```

It reruns every synthetic task from scratch with the same LR settings across all rows. Old partial synthetic outputs are superseded.

## Current Optimizer Defaults

```text
optimizer                                           rational_matrix_policy_onpolicy
activation                                          rlb_fused_fixed_strong_ffn
base_lr                                             3e-4
min_lr                                              3e-5
warmup_steps                                        200
global_schedule                                     shared warmup/cosine for every compared row
backbone_optimizer                                  AdamW
backbone_beta2                                      0.999
matrix_policy_beta2                                 0.999
matrix_policy_adam_lr_scale                         3.00
matrix_policy_adam_lr_scale_final                   null
matrix_policy_adam_role_strength                    1.20
matrix_policy_adam_min_lr_scale                     0.40
matrix_policy_adam_max_lr_scale                     4.00
matrix_policy_input_depth_gain                     -0.50
matrix_policy_output_depth_gain                     1.00
matrix_policy_muon_strength                         0.75
matrix_policy_muon_lr_scale                         1.00
matrix_policy_start                                 0.02
matrix_policy_end                                   0.12
matrix_policy_decay_start                           0.20
matrix_policy_decay_end                             0.36
matrix_policy_final_muon                            0.00
matrix_policy_muon_reset_adam_state                 false
matrix_policy_function_coeff                        false
matrix_policy_group_gain_strength                   0.00
matrix_policy_group_pressure_strength               0.00
matrix_policy_group_activity_damping                0.00
role_specific_beta2_finals                          null by default
transport_strength                                  0.00
adaptive_coeff_strength                             0.00
rlb_gauge_strength                                  0.50
rlb_gauge_start                                     0.03
rlb_gauge_end                                       0.35
rlb_gauge_every                                     5
```

The global warmup/cosine schedule is shared across rows. MatrixPolicy is not a global LR scheduler; it changes the per-role update rule on RLB matrices.

## Verified WikiText Result

| row | final loss | final PPL |
| --- | ---: | ---: |
| RLB MatrixPolicy-Muon | 3.476232 | 32.34 |
| RLB Smooth-MatrixPolicy | 3.493210 | 32.89 |
| SiLU/SwiGLU+AdamW beta2=0.999 | 3.549346 | 34.79 |
| RLB+AdamW beta2=0.999 | 3.550018 | 34.81 |
| RLB+AdamW | 3.617501 | 37.24 |
| SiLU/SwiGLU+AdamW | 3.621982 | 37.41 |
| SiLU/SwiGLU+Muon | 3.644921 | 38.28 |
| RLB+Muon | 3.657877 | 38.78 |

The current verified gap versus `SiLU/SwiGLU+AdamW beta2=0.999` is `0.0731` loss and `2.45` PPL. That is a real result, but it is not the final target.

## After The Fair Job Finishes

Run:

```bash
.venv-cu128/bin/python experiments/scripts/summarize_synthetic_fair_full_20260529.py
```

Then update all README files and the compact result artifact. Keep raw JSONL and Slurm logs uncommitted.
