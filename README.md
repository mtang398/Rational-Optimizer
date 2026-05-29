# RationalOPT

RationalOPT is an optimizer-design repo for the no-GLU Rational Local Basis FFN (RLB). The target is not an LR trick: the target is an RLB-specific optimizer that beats SiLU/SwiGLU+AdamW and RLB+AdamW under the same global learning-rate schedule.

Jacobian, quotient, transport, and coefficient-function optimizers are ablations. They are not the baseline. The baseline that matters is the standard SiLU/SwiGLU FFN with AdamW or Muon, plus RLB with the same generic optimizers.

## Current Status

State on May 29, 2026:

```text
verified best on WikiText-103: RLB MatrixPolicy-Muon
large requested target:       not reached yet
fair synthetic rerun:         running as Slurm job 937608
fair rerun script:            experiments/scripts/run_synthetic_fair_full_20260529.sh
GPU policy:                   one 4x A6000 job, 24h walltime
```

Old partial synthetic/code, symbolic, and reasoning_mix outputs are not claim-grade and should not be used as the final story. They were superseded by the clean full rerun above.

## Verified WikiText-103 Result

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

Best verified row: `3.476232` validation loss and `32.34` PPL at step 3051, seed 1337. The gap versus the tuned `SiLU/SwiGLU+AdamW beta2=0.999` control is `0.0731` loss and `2.45` PPL. That is useful, but it is still below the requested `0.2-0.3` loss gap.

## Current Optimizer

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

This optimizer applies the same base LR schedule as every control. The changed part is the update rule on RLB `W_in` and `W_out` matrices.

## How RLB Is Used

RLB computes grouped rational features:

```text
v = x W_in
s_g = sqrt(mean(v_g^2) + eps)
u_g = v_g / s_g
h_g = s_g R_g(u_g)
y = h W_out
```

`W_in` sets rational input domains and derivative exposure. `W_out` composes rational features back into the model stream. MatrixPolicy treats those roles differently across depth, runs a short early Muon phase only on those matrices, then switches the same matrices back to MatrixPolicy AdamW. The outer wrapper applies exact RLB gauge balance after optimizer steps.

## Fair Synthetic Rerun

| task | compared rows |
| --- | --- |
| `synthetic/code` | SiLU/SwiGLU+AdamW, RLB+AdamW, SiLU/SwiGLU+Muon, RLB+Muon, RLB MatrixPolicy, RLB MatrixPolicy group-stat |
| `synthetic/symbolic` | same rows |
| `synthetic/reasoning_mix` | same rows |

Every row uses the same seed, model shape, token budget, base LR, warmup/cosine schedule, batch size, gradient accumulation, eval cadence, and A6000 fallback. The exploratory `group-stat` row is a rational-specific variant that uses live RLB group statistics to reweight matrix gradients; it is not claimed as an improvement until the fair rerun finishes.

After job `937608` finishes, summarize it with:

```bash
.venv-cu128/bin/python experiments/scripts/summarize_synthetic_fair_full_20260529.py
```

The summarizer writes compact CSV/Markdown/PNG artifacts under `experiments/results/synthetic_fair_full_2026_05_29/` and leaves raw JSONL logs under `experiments/runs/`.

## Plots From The Verified Result

![Same-LR validation loss](experiments/results/rlb_matrix_policy_muon_switch_2026_05_28/same_lr_validation_loss.png)

![Same-LR validation PPL](experiments/results/rlb_matrix_policy_muon_switch_2026_05_28/same_lr_validation_ppl.png)

![Same-LR training loss from step 1](experiments/results/rlb_matrix_policy_muon_switch_2026_05_28/same_lr_training_loss_from_step1.png)

## Layout

```text
activation/         RLB activation package and CUDA extension
training/           WikiText-103 and synthetic LLM training entrypoints
optimizer_design/   RLB-specific optimizer components
experiments/        launchers, compact result artifacts, local raw runs
```
