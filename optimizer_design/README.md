# Optimizer Design

This folder contains optimizer components for the no-GLU RLB FFN. The active research question is whether an optimizer that uses RLB structure can beat SiLU/SwiGLU+AdamW and RLB+AdamW under the same global LR schedule.

## Active Optimizer

```text
training name:  rational_matrix_policy_onpolicy
implementation: RationalMatrixPolicyOptimizer
activation:     rlb_fused_fixed_strong_ffn
```

The optimizer separates RLB matrices from the ordinary backbone. Non-RLB weights use AdamW. RLB `W_in` and `W_out` matrices use MatrixPolicy AdamW plus a short early Muon phase on those same matrices. The outer wrapper then applies exact RLB gauge balance.

## Exact Defaults

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

Role-specific beta2 finals are implemented for experiments, but default to `null`; the last role-beta2 probe was not an improvement.

## Why It Is RLB-Specific

RLB has a positive gauge:

```text
W_in,g  <- c W_in,g
W_out,g <- W_out,g / c
```

The function can stay almost unchanged while the matrix conditioning changes. The optimizer uses that structure in three places:

| RLB structure | optimizer use |
| --- | --- |
| `W_in` | controls rational input domains and derivative exposure |
| `W_out` | controls rational feature composition back into the stream |
| layer depth | later `W_out` receives more matrix-policy pressure, later `W_in` receives less |
| group stats | optional on-policy diagnostics and exploratory gradient scaling |
| gauge balance | exact post-step rebalance to keep basis scales controlled |

For an RLB matrix at normalized depth `d`:

```text
role_factor = clamp(1 + gain(role) * (d - 0.5), 0.55, 1.40)
gain(in)    = -0.50
gain(out)   =  1.00
adam_lr     = global_lr * clamp(3.00 * (1 + 1.20 * (role_factor - 1)), 0.40, 4.00)
```

Muon is only an early matrix-local switch:

```text
on_phase      = smoothstep(0.02, 0.12, progress)
off_phase     = smoothstep(0.20, 0.36, progress)
muon_fraction = clamp(0.75 * on_phase * (1 - off_phase) * role_factor * stat_factor, 0.0, 0.75)
```

For a 3051-step run, that means Muon ramps in around steps 61-366 and fades out around steps 610-1098. It is not left on late.

## Verified Result

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

## Recent Design Lessons

What is still useful:

```text
short early Muon only on RLB matrices
role/depth asymmetry between W_in and W_out
MatrixPolicy AdamW after the early switch
beta2=0.999 on MatrixPolicy and backbone AdamW
keeping Adam state through the Muon switch
exact post-step RLB gauge balance
```

What did not widen the gap in probes:

```text
plain full-model Muon as the main optimizer
late Muon tails
lowering beta2 late from 0.999 to 0.995
function-space coefficient optimizer variants
role-beta2 asymmetry
stronger or weaker role/depth scaling than the current default
```

The fair rerun adds one exploratory `group-stat` MatrixPolicy row. That row is rational-specific, but it is not a claimed improvement until the complete same-budget run finishes.
