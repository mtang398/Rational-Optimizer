# Optimizer Design

This folder contains optimizer components for the no-GLU RLB FFN. The real claim target is to beat SiLU/SwiGLU-like AdamW controls with an RLB-specific optimizer under the same global LR schedule.

## Active Optimizer

```text
rational_matrix_policy_onpolicy
RationalMatrixPolicyOptimizer
```

It separates RLB `W_in` and `W_out` matrices from ordinary AdamW, applies a short early Muon phase only to those matrices, then switches back to MatrixPolicy AdamW. The outer optimizer applies exact RLB gauge balance after child optimizer steps.

## Exact Defaults

```text
optimizer                                           rational_matrix_policy_onpolicy
activation                                          rlb_fused_fixed_strong_ffn
optimizer_lr                                        3e-4
optimizer_min_lr                                    3e-5
warmup_steps                                        200
lr_schedule                                         cosine
rational_matrix_policy_backbone_optimizer           adamw
rational_matrix_policy_backbone_beta2               0.999
rational_matrix_policy_beta2                        0.999
rational_matrix_policy_adam_lr_scale                3.00
rational_matrix_policy_adam_lr_scale_final          null
rational_matrix_policy_adam_decay_start             1.10
rational_matrix_policy_adam_decay_end               1.10
rational_matrix_policy_adam_decay_depth_shift       0.00
rational_matrix_policy_adam_beta2_final             null
rational_matrix_policy_adam_beta2_decay_start       1.10
rational_matrix_policy_adam_beta2_decay_end         1.10
rational_matrix_policy_adam_beta2_decay_depth_shift 0.00
rational_matrix_policy_adam_role_strength           1.20
rational_matrix_policy_adam_stat_strength           0.00
rational_matrix_policy_adam_pressure_balance        0.00
rational_matrix_policy_adam_stat_start              0.00
rational_matrix_policy_adam_stat_end                0.00
rational_matrix_policy_adam_min_lr_scale            0.40
rational_matrix_policy_adam_max_lr_scale            4.00
rational_matrix_policy_input_depth_gain            -0.50
rational_matrix_policy_output_depth_gain            1.00
rational_matrix_policy_muon_strength                0.75
rational_matrix_policy_muon_lr_scale                1.00
rational_matrix_policy_max_muon                     0.75
rational_matrix_policy_min_muon                     0.00
rational_matrix_policy_final_muon                   0.00
rational_matrix_policy_start                        0.02
rational_matrix_policy_end                          0.12
rational_matrix_policy_decay_start                  0.20
rational_matrix_policy_decay_end                    0.36
rational_matrix_policy_muon_decay_depth_shift       0.00
rational_matrix_policy_muon_input_decay_shift       0.00
rational_matrix_policy_muon_output_decay_shift      0.00
rational_matrix_policy_muon_reset_adam_state        false
rational_matrix_policy_pressure_weight              0.30
rational_matrix_policy_activity_weight              0.65
rational_matrix_policy_weight_decay_scale           1.00
rational_matrix_policy_function_coeff               false
rational_matrix_policy_group_gain_strength          0.00
rational_matrix_policy_group_pressure_strength      0.00
rational_matrix_policy_group_activity_damping       0.00
rational_transport_strength                         0.00
rational_transport_matrix_strength                  0.00
rational_transport_quotient_strength                0.00
rational_adaptive_coeff_strength                    0.00
rlb_gauge_strength                                  0.50
rlb_gauge_start                                     0.03
rlb_gauge_end                                       0.35
rlb_gauge_every                                     5
```

## Why It Is RLB-Specific

RLB computes:

```text
v = x W_in
s_g = sqrt(mean(v_g^2) + eps)
u_g = v_g / s_g
h_g = s_g R_g(u_g)
y = h W_out
```

The optimizer uses that structure directly:

| parameter set | optimizer behavior |
| --- | --- |
| Non-RLB backbone weights | AdamW, same global LR schedule, beta2=0.999 in this branch |
| Norms, biases, tied embeddings | AdamW no-decay group |
| RLB `W_in` and `W_out` matrices | MatrixPolicy AdamW plus early Muon on the same tensors |
| Rational coefficients | AdamW by default |

For an RLB matrix group at layer depth `d in [0, 1]`:

```text
role_factor = clamp(1 + gain(role) * (d - 0.5), 0.55, 1.40)
gain(in)    = -0.50
gain(out)   =  1.00

adam_scale = clamp(3.00 * (1 + 1.20 * (role_factor - 1)), 0.40, 4.00)
adam_lr    = global_lr * adam_scale * (1 - muon_fraction)
```

The early Muon fraction is:

```text
on_phase      = smoothstep(0.02, 0.12, progress)
off_phase     = smoothstep(0.20, 0.36, progress)
muon_fraction = clamp(0.75 * on_phase * (1 - off_phase) * role_factor * on_policy_stat_factor, 0.0, 0.75)
muon_lr       = global_lr * muon_fraction
```

For 3051 steps, Muon ramps in around steps 61-366 and fades out around steps 610-1098.

## Verified Result

| row | final loss | final PPL |
| --- | ---: | ---: |
| RLB MatrixPolicy-Muon | 3.476232 | 32.34 |
| RLB Smooth-MatrixPolicy | 3.493210 | 32.89 |
| SiLU+AdamW beta2=0.999 | 3.549346 | 34.79 |
| RLB+AdamW beta2=0.999 | 3.550018 | 34.81 |
| RLB+AdamW | 3.617501 | 37.24 |
| SiLU+AdamW | 3.621982 | 37.41 |
| SiLU+Muon | 3.644921 | 38.28 |
| RLB+Muon | 3.657877 | 38.78 |

## Latest Design Lessons

What still works:

```text
- sustained RLB matrix policy
- role/depth asymmetry between W_in and W_out
- beta2=0.999 for MatrixPolicy and backbone AdamW
- short early Muon only on RLB matrices
- keeping Adam moments through the Muon switch
- exact post-step RLB gauge balance
```

What did not widen the gap:

```text
- plain full-model Muon on SiLU or RLB
- late Muon tails after the early window
- beta2 tail from 0.999 to 0.995
- per-rational-group gradient policy
- scalar on-policy stat gates
- stat-gate plus group-gradient policy
- synthetic arithmetic transfer as a final-loss benchmark
```

The current readout is conservative: the existing MatrixPolicy-Muon remains best; the new variants added complexity without a material loss gap.
