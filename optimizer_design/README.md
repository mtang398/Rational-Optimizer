# Optimizer Design

This folder contains optimizer components for RLB, the no-GLU Rational Local Basis FFN. Optimizer claims must beat `SiLU+AdamW` and `RLB+AdamW` under the same global LR schedule.

## Active Optimizer

The active optimizer is:

```text
rational_matrix_policy_onpolicy
```

The active component inside it is:

```text
RationalMatrixPolicyOptimizer
```

It separates RLB `W_in` and `W_out` matrices from ordinary AdamW, applies a short early Muon phase only to those matrices, then switches back to MatrixPolicy AdamW. The outer on-policy wrapper applies exact RLB gauge balance after child optimizer steps.

## Exact Defaults

The names below match the JSONL config keys where possible. `optimizer_lr` and `optimizer_min_lr` are the values passed by `--lr` and `--min-lr`.

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
rational_matrix_policy_adam_role_strength           1.20
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

## Parameter Groups

| parameter set | optimizer behavior |
| --- | --- |
| Non-RLB backbone weights | AdamW, same global LR schedule, beta2=0.999 in this branch |
| Norms, biases, tied embeddings | AdamW no-decay group |
| RLB `W_in` and `W_out` matrices | `RationalMatrixPolicyOptimizer`, with AdamW plus early Muon on the same tensors |
| Rational coefficients | ordinary AdamW by default; `FunctionSpaceRationalOptimizer` is off |

## Why It Is RLB-Specific

RLB computes:

```text
v = x W_in
s_g = sqrt(mean(v_g^2) + eps)
u_g = v_g / s_g
h_g = s_g R_g(u_g)
y = h W_out
```

`W_in` controls the rational input domain and derivatives. `W_out` composes rational features back into the model stream. MatrixPolicy uses different role/depth scaling for these two sides, and the outer optimizer uses the exact RLB gauge to rebalance equivalent matrix representatives.

## Formula

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
muon_fraction = clamp(
  0.75 * on_phase * (1 - off_phase)
  * role_factor
  * on_policy_stat_factor,
  0.0,
  0.75
)
muon_lr = global_lr * 1.00 * muon_fraction
```

For the 3051-step run, Muon ramps in around steps 61-366 and fades out around steps 610-1098. After that, RLB matrices are updated by MatrixPolicy AdamW only.

## Step Order

```text
1. backward pass computes gradients
2. outer on-policy wrapper updates live RLB statistics
3. transport, coefficient-function optimizer, and group-gradient policy are off by default
4. ordinary AdamW steps the non-RLB backbone, no-decay group, and rational coefficients
5. MatrixPolicy handles RLB W_in/W_out matrices:
   - compute role/depth AdamW scale
   - compute early Muon fraction
   - step AdamW with lr = global_lr * adam_scale * (1 - muon_fraction)
   - step Muon with lr = global_lr * muon_lr_scale * muon_fraction
   - restore scheduler-provided base group lrs
6. outer wrapper applies exact RLB gauge balance
```

## Verified Result

Seed-1337 full 3051-step WikiText-103, same global LR schedule:

| row | final loss | final PPL |
| --- | ---: | ---: |
| RLB MatrixPolicy-Muon | 3.476232 | 32.34 |
| RLB Smooth-MatrixPolicy | 3.493210 | 32.89 |
| SiLU+AdamW beta2=0.999 | 3.549346 | 34.79 |
| RLB+AdamW beta2=0.999 | 3.550018 | 34.81 |
| RLB+AdamW | 3.617501 | 37.24 |
| SiLU+AdamW | 3.621982 | 37.41 |

This is a same-LR optimizer win. It is not yet the desired 0.2-0.3 final loss gap.

## Kept Components

| component | role |
| --- | --- |
| `RationalMatrixPolicyOptimizer` | current best active optimizer |
| `RationalOnPolicyBalanceOptimizer` | exact RLB group-gauge rebalance |
| `FunctionSpaceRationalOptimizer` | coefficient/function-space probe utility, off by default |
| `RationalTransportOnPolicyOptimizer` | rational-curve amplitude transport, off by default |
| quotient/Jacobian/switching/factored optimizers | retained as ablation tools, not baselines |

## Lessons

What worked:

```text
- sustained RLB matrix policy
- strong layer/side asymmetry
- beta2=0.999 for the MatrixPolicy branch
- short early Muon on RLB matrices only
- switching back to MatrixPolicy AdamW for late training
- keeping Adam moments through the switch
```

What did not work:

```text
- Muon on the non-RLB backbone
- stronger or weaker Muon settings than the promoted window
- resetting Adam state after Muon
- earlier global Muon shutoff
- stronger/extra on-policy Muon damping
- layer/role Muon timing shifts
- coefficient freezing or function-space coefficient switching
- rational amplitude transport as a durable default
```
