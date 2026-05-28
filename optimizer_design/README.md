# Optimizer Design

This folder contains optimizer components for RLB, the no-GLU Rational Local Basis FFN. Optimizer claims must beat `SiLU+AdamW` and `RLB+AdamW` under the same global LR schedule.

## Active Optimizer

`RationalMatrixPolicyOptimizer` is the active component behind:

```text
rational_matrix_policy_onpolicy
```

The current default is MatrixPolicy-Muon. It separates RLB `W_in` and `W_out` matrices from ordinary AdamW, applies a short early Muon phase only to those matrices, then switches back to MatrixPolicy AdamW. The wrapper still applies exact RLB gauge balance after the step.

Verified default settings:

```text
adam_lr_scale                         3.00
adam_role_strength                    1.20
input_depth_gain                     -0.50
output_depth_gain                     1.00
adam_min_lr_scale                     0.40
adam_max_lr_scale                     4.00
rational_matrix_policy_beta2          0.999
rational_matrix_policy_backbone_beta2 0.999
muon_strength                         0.75
muon_lr_scale                         1.00
muon window                           start 0.02, end 0.12, decay 0.20-0.36
muon_reset_adam_state                 false
transport_strength                    0.00
```

## Why It Is RLB-Specific

RLB computes:

```text
v = x W_in
u_g = v_g / rms(v_g)
h_g = rms(v_g) R_g(u_g)
y = h W_out
```

`W_in` controls the rational input domain and derivatives. `W_out` composes rational features back into the model stream. MatrixPolicy uses different role/depth scaling for these two sides, and the outer optimizer uses the exact RLB gauge to rebalance equivalent matrix representatives.

The early Muon switch is also restricted to RLB matrices. The non-RLB backbone is not moved to Muon because that was tested and was worse.

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
| `FunctionSpaceRationalOptimizer` | coefficient/function-space probe utility |
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
- on-policy Muon damping
- layer/role Muon timing shifts
- coefficient freezing or function-space coefficient switching
- rational amplitude transport as a durable default
```
