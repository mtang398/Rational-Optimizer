# Optimizer Design

This folder contains optimizer components for RLB, the no-GLU Rational Local Basis FFN. Optimizer claims must beat `SiLU/SwiGLU+AdamW` and `RLB+AdamW` under the same global LR schedule.

## Current Active Optimizer

`RationalMatrixPolicyOptimizer` is the active optimizer component behind:

```text
rational_matrix_policy_onpolicy
```

The current default is Smooth-MatrixPolicy. It separates RLB `W_in` and `W_out` matrices from ordinary AdamW, applies a layer/side-specific matrix policy, and then uses the existing on-policy gauge wrapper for exact RLB balancing. The MatrixPolicy branch now uses smoother AdamW second moments by default.

Verified defaults:

```text
adam_lr_scale                         3.00
adam_role_strength                    1.20
input_depth_gain                     -0.50
output_depth_gain                     1.00
adam_min_lr_scale                     0.40
adam_max_lr_scale                     4.00
rational_matrix_policy_beta2          0.999
rational_matrix_policy_backbone_beta2 0.999
muon_strength                         0.00
transport_strength                    0.00
```

The policy is rational-specific because it uses the RLB decomposition:

```text
v = x W_in
u_g = v_g / rms(v_g)
h_g = rms(v_g) R_g(u_g)
y = h W_out
```

`W_in` and `W_out` receive different layer/depth scaling because they see different rational paths: derivative/domain formation on the input side and feature-amplitude composition on the output side.

## Verified Result

Seed-1337 full 3051-step WikiText-103, same global LR schedule:

| row | final loss | final PPL |
| --- | ---: | ---: |
| RLB Smooth-MatrixPolicy | 3.493210 | 32.89 |
| SiLU/SwiGLU + AdamW beta2=0.999 | 3.549346 | 34.79 |
| RLB + AdamW beta2=0.999 | 3.550018 | 34.81 |
| RLB + AdamW | 3.617501 | 37.24 |
| SiLU/SwiGLU + AdamW | 3.621982 | 37.41 |

This is a same-LR optimizer win, not an LR-schedule win. The tuned-control gap is still below the final target.

## Components Kept

| component | role |
| --- | --- |
| `FunctionSpaceRationalOptimizer` | coefficient/function-space probe utility |
| `RationalOnPolicyBalanceOptimizer` | exact RLB group-gauge rebalance |
| `RationalQuotientOnPolicyOptimizer` | removes pure gauge-gradient direction |
| `RationalJacobianOnPolicyOptimizer` | stable earlier rational-specific comparator |
| `RationalTransportOnPolicyOptimizer` | exact rational-curve amplitude transport, off by default |
| `RationalMatrixPolicyOptimizer` | current best active optimizer |
| `FactoredAdamW` and switching wrappers | retained as negative/ablation tools |

## Lessons

What worked:

```text
- sustained RLB matrix policy
- strong layer/side asymmetry
- smoother AdamW second moments for the MatrixPolicy branch
- default AdamW-style updates for non-RLB small parameters
- exact RLB gauge balancing after the step
```

What did not work:

```text
- pushing MatrixPolicy scale beyond the Y setting
- simple on-policy group gain/pressure equalization
- stronger gauge balance late in training
- Muon on RLB matrices or the non-RLB backbone
- coefficient freezing or function-space coefficient switching
- rational amplitude transport as a durable default
```

## Next Design Direction

The next useful optimizer should keep Smooth-MatrixPolicy as the base and make the smooth moment policy more RLB-specific:

```text
1. split beta2 by RLB matrix side and layer depth
2. use live pressure to lengthen or shorten RLB matrix memory
3. keep global LR fixed while changing only optimizer state dynamics
4. compare against beta2-tuned AdamW controls, not only default AdamW
```
