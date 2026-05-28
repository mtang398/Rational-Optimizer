# Optimizer Design

This folder contains optimizer components for RLB, the no-GLU Rational Local Basis FFN. The design rule is strict: optimizer claims must beat `SiLU/SwiGLU+AdamW` and `RLB+AdamW` under the same global LR schedule.

## Current Active Optimizer

`RationalMatrixPolicyOptimizer` is the active optimizer component behind:

```text
rational_matrix_policy_onpolicy
```

It separates the RLB FFN matrices from the rest of the model. Ordinary non-RLB parameters remain on AdamW. The RLB `W_in` and `W_out` matrices get a layer/side-specific matrix policy, then the existing on-policy gauge wrapper rebalances the RLB gauge after each step.

Verified same-LR Probe Y defaults:

```text
adam_lr_scale       = 3.00
adam_role_strength  = 1.20
input_depth_gain    = -0.50
output_depth_gain   = 1.00
adam_min_lr_scale   = 0.40
adam_max_lr_scale   = 4.00
muon_strength       = 0.00
transport_matrix    = 0.00
```

The policy is rational-specific because it uses the RLB decomposition:

```text
W_in gradient path:   derivative of R_g(u)
W_out gradient path:  feature amplitude R_g(u)
shallow layers:       input/domain formation matters more
deep layers:          output feature composition matters more
```

## Verified Result

Seed-1337 full 3051-step WikiText-103, same global LR/schedule:

| row | final loss | final PPL |
| --- | ---: | ---: |
| RLB MatrixPolicy-Y | 3.548665 | 34.77 |
| RLB + Jacobian | 3.614862 | 37.15 |
| RLB + AdamW | 3.617501 | 37.24 |
| SiLU/SwiGLU + AdamW | 3.621982 | 37.41 |

This is a real optimizer win, not an LR-schedule win. It is still below the desired `0.2-0.3` loss gap.

## Components Kept

| component | role |
| --- | --- |
| `FunctionSpaceRationalOptimizer` | coefficient/function-space utility, useful for probes |
| `RationalOnPolicyBalanceOptimizer` | exact RLB group-gauge rebalance |
| `RationalQuotientOnPolicyOptimizer` | removes pure gauge-gradient direction |
| `RationalJacobianOnPolicyOptimizer` | stable earlier rational-specific comparator |
| `RationalTransportOnPolicyOptimizer` | wrapper used for gauge/stat collection around matrix policy |
| `RationalMatrixPolicyOptimizer` | current best active optimizer |
| `FactoredAdamW` and switching wrappers | retained as negative/ablation tools |

The failed manual `RationalMatrixAdamWOptimizer` was removed from active code because it did not beat the matrix-policy path.

## Lessons

What worked:

```text
- sustained RLB matrix policy
- strong layer/side asymmetry
- shallow W_in emphasis and deep W_out emphasis
- default AdamW for non-RLB parameters
- exact RLB gauge balancing after the step
```

What did not work:

```text
- Muon on RLB matrices
- scheduled coefficient phase switches
- functional-trust coefficient updates
- transport-matrix preconditioning as the main lever
- early pressure/activity gating
- factored matrix moments
```

## Next Design Direction

The next useful optimizer should keep MatrixPolicy-Y as the base and make the controller more on-policy without changing global LR:

```text
1. collect per-layer/per-group update pressure
2. predict matrix-only functional improvement
3. predict coefficient functional improvement on actual u samples
4. apply coefficient updates only when they beat matrix-only updates safely
5. otherwise spend the step on layer/side matrix policy and gauge rebalance
```

Do not treat high-LR runs as evidence for this optimizer until the same-LR loss gap is much larger.
