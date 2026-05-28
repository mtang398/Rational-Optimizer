# 2026-05-27 High-LR Optimizer Follow-Up

Seed 1337, 100M tokens, WikiText-103. The high-LR rows use `--lr 5e-4 --min-lr 5e-5`; fixed-LR rows use the original `3e-4` schedule. Validation curves use all eval points and set the x-axis left edge to step 1. The train-loss plot includes the actual step-1 train point.

| row | final loss | final PPL | gap vs fixed SiLU+AdamW | gap vs high-LR SiLU+AdamW | gap vs high-LR RLB+AdamW |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed LR AdamW + SiLU/SwiGLU | 3.621982 | 37.412 | +0.000000 | +0.165357 | +0.166190 |
| Fixed LR AdamW + RLB h3072 | 3.617501 | 37.244 | -0.004480 | +0.160876 | +0.161709 |
| Fixed LR RLB + rational_jacobian_onpolicy | 3.614862 | 37.146 | -0.007120 | +0.158237 | +0.159070 |
| High LR AdamW + SiLU/SwiGLU | 3.456625 | 31.710 | -0.165357 | +0.000000 | +0.000833 |
| High LR AdamW + RLB h3072 | 3.455792 | 31.683 | -0.166190 | -0.000833 | +0.000000 |
| High LR RLB + rational_jacobian_onpolicy | 3.459508 | 31.801 | -0.162473 | +0.002883 | +0.003716 |


## Interpretation

The large apparent win came from the higher learning-rate schedule, not from the new rational-specific optimizer. High-LR `RLB + rational_jacobian_onpolicy` reaches `3.459508` loss and `31.801` PPL, which is a big gap versus the original fixed-LR `SiLU+AdamW` row (`-0.162473` loss, `-5.610` PPL). But the fair high-LR controls erase that conclusion: high-LR `SiLU+AdamW` is `3.456625`, and high-LR `RLB+AdamW` is best at `3.455792` / `31.683`.

The honest ranking for this seed is high-LR `RLB+AdamW`, then high-LR `SiLU+AdamW`, then high-LR `RLB+rational_jacobian_onpolicy`. The high-LR RLB activation advantage over high-LR SiLU is only `-0.000833` loss and `-0.026` PPL. That is far below the target gap and should not be treated as a robust optimizer result without more seeds.

Negative probes sharpened the diagnosis: ASAM did not improve the fixed-LR rows; factored AdamW second moments badly hurt early learning; and the aggressive layerwise coefficient switch was worse by step 1000. The useful signal remains conservative RLB matrix/gauge geometry. The damaging signal is scheduled or factored movement that changes coefficient behavior or matrix curvature before the model has earned it on-policy.
