# RationalOPT

Read [READ_FIRST.md](READ_FIRST.md) before running jobs.

This repo studies rational activations and optimizers in a controlled WikiText-103 causal language-modeling benchmark. The current benchmark is a 123M-parameter LLaMA-style decoder-only Transformer. The active question is optimizer-specific: design an optimizer that uses the structure of the no-GLU Rational Local Basis FFN, then compare it fairly against RLB with AdamW/Muon and SiLU/SwiGLU with AdamW/Muon.

## Accepted Comparison

New optimizer experiments use this grid:

```text
SiLU/SwiGLU + AdamW
SiLU/SwiGLU + Muon
RLB + AdamW
RLB + Muon
RLB + rational_onpolicy_balance
RLB + rational_quotient_onpolicy
RLB + rational_jacobian_onpolicy
RLB + rational_quotient_jacobian_onpolicy   prototype
RLB + rational_adaptive_metric_onpolicy     prototype
```

Rational-specific optimizers are applied only to RLB. The standard optimizer names are `adamw` and `muon`.

## RLB FFN

RLB is a no-GLU feed-forward layer. It has one expansion projection, grouped rational feature generation, and one output projection:

```text
v = x W_in
s_g = sqrt(mean(v_g^2) + eps)
u_g = v_g / s_g
h_g = s_g R_g(u_g)
y = h W_out
```

There is no gate projection, no up projection, and no SiLU inside the RLB FFN. The current activation variants are:

```text
rlb_fused_fixed_strong_ffn        h = 3072
rlb_fused_fixed_strong_h2880_ffn  h = 2880
```

## Why The Optimizer Is RLB-Specific

RLB has an exact positive group gauge:

```text
W_in,g  <- c W_in,g
W_out,g <- W_out,g / c
```

For positive `c`, the layer function is unchanged. This gives an optimizer a real rational-specific degree of freedom: it can choose the scale representative of each group without changing the model function.

The active optimizer path is:

```text
rational_onpolicy_balance
rational_quotient_onpolicy
rational_jacobian_onpolicy
rational_quotient_jacobian_onpolicy
rational_adaptive_metric_onpolicy
```

`rational_onpolicy_balance` uses live gradient pressure, rational curve activity, and layer depth to apply a function-preserving group-scale correction after each child optimizer step.

`rational_quotient_onpolicy` removes pure gauge motion from the RLB matrix gradients before the child optimizer step, then applies the same on-policy balance transform.

`rational_jacobian_onpolicy` keeps the on-policy balance transform and adds a low-overhead curve-aware preconditioner. It scales each group of `W_in` by the inverse relative rational derivative gain and each group of `W_out` by the inverse relative rational output gain. This directly uses the fact that RLB matrix updates pass through the current learned rational functions.

`rational_quotient_jacobian_onpolicy` is a prototype that combines quotient projection with the Jacobian preconditioner. It is useful as an ablation but did not beat the verified Jacobian row in the seed-1337 probe.

`rational_adaptive_metric_onpolicy` is a prototype that can use live on-policy RLB activation statistics. Its default keeps coefficient Gram conditioning off because that over-conditioned the small rational tensors in probes.

## Current Full Result

Completed full sweep:

```text
run name: rlb_optimizer_empirical_ngram_full
job ids:  763059 + 813929 + 821187
seeds:    1337, 2024, 31415
budget:   100M training tokens per row
```

Aggregate losses from `experiments/results/rlb_optimizer_empirical_ngram_full/aggregate.csv`:

```text
AdamW + SiLU/SwiGLU                         3.610129  PPL 36.973  sec/step 0.188997
AdamW + RLB h3072                           3.606629  PPL 36.845  sec/step 0.205268
RLB h3072 + rational_onpolicy_balance       3.606226  PPL 36.831  sec/step 0.209027
RLB h3072 + rational_quotient_onpolicy      3.606664  PPL 36.847  sec/step 0.205176
RLB h3072 + rational_jacobian_onpolicy      3.605394  PPL 36.800  sec/step 0.204885
```

The best measured row is `rational_jacobian_onpolicy + rlb_fused_fixed_strong_ffn`, with mean loss gap `-0.004736` versus AdamW + SiLU/SwiGLU and mean gap `-0.001236` versus AdamW on the same RLB activation.

## 2026-05-27 Optimizer Probes

Additional RLB-specific optimizers were implemented and probed on seed 1337 before deciding whether to launch a full multi-seed sweep:

```text
rational_adaptive_metric_onpolicy h3072   3.615887  PPL 37.184
rational_adaptive_metric_onpolicy h2880   3.615114  PPL 37.156
rational_quotient_jacobian_onpolicy h3072 3.615571  PPL 37.173
```

The seed-1337 incumbents are `3.614862` for `rational_jacobian_onpolicy + h3072` and `3.614475` for `rational_quotient_onpolicy + h2880`. The new probes beat AdamW/SILU on that seed, but they did not beat the existing rational-specific rows, so the full three-seed recommendation remains unchanged. The coefficient-Gram probe and the h2880 quotient-Jacobian row were stopped early after underperforming checkpoints.

## Layout

```text
activation/         rational activation package and CUDA extension
training/           WikiText-103 training, sweep, and aggregation scripts
optimizer_design/   RLB-specific optimizer components
experiments/        cache, active runs, logs, and aggregate outputs
```

## Commands

Run the active full comparison set:

```bash
env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_P2P_DISABLE=1 \
  RUN_NAME=rlb_optimizer_empirical_ngram_full \
  STEPS=0 \
  SEEDS="1337 2024 31415" \
  OPTIMIZERS="adamw muon rational_onpolicy_balance rational_quotient_onpolicy rational_jacobian_onpolicy rational_quotient_jacobian_onpolicy rational_adaptive_metric_onpolicy" \
  ACTIVATIONS="silu rlb_fused_fixed_strong_ffn rlb_fused_fixed_strong_h2880_ffn" \
  EVAL_INTERVAL=250 EVAL_BATCHES=20 LOG_INTERVAL=100 \
  sbatch --time=08:00:00 --gres=gpu:nvidia_rtx_6000_ada_generation:4 training/run_wikitext103_optimizer_sweep.sbatch
```

Aggregate completed jobs:

```bash
.venv-cu128/bin/python training/aggregate_wikitext103_multiseed.py \
  --run-dir experiments/runs/wikitext103/rlb_optimizer_empirical_ngram_full \
  --out-dir experiments/results/rlb_optimizer_empirical_ngram_full \
  --baseline silu \
  --baseline-optimizer adamw \
  --classic-optimizer adamw \
  --job-id 763059+813929+821187 \
  --log-path experiments/runs/logs/ract-wt103-opt-821187.out
```
