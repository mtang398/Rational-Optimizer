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
RLB + rational_transport_onpolicy           tested prototype
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
rational_transport_onpolicy
```

`rational_onpolicy_balance` uses live gradient pressure, rational curve activity, and layer depth to apply a function-preserving group-scale correction after each child optimizer step.

`rational_quotient_onpolicy` removes pure gauge motion from the RLB matrix gradients before the child optimizer step, then applies the same on-policy balance transform.

`rational_jacobian_onpolicy` keeps the on-policy balance transform and adds a low-overhead curve-aware preconditioner. It scales each group of `W_in` by the inverse relative rational derivative gain and each group of `W_out` by the inverse relative rational output gain. This directly uses the fact that RLB matrix updates pass through the current learned rational functions.

`rational_quotient_jacobian_onpolicy` is a prototype that combines quotient projection with the Jacobian preconditioner. It is useful as an ablation but did not beat the verified Jacobian row in the seed-1337 probe.

`rational_adaptive_metric_onpolicy` is a prototype that can use live on-policy RLB activation statistics. Its default keeps coefficient Gram conditioning off because that over-conditioned the small rational tensors in probes.

`rational_transport_onpolicy` is a tested prototype. It adds optional rational-only amplitude transport, optional pressure preconditioning, and an on-policy coefficient selector. The selector can switch from aggressive early rational coefficient updates to safer late updates by layer using live coefficient-vs-matrix gradient activity. The validated default is conservative: baseline coefficient dynamics plus matrix preconditioning, because aggressive coefficient schedules caused a late loss penalty.

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
rational_adaptive_metric_onpolicy h3072             3.615887  PPL 37.184
rational_adaptive_metric_onpolicy h2880             3.615114  PPL 37.156
rational_quotient_jacobian_onpolicy h3072           3.615571  PPL 37.173
rational_transport_onpolicy h3072 matrix=0.65       3.615149  PPL 37.157
rational_transport_onpolicy h3072 matrix=0.70       3.615180  PPL 37.158
```

The seed-1337 incumbents are `3.614862` for `rational_jacobian_onpolicy + h3072` and `3.614475` for `rational_quotient_onpolicy + h2880`. The new probes beat AdamW/SILU on that seed, but they did not beat the existing rational-specific rows, so the full three-seed recommendation remains unchanged. Transport experiments showed that aggressive rational coefficient schedules improve some early checkpoints but create a late penalty; selector cooldown and coefficient pullback reduced but did not remove that penalty. The safest transport setting found is matrix-only transport with baseline coefficient dynamics.

### Transport Probe Analysis

The compact analysis artifacts are in `experiments/results/transport_optimizer_analysis_2026_05_27/`:

```text
loss_ppl_curves.png          validation loss and PPL curves
final_loss_ppl_bars.png      final validation loss and PPL bars
transport_probe_summary.csv  retained probe metrics
```

![Seed-1337 h3072 validation curves](experiments/results/transport_optimizer_analysis_2026_05_27/loss_ppl_curves.png)

![Seed-1337 h3072 final metrics](experiments/results/transport_optimizer_analysis_2026_05_27/final_loss_ppl_bars.png)

Same seed, same h3072 RLB setting:

| row | final loss | final PPL | gap vs Jacobian |
| --- | ---: | ---: | ---: |
| `rational_jacobian_onpolicy` | 3.614862 | 37.146 | +0.000000 |
| `rational_transport_onpolicy`, matrix `0.65`, baseline coeffs | 3.615149 | 37.157 | +0.000287 |
| `rational_transport_onpolicy`, matrix `0.70`, baseline coeffs | 3.615180 | 37.158 | +0.000318 |
| `rational_transport_onpolicy`, matrix-only early probe | 3.615939 | 37.186 | +0.001077 |
| `rational_transport_onpolicy`, matrix `0.60` plus time ramp | 3.616660 | 37.213 | +0.001798 |
| `rational_adaptive_metric_onpolicy` | 3.617174 | 37.232 | +0.002312 |
| AdamW on h3072 RLB | 3.617501 | 37.244 | +0.002639 |
| layer-staggered coefficient switch, pre-fix run | 3.619816 | 37.331 | +0.004954 |
| selector plus coefficient pullback | 3.619819 | 37.331 | +0.004957 |
| global switch at 43% progress | 3.620000 | 37.338 | +0.005138 |
| depth-corrected layer switch | 3.621418 | 37.391 | +0.006556 |
| aggressive xfast coefficient schedule | 3.625419 | 37.540 | +0.010557 |

The main positive result is narrow but real: RLB-aware matrix geometry is consistently useful. The best transport rows kept the rational coefficients on the conservative baseline path and only changed how the matrices see the learned rational curves. Raising the matrix preconditioner from the first matrix-only attempt to `0.65` closed most of the gap to the incumbent Jacobian optimizer, and `0.70` was essentially tied but slightly worse. A time ramp on the same mechanism was worse, which suggests the useful part is stable curve-aware scaling, not late extra pressure.

The main negative result is also consistent: aggressive coefficient motion is the wrong place to spend risk in this benchmark. The coefficient selector, layer-specific switches, reset-on-switch, freezes, and late pullback were all trying to avoid the late-penalty pattern, but they still landed well behind the matrix-only transport rows. The likely reason is that rational coefficients are small function parameters, not ordinary dense weights. Early large moves can change the learned scalar nonlinearity enough that later cooldown only stops further damage; it does not restore the better function-space basin.

The next optimizer design should therefore make selection reversible and acceptance-based instead of only scheduled. A stronger rational-specific optimizer should treat matrix preconditioning and gauge balancing as the default path, then allow coefficient proposals only when a local function-space test accepts them: bound the change on the probe grid, compare a short on-policy loss proxy against a frozen-coefficient shadow update, and roll back or decay the proposal when it loses. Layer-specific behavior is still worth using, but it should choose among validated conservative actions rather than switching into aggressive coefficient modes because a schedule says the phase changed.

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
