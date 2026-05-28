# RationalOPT

Read [READ_FIRST.md](READ_FIRST.md) before running jobs.

RationalOPT studies whether a no-GLU rational feed-forward layer can justify its own optimizer. The target is not simply to lower loss with a better global learning-rate schedule. The target is an on-policy optimizer that uses structure unique to the Rational Local Basis FFN (RLB) and produces a real lead over both:

```text
SiLU/SwiGLU + AdamW
RLB + AdamW
```

A useful result must survive those controls under the same model size, token budget, evaluation protocol, and training schedule. Jacobian, transport, layerwise switching, and any other rational-specific optimizer are candidate rows, not baselines.

## Model

The benchmark is a 123M-parameter LLaMA-style decoder-only Transformer on WikiText-103 with 100M training tokens per row. The main RLB activation row is:

```text
rlb_fused_fixed_strong_ffn        h = 3072
```

RLB replaces the SwiGLU FFN with one expansion projection, grouped rational feature generation, and one output projection:

```text
v = x W_in
s_g = sqrt(mean(v_g^2) + eps)
u_g = v_g / s_g
h_g = s_g R_g(u_g)
y = h W_out
```

There is no gate projection, no up projection, and no SiLU inside the RLB FFN.

## Fair Comparisons

The current comparison set is:

```text
SiLU/SwiGLU + AdamW                    primary external baseline
RLB + AdamW                            activation-controlled baseline
RLB + rational_onpolicy_balance        tested
RLB + rational_quotient_onpolicy       tested
RLB + rational_jacobian_onpolicy       tested, best fixed-LR rational row
RLB + rational_transport_onpolicy      tested prototype
RLB + rational_adaptive_metric_onpolicy prototype
RLB + rational_quotient_jacobian_onpolicy prototype
RLB + rational_jacobian_factored_onpolicy negative probe
RLB + rational_layerwise_switch_onpolicy  negative probe
```

Muon and `factored_adamw` are ablation controls, not the main claim. Rational-specific optimizers are only valid on RLB rows.

## Current Result

Three-seed fixed-LR full sweep:

```text
run name: rlb_optimizer_empirical_ngram_full
job ids:  763059 + 813929 + 821187
seeds:    1337, 2024, 31415
budget:   100M training tokens per row
```

Mean validation result:

| row | loss | PPL | gap vs SiLU+AdamW | gap vs RLB+AdamW |
| --- | ---: | ---: | ---: | ---: |
| AdamW + SiLU/SwiGLU | 3.610129 | 36.973 | +0.000000 | +0.003500 |
| AdamW + RLB h3072 | 3.606629 | 36.845 | -0.003500 | +0.000000 |
| RLB + `rational_onpolicy_balance` | 3.606226 | 36.831 | -0.003903 | -0.000403 |
| RLB + `rational_quotient_onpolicy` | 3.606664 | 36.847 | -0.003465 | +0.000035 |
| RLB + `rational_jacobian_onpolicy` | 3.605394 | 36.800 | -0.004736 | -0.001236 |

The honest conclusion is that the best measured rational optimizer is real but small. It improves the three-seed mean by `-0.004736` loss versus `SiLU/SwiGLU+AdamW` and by `-0.001236` loss versus `RLB+AdamW`. This is not close to the desired `0.2-0.3` loss or `2-3` PPL gap.

## What The Probes Mean

The recent probes are diagnostic, not separate stacked recommendations.

| probe | result | interpretation |
| --- | ---: | --- |
| Transport matrix-only, seed 1337 | best `3.615149`, PPL `37.157` | close to Jacobian, but not better; matrix geometry helps a little |
| Aggressive coefficient transport/switching | up to `3.625419`, PPL `37.540` | scheduled coefficient motion damages the function path |
| ASAM with fixed-LR rows | SiLU `3.622873`, RLB `3.618500` | did not improve the AdamW controls |
| Factored second moments | step-1000 loss `4.775638` | badly hurts this setup |
| Aggressive layerwise switch | step-1000 loss `4.221224` | switching by schedule is not enough |
| High-LR control | RLB+AdamW `3.455792`, SiLU+AdamW `3.456625` | reveals the old schedule was undertrained; not an optimizer win |

The high-LR run is kept because it prevents a false claim. High-LR `RLB+rational_jacobian_onpolicy` reaches `3.459508`, which looks excellent versus the old fixed-LR `SiLU+AdamW` row, but it loses to high-LR `RLB+AdamW` by `+0.003716` loss. Therefore the large absolute loss drop is a schedule/control issue, not the rational optimizer we are trying to design.

Compact artifacts:

```text
experiments/results/rlb_optimizer_empirical_ngram_full/
experiments/results/transport_optimizer_analysis_2026_05_27/
experiments/results/high_lr_optimizer_followup_2026_05_27/
```

![Fixed-LR rational optimizer comparison](experiments/results/transport_optimizer_analysis_2026_05_27/loss_ppl_curves.png)

![High-LR diagnostic control](experiments/results/high_lr_optimizer_followup_2026_05_27/loss_ppl_curves.png)

## RLB Structure The Optimizer Must Use

RLB has more exploitable structure than an ordinary MLP layer:

```text
W_in,g  <- c W_in,g
W_out,g <- W_out,g / c
```

This exact positive group gauge preserves the represented function. An optimizer can choose the best scale representative without changing the model.

The rational curve also gives on-policy functional information that AdamW ignores:

```text
R_g(u)                 feature amplitude seen by W_out
R'_g(u)                derivative seen by W_in
J_coeff R_g(u)         functional effect of coefficient updates
pole / denominator     stability risk of coefficient movement
u distribution         actual active scalar domain per layer/group
```

A real RLB optimizer should use these signals every step, by layer and by group, without relying on a hand-written phase schedule.

## Next Optimizer Target

The next serious design should be an on-policy functional trust optimizer for RLB. The intended behavior is:

1. Collect per-layer/per-group samples of `u`, `R(u)`, `R'(u)`, denominator margin, feature scale, and incoming/outgoing gradient pressure.
2. Remove exact gauge-gradient components from `W_in/W_out`, then apply a function-preserving gauge rebalance that minimizes predicted optimizer noise.
3. Precondition matrix gradients with on-policy derivative/output metrics, not a fixed probe grid.
4. Treat rational coefficients as function parameters: build a tiny per-group Gram/Gauss-Newton metric from `J_coeff R(u)`, damp it by denominator risk, and trust-clip by predicted function change.
5. Couple coefficient and matrix updates through a per-group controller. The controller can choose matrix-only, coefficient-natural-gradient, gauge-rebalance, or freeze mode based on live improvement/risk signals, not training step.
6. Accept coefficient motion only when on-policy gradient agreement and predicted functional gain are strong enough; otherwise spend the step on matrix and gauge updates.

This is the path most aligned with the goal: use all rational-specific advantages to beat `SiLU+AdamW` and `RLB+AdamW`, rather than finding an easier global schedule.

## Layout

```text
activation/         rational activation package and CUDA extension
training/           WikiText-103 training, sweep, and aggregation scripts
optimizer_design/   RLB-specific optimizer components
experiments/        cache, active runs, logs, and aggregate outputs
```

## Commands

Run a fixed-schedule comparison sweep:

```bash
env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_P2P_DISABLE=1 \
  RUN_NAME=rlb_optimizer_empirical_ngram_full \
  STEPS=0 \
  SEEDS="1337 2024 31415" \
  OPTIMIZERS="adamw rational_onpolicy_balance rational_quotient_onpolicy rational_jacobian_onpolicy rational_transport_onpolicy" \
  ACTIVATIONS="silu rlb_fused_fixed_strong_ffn" \
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
