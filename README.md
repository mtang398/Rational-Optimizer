# RationalOPT

RationalOPT is a research artifact for one question:

Can a no-GLU Rational Local Basis FFN train better than a SiLU/SwiGLU FFN when the optimizer uses the rational block's own geometry?

The current answer is empirical and optimizer-specific. The public claim is not "RLB alone is better." Plain RLB with generic optimizers is inconsistent. The supported claim is that RLB exposes useful optimizer-visible structure, and `rational_matrix_policy_onpolicy` uses that structure better than AdamW or Muon under the same base training protocol.

## Current Claim

The strongest current evidence is a 3-seed real-corpus language-model screen on FineWeb and FineWeb-Edu. Each row uses the same model size, tokenizer, token budget, validation slice, seed set, global batch, base LR schedule, weight decay, and evaluation cadence.

Protocol summary:

```text
model: 12-layer GPT-style Transformer, d_model=768, heads=12, 123.6M params
tokenizer: GPT-2
train budget: 100M tokens
validation budget: 4M tokens after a 110M-token stream offset
sequence length: 256
global tokens per step: 32,768
steps: 3,050
seeds: 1337, 2027, 3407
hardware rule: 4 A6000 GPUs per job; at most 8 A6000 GPUs active total
```

### FineWeb, 3 Seeds

| method | n | mean val loss | std | mean PPL | gap vs SiLU+AdamW | gap vs best control |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SiLU+AdamW | 3 | 4.528963 | 0.029611 | 92.69 | 0.000000 | -0.006960 |
| RLB+AdamW | 3 | 4.522311 | 0.029832 | 92.08 | 0.006653 | -0.000308 |
| SiLU+Muon | 3 | 4.566661 | 0.041469 | 96.28 | -0.037698 | -0.044658 |
| RLB+Muon | 3 | 4.571341 | 0.027720 | 96.70 | -0.042377 | -0.049337 |
| RLB+MatrixPolicy (group-stat) | 3 | 4.369701 | 0.026358 | 79.04 | 0.159263 | 0.152302 |

### FineWeb-Edu, 3 Seeds

| method | n | div | mean val loss | std | mean PPL | gap vs SiLU+AdamW | gap vs best control |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SiLU+AdamW | 3 | 0 | 4.223572 | 0.001635 | 68.28 | 0.000000 | -0.000748 |
| RLB+AdamW | 3 | 1 | 5.618928 | 2.418773 | 1545.54 | -1.395356 | -1.396103 |
| SiLU+Muon | 3 | 0 | 4.258871 | 0.014706 | 70.74 | -0.035300 | -0.036047 |
| RLB+Muon | 3 | 0 | 4.263744 | 0.008026 | 71.08 | -0.040173 | -0.040920 |
| RLB+MatrixPolicy (group-stat) | 3 | 0 | 4.069422 | 0.002281 | 58.52 | 0.154149 | 0.153402 |

Positive gaps mean lower validation loss than the comparison row. The full per-seed table is in `experiments/results/real_lm_multiseed_2026_05_31/summary.md`.

## Interpretation

What is supported:

- `RLB+MatrixPolicy (group-stat)` replicated on both real corpora across 3 seeds.
- The gain is about 0.15 validation loss over `SiLU+AdamW` and the best non-MatrixPolicy control on both datasets.
- Generic Muon is not the explanation; both `SiLU+Muon` and `RLB+Muon` are worse than AdamW controls in these runs.
- Plain `RLB+AdamW` is not the explanation; it is marginal on FineWeb and diverges for one FineWeb-Edu seed.

What is not yet claimed:

- This is not yet an ICLR-ready optimizer paper by itself.
- The baselines are same-protocol but not fully hyperparameter-tuned against the new method.
- Mechanism diagnostics for gauge drift and function-space movement still need to be added.
- The result has not yet been stress-tested at a larger model size, longer token budget, or third corpus.

## Method Sketch

RLB replaces the FFN nonlinearity with grouped normalized rational functions:

```text
z = W_in x
z_g = group_g(z)
r_g = sqrt(mean(z_g^2) + eps)
u_g = z_g / r_g
h_g = r_g R_g(u_g)
y = W_out concat_g(h_g)
```

This creates a positive group gauge. For any `a_g > 0`:

```text
W_in[g]  <- a_g W_in[g]
W_out[g] <- W_out[g] / a_g
```

The represented function is unchanged, but generic optimizers see different norms and conditioning. MatrixPolicy exploits this structure by partitioning parameters into backbone weights, rational coefficients, `W_in`, and `W_out`; applying role/depth/time-aware matrix updates; using mild on-policy group-stat scaling; and applying an exact post-step gauge rebalance.

The best current row is:

```text
activation: rlb_fused_fixed_strong_ffn
optimizer: rational_matrix_policy_onpolicy
variant: MatrixPolicy with group-stat scaling
backbone: AdamW
base LR schedule: same as controls, optimizer_lr=3e-4 to optimizer_min_lr=3e-5
```

## Reproducing The Tables

Regenerate the current multi-seed summary from committed JSONL traces plus the seed-1337 baseline summary:

```bash
.venv-cu128/bin/python experiments/scripts/summarize_real_lm_multiseed.py \
  --run-root experiments/runs/real_lm_multiseed_20260531 \
  --baseline-summary-csv experiments/results/real_lm_screen_2026_05_30/summary.csv \
  --baseline-seed 1337
```

Primary result artifacts:

```text
experiments/results/real_lm_multiseed_2026_05_31/summary.md
experiments/results/real_lm_multiseed_2026_05_31/per_seed_summary.csv
experiments/results/real_lm_multiseed_2026_05_31/aggregate_summary.csv
experiments/runs/real_lm_multiseed_20260531/
```

The older one-seed result package still contains plot images and curve CSVs:

```text
experiments/results/real_lm_screen_2026_05_30/
```

## Resource Rules

These are hard operational constraints for this repo:

```text
max 4 A6000 GPUs per task/job
max 8 A6000 GPUs active at the same time
keep repository size below 200G
```

The real-LM launcher refuses to run above its configured repository-size guard and is written for 4-GPU A6000 jobs. Submit at most two active jobs at once; dependency-queued jobs are acceptable if they cannot increase active usage beyond 8 GPUs.

## Repository Map

```text
activation/         RLB activation implementation and math notes
optimizer_design/   MatrixPolicy optimizer implementation and design notes
training/           LM harness, dataset streaming, and optimizer wiring
experiments/        launchers, summarizers, result summaries, compact JSONL traces
paper/              Overleaf-ready ICLR method draft
READ_FIRST.md       short reading order and claim boundary
TODO.md             research backlog and paper-readiness checklist
```

## Next Work

The next phase is about making the result paper-grade, not just larger:

1. Add one-command reproduction, curves, AUC summaries, and bootstrap confidence intervals.
2. Add diagnostics for gauge drift, group norm products, rational activity, denominator risk, and function-space movement.
3. Run full ablations: no group stats, no early Muon phase, no exact gauge rebalance, gain-only, pressure-only, activity-only.
4. Tune strong baselines fairly and add at least one serious modern optimizer baseline if practical.
5. Repeat at a larger model size and a longer token budget while respecting the 4-GPU/job and 8-GPU-active cap.
