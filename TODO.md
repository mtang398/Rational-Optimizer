# Research TODO

## North-Star Claim

The paper claim should stay optimizer-specific:

> Rational FFNs expose optimizer-visible geometry, and an on-policy optimizer that uses that geometry trains rational language models faster and more robustly than generic AdamW/Muon under the same base LR schedule.

A result is paper-level only if it beats the strongest `SiLU/SwiGLU+AdamW`, `RLB+AdamW`, `SiLU/SwiGLU+Muon`, and `RLB+Muon` controls on dense curves and real LM settings.

## Current Evidence Read

### 3-Seed Real-Corpus LM Pilot

| task | MatrixPolicy mean | SiLU+AdamW mean | best non-MatrixPolicy mean | gap vs SiLU+AdamW | gap vs best control |
| --- | ---: | ---: | ---: | ---: | ---: |
| FineWeb | 4.369701 loss / 79.04 PPL | 4.528963 loss / 92.69 PPL | 4.522311 loss / 92.08 PPL | 0.159263 | 0.152302 |
| FineWeb-Edu | 4.069422 loss / 58.52 PPL | 4.223572 loss / 68.28 PPL | 4.223572 loss / 68.28 PPL | 0.154149 | 0.153402 |

This is the current pilot evidence. It replicated across seeds `1337`, `2027`, and `3407` on both datasets, but it must not define the final paper plan or the final test setting.

Important caveat: FineWeb-Edu `RLB+AdamW` has one divergent seed, so its aggregate is poor. This should be reported, not hidden. It supports the optimizer-specific story.

### WikiText-103 Anchor

```text
RLB MatrixPolicy-Muon:       3.476232 loss / 32.34 PPL
Best SiLU/SwiGLU+AdamW row:  3.549346 loss / 34.79 PPL
Gap:                         0.073114 loss / 2.45 PPL
```

WikiText remains a useful same-LR LM anchor, but it is no longer the main result.

## Completed

```text
same-protocol FineWeb and FineWeb-Edu screen
3 seeds on both real-corpus tasks
AdamW and Muon controls for SiLU and RLB
committed compact JSONL traces for the new seeds
multi-seed summarizer and result tables
bootstrap gap CI table
multi-seed train/eval curve CSVs and mean plots
requeue-safe launcher behavior for activation-level reruns
1000-step MatrixPolicy ablation launcher prepared but intentionally not next in the paper sequence
full ICLR optimizer experiment blueprint
optimizer telemetry instrumentation for gradient, timing, CUDA memory, probe movement, RLB stats, MatrixPolicy role stats, and SVD entropy
broad baseline optimizer wiring for Lion, paper-style AdEMAMix, Schedule-Free AdamW-style, Adafactor/CAME-style, and SOAP/Shampoo-style AdamW
CUDA/DDP telemetry validation launcher/checker and completed validation summary
bounded tuning launcher and summarizer scaffolding
reference-aligned AdEMAMix behavior: no slow-EMA bias correction, alpha warmup, beta3 half-life warmup
```

## Immediate TODO

The exact new experiment matrix is `experiments/ICLR_EXACT_RUN_PLAN.md`. The goal is to copy accepted optimizer-paper experiment shapes with new paper-making runs.

1. Validate DCLM and Dolma data loaders with tiny 500-step M0 runs.
2. Validate the M1-260M model on 4 A6000s with a 500-step FineWeb-Edu smoke.
3. Run Sophia/SOAP-style speed-to-target experiments: M0 on FineWeb-Edu, FineWeb, and DCLM at 100M/300M/600M tokens.
4. Run Fantastic-style model/data scaling: M0 and M1 across token budgets and data ratios, reporting ranking flips.
5. Run SOAP-style batch-size and overhead experiments at 16k/32k/65k global tokens.
6. Run Adam-mini/GaLore/CAME-style memory and throughput accounting.
7. Run Lion/Schedule-Free-style broad transfer across FineWeb-Edu, FineWeb, DCLM, and Dolma without retuning.
8. Run AdEMAMix-style long-horizon corpus shift and forgetting: FineWeb-Edu continuation into DCLM.
9. Run post-training probe on selected checkpoints.
10. Run LR/WD landscapes only as reviewer defense.
11. Run mechanism diagnostics from logs as support.
12. Run method ablations last, only after the main optimizer evidence exists.

## Mechanism Diagnostics Needed

Required metrics per RLB layer:

| metric | meaning |
| --- | --- |
| group input RMS | whether `W_in` chooses usable domains. |
| group output RMS | whether features are used. |
| derivative pressure | whether groups are saturated or active. |
| denominator/pole margin | rational stability. |
| `W_in`/`W_out` norm product | role-scale drift and optimizer-induced imbalance. |
| coefficient update norm | rational-shape movement. |
| function probe delta | output function change on fixed probe inputs. |

Pass criterion: MatrixPolicy should show better loss/AUC with better function-delta-per-parameter-delta, lower wasted role-scale drift, and acceptable optimizer overhead versus generic optimizers.

## MatrixPolicy v2 Design Target

The next optimizer should remain a policy over RLB roles and groups, not a global LR schedule.

Inputs:

```text
layer depth
matrix role: W_in, coefficients, W_out
group activity and output use
derivative pressure / saturation
denominator risk
W_in/W_out scale drift
recent gradient agreement
```

Actions:

```text
role-specific update rule and beta2
per-group matrix scale from live stats
coefficient trust radius when denominator risk is high
rebalance strength when W_in/W_out scale drift grows
group revive/damp decisions for dead or saturated groups
```

Hard rule: do not count a global LR schedule change as optimizer progress.

## Harsh Self-Review

Current internal score: 8.0 / 10.

Why it improved:

```text
3-seed replication is now complete on two real web corpora.
MatrixPolicy beats SiLU+AdamW and the best non-MatrixPolicy control by about 0.15 loss.
Muon controls are included and are worse.
Plain RLB+AdamW instability is visible and supports the optimizer-specific claim.
```

Remaining weaknesses:

```text
broad baselines are implemented but not yet reference-matched or tuned enough for final claims
Sophia-style and exact reference SOAP/CAME comparisons are still missing or only approximate
mechanism telemetry is implemented and CUDA/DDP validation passed, but paper figures are missing
method-component ablation table is missing and should wait until tuned configs exist
larger model scale and longer token budget are missing
third/fourth corpus and transfer-task evidence are missing
wall-clock/tokens-to-target/GPU-hour story is not yet clean
statistical reporting needs mean +/- std curves, CIs, divergence accounting, and exact failed-run policy
```

Score needed before a strong ICLR submission: at least 8.7 / 10. The fastest path is a decisive tuned headline benchmark, speed-to-target/overhead accounting, scale and transfer evidence, mechanism diagnostics tied to optimizer role behavior, and only then explanatory ablations.
