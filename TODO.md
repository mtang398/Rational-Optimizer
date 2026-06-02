# Research TODO

## North-Star Claim

The paper claim should stay optimizer-specific:

> Rational FFNs expose optimizer-visible geometry, and an on-policy optimizer that uses that geometry trains rational language models faster and more robustly than generic AdamW/Muon under the same base LR schedule.

A result is paper-level only if it beats the strongest `SiLU/SwiGLU+AdamW`, `RLB+AdamW`, `SiLU/SwiGLU+Muon`, and `RLB+Muon` controls on dense curves and real LM settings.

## Current Evidence Read

### 3-Seed Real-Corpus LM

| task | MatrixPolicy mean | SiLU+AdamW mean | best non-MatrixPolicy mean | gap vs SiLU+AdamW | gap vs best control |
| --- | ---: | ---: | ---: | ---: | ---: |
| FineWeb | 4.369701 loss / 79.04 PPL | 4.528963 loss / 92.69 PPL | 4.522311 loss / 92.08 PPL | 0.159263 | 0.152302 |
| FineWeb-Edu | 4.069422 loss / 58.52 PPL | 4.223572 loss / 68.28 PPL | 4.223572 loss / 68.28 PPL | 0.154149 | 0.153402 |

This is now the main evidence. It replicated across seeds `1337`, `2027`, and `3407` on both datasets.

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
Phase A HPO launcher and summarizer scaffolding
reference-aligned AdEMAMix behavior: no slow-EMA bias correction, alpha warmup, beta3 half-life warmup
```

## Immediate TODO

This TODO is the research standard, not a resource-budgeted shortcut. Do not weaken the paper plan because of the current cluster allocation. Also do not pretend industrial LLM pretraining is required or feasible. The target is the strongest academic version of the project: scalable proxy evidence, tuned fair controls, enough model/token scaling to show the law of motion, transfer tests, and mechanism evidence. Operational partitioning belongs in launch scripts and `experiments/README.md`, not in this evidence checklist.

1. Build the accepted-paper comparison table: AdamW, Muon, Lion, AdEMAMix, Schedule-Free AdamW, SOAP/Shampoo, Adafactor/CAME, Sophia-style second-order where feasible, and MatrixPolicy; for every baseline record exact implementation source, deviations, hyperparameter grid, overhead, and stability.
2. Run fair Phase A HPO on FineWeb-Edu and FineWeb with dense LR/WD/beta/clip/eps surfaces, not single points; plot mean +/- std validation curves and heatmaps for every optimizer family.
3. Select tuned configs only from Phase A, then run final benchmarks on FineWeb-Edu, FineWeb, DCLM, and one additional transfer corpus/task with enough seeds for confidence intervals.
4. Report speed-to-target in tokens, steps, GPU-hours, and wall-clock time, plus optimizer memory and throughput overhead.
5. Run academic-scale scaling studies: current 123M setting, a larger model, a longer token budget, and a transfer setting where HPO-selected configs are not retuned; do not frame this as industrial pretraining.
6. Run mechanism experiments after tuned configs exist: gauge-equivalent initialization, mid-training gauge intervention, optimizer-state gauge covariance, rational function movement, role-specific update geometry, denominator/pole safety, and group activity pressure.
7. Treat divergent runs as data: report divergence rate, clipping behavior, nonfinite policy, and excluded PPL rows explicitly.
8. Keep the current 3-seed FineWeb/FineWeb-Edu result as preliminary evidence only, not the final paper benchmark.

## Mechanism Diagnostics Needed

Required metrics per RLB layer:

| metric | meaning |
| --- | --- |
| group input RMS | whether `W_in` chooses usable domains. |
| group output RMS | whether features are used. |
| derivative pressure | whether groups are saturated or active. |
| denominator/pole margin | rational stability. |
| `W_in`/`W_out` norm product | gauge drift. |
| coefficient update norm | rational-shape movement. |
| function probe delta | output function change on fixed probe inputs. |

Pass criterion: MatrixPolicy should show better loss/AUC with better function-delta-per-parameter-delta or lower harmful gauge drift than generic optimizers.

## MatrixPolicy v2 Design Target

The next optimizer should remain a policy over RLB roles and groups, not a global LR schedule.

Inputs:

```text
layer depth
matrix role: W_in, coefficients, W_out
group activity and output use
derivative pressure / saturation
denominator risk
gauge drift
recent gradient agreement
```

Actions:

```text
role-specific update rule and beta2
per-group matrix scale from live stats
coefficient trust radius when denominator risk is high
gauge rebalance strength when W_in/W_out drift grows
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

Score needed before a strong ICLR submission: at least 8.7 / 10. The fastest path is telemetry validation, broad tuned baselines, Phase A HPO, speed-to-target, scale, and then mechanism interventions/ablations.
