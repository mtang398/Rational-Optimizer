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
bounded protocol-lock launcher and summarizer scaffolding
reference-aligned AdEMAMix behavior: no slow-EMA bias correction, alpha warmup, beta3 half-life warmup
Phase 0A/0B smokes completed on dclm, fineweb_edu, dolma_sample, and c4_en
Phase 0C M1 DCLM smoke completed
first two Phase 1 protocol-lock DCLM shards submitted as jobs 67183 and 67184
```

## Immediate TODO

Use `experiments/ICLR_EXACT_RUN_PLAN.md` as the source of truth. The plan contains distinct optimizer-paper experiments, not a dressed-up version of the pilot.

1. Monitor jobs `67183` and `67184`; if either fails, inspect and rerun only the failed shard.
2. For every future result summary, generate and inspect curves first: densely sampled validation loss with eval interval <= 50, training loss, mean +/- std when multi-seed, AUC, and loss-vs-GPU-hour when timing exists. Final loss alone is insufficient.
3. If both complete, summarize with `experiments/scripts/summarize_iclr_phase1_protocol_lock.py`.
4. Continue Phase 1 protocol lock on `dclm` and `fineweb_edu` with at most two 4-GPU jobs active.
5. Run Phase 2 M0 100M main suite on `dclm`, `fineweb_edu`, `fineweb`, `dolma_sample`, and `c4_en`.
6. Repeat Phase 2 100M for seeds 2027 and 3407.
7. Run Phase 2 300M after 100M summaries.
8. Run Phase 3 600M long-horizon frontier on decisive rows.
9. Run Phase 4 M1 scale confirmation and optional M2 stretch.
10. Run Phase 5 batch, throughput, memory, and optimizer-state accounting.
11. Run Phase 6 cross-corpus transfer from 300M checkpoints.
12. Run Phase 7 corpus-shift continued training.
13. Run Phase 8 sensitivity maps.
14. Run Phase 9 mechanism diagnostics from main logs.
15. Run Phase 10 method ablations last.

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
method-component ablation table is missing and should wait until baseline-locked configs exist
larger model scale and longer token budget are missing
third/fourth corpus and transfer-task evidence are missing
wall-clock/tokens-to-target/GPU-hour story is not yet clean
statistical reporting needs mean +/- std curves, CIs, divergence accounting, and exact failed-run policy
```

Score needed before a strong ICLR submission: at least 8.7 / 10. The fastest path is a decisive headline benchmark, speed-to-target/overhead accounting, scale and transfer evidence, mechanism diagnostics tied to optimizer role behavior, and only then explanatory ablations.
