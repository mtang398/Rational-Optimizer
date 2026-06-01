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
1000-step MatrixPolicy ablation launcher prepared
```

## Immediate TODO

1. Finish the provenance bundle around the multi-seed table: commit hash, commands, job IDs, seed list, dataset slices, and environment metadata.
2. Add function-space and gauge-drift diagnostics to the training loop.
3. Run and summarize the prepared 1000-step ablation probes: no group stats, no early Muon phase, no exact gauge rebalance, gain-only, pressure-only, activity-only.
4. Tune AdamW, RLB+AdamW, and MatrixPolicy on a small grid under the same token budget.
5. Add at least one stronger modern optimizer baseline if feasible.
6. Run one larger-model or longer-budget test while respecting the 4-GPU/job and 8-GPU-active cap.
7. Fix DCLM/Dolma environment issues and add a third corpus.

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
baselines are not yet fully tuned
mechanism diagnostics are missing
ablation table is missing
larger scale and longer budget are missing
third corpus is missing
wall-clock/tokens-to-target story is not yet clean
```

Score needed before a strong ICLR submission: at least 8.7 / 10. The fastest path is diagnostics plus ablations plus one harder baseline/scale test.
