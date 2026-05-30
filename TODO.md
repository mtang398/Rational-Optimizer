# Research TODO

## North-Star Claim

The paper claim should be optimizer-specific:

> Rational FFNs expose optimizer-visible geometry, and an on-policy optimizer that uses that geometry trains rational language models faster and more robustly than generic AdamW/Muon under the same base LR schedule.

A result is not paper-level unless it beats the strongest `SiLU/SwiGLU+AdamW`, `RLB+AdamW`, `SiLU/SwiGLU+Muon`, and `RLB+Muon` controls on dense curves and real LM settings.

## Current Evidence Read

### Real-Corpus LM

The May 30 real-corpus screen is the main evidence:

| task | MatrixPolicy row | main AdamW control | gap vs SiLU+AdamW | extra readout |
| --- | ---: | ---: | ---: | --- |
| FineWeb | 4.344150 loss / 77.03 PPL | 4.504617 loss / 90.43 PPL | 0.160467 loss / 13.40 PPL | also beats RLB+AdamW by 0.148863 loss / 12.36 PPL. |
| FineWeb-Edu | 4.072055 loss / 58.68 PPL | 4.225019 loss / 68.38 PPL | 0.152964 loss / 9.70 PPL | RLB+AdamW diverges at train step 80, validation step 100. |

This is substantially stronger than the WikiText anchor and is now the main story. It is still one seed, so it should be treated as strong evidence, not a final paper claim.

### WikiText-103 Anchor

```text
RLB MatrixPolicy-Muon:       3.476232 loss / 32.34 PPL
Best SiLU/SwiGLU+AdamW row:  3.549346 loss / 34.79 PPL
Gap:                         0.073114 loss / 2.45 PPL
```

WikiText is kept because it is still a useful real-LM control run. It is no longer the main result.

### Removed Synthetic Evidence

The earlier saturated synthetic result packages were removed from tracked public artifacts. They were useful for finding curve-speed behavior but are not strong enough for the current paper story because all rows compress near the floor. Future synthetic work must be harder, non-saturated, and mechanism-targeted.

## What Is Actually Paper-Level

### 1. Multi-Seed Real-Corpus Confirmation

Repeat the exact FineWeb and FineWeb-Edu best comparison for at least one more seed.

Pass criterion:

```text
MatrixPolicy remains best on validation loss/PPL and AUC.
Average gap vs SiLU+AdamW stays clearly positive.
Plain RLB+AdamW instability or weakness is characterized, not hidden.
```

### 2. Function-Space Movement Audit

Dense curves show performance but not mechanism. Add diagnostics that measure whether each optimizer spends updates on useful function change rather than gauge drift.

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

### 3. Stronger Real-LM Task Set

The next tasks should be closer to actual pretraining than toy synthetic tasks:

| benchmark | reason |
| --- | --- |
| DCLM baseline slice | curated modern web corpus; good pretraining proxy once zstd support is installed. |
| OpenWebText/C4-style slice | common web LM benchmark family; useful external comparability. |
| code-heavy real corpus | tests structured long-range token patterns unlike web prose. |
| longer FineWeb run | checks whether the 0.15-0.16 loss gap persists or grows beyond 100M tokens. |

### 4. MatrixPolicy v2 Design Target

The next optimizer should remain a policy over RLB roles and groups, not a global LR schedule.

Inputs:

- layer depth
- matrix role: `W_in`, coefficients, `W_out`
- group activity and output use
- derivative pressure / saturation
- denominator risk
- gauge drift
- recent gradient agreement

Actions:

- role-specific update rule and beta2
- per-group matrix scale from live stats
- coefficient trust radius when denominator risk is high
- gauge rebalance strength when `W_in`/`W_out` drift grows
- group revive/damp decisions for dead or saturated groups

Hard rule: do not count a global LR schedule change as optimizer progress.

## Immediate TODO

1. Add function-space and gauge-drift diagnostics to the training loop.
2. Repeat FineWeb and FineWeb-Edu with a second seed using the same protocol.
3. Fix the environment for DCLM streaming and run the same five-row control set.
4. Add a code-heavy real-corpus task with the same 100M-token budget if storage allows.
5. Design MatrixPolicy v2 around measured failure modes, not around more LR scheduling.
6. Only after a large same-LR advantage is established, run LR robustness sweeps.

## Harsh Self-Review

Current internal score: 7.4 / 10.

Why it improved:

- The strongest evidence is now real-corpus LM, not saturated synthetic tasks.
- MatrixPolicy beats both SiLU+AdamW and RLB controls on FineWeb, and it avoids the FineWeb-Edu instability that breaks RLB+AdamW.
- Muon controls are included and are worse, so the result is not explained by generic matrix optimization.

Remaining weaknesses:

- One seed is not enough.
- The gap is strong but still below the original `0.2-0.3` loss target.
- Mechanism diagnostics are not yet implemented.
- DCLM/code-style transfer is still missing.

Score needed before paper claim: at least 8.0 / 10. The fastest path is multi-seed real-corpus confirmation plus function-space/gauge diagnostics showing why MatrixPolicy works.
