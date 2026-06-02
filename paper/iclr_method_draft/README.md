# ICLR Method Draft

This folder contains an Overleaf-ready ICLR-style method draft for RationalOPT. The draft should now be written around the 3-seed FineWeb/FineWeb-Edu result, not the older one-seed screen.

## Current Paper Position

The paper should make one focused claim:

```text
Rational FFNs expose optimizer-visible geometry, and an on-policy matrix optimizer that uses this geometry trains rational language models more robustly and to lower heldout loss than generic AdamW or Muon under the same base protocol.
```

The current evidence is promising but not yet complete for an ICLR submission. The draft needs:

```text
3-seed primary table: available
same-protocol AdamW/Muon controls: available
RLB+AdamW divergence reported: available
mechanism telemetry implementation: available
mechanism diagnostic result figures: missing
method-component ablation table: deferred until tuned configs exist
stronger/tuned baselines: missing
Phase A HPO surfaces and rank-over-horizon plots: missing
scale or longer-budget test: missing
speed-to-target curves: missing
downstream sanity checks: missing
reproducibility statement and exact protocol appendix: partially available
bootstrap gap CIs and multi-seed mean plots: available
```

## Template Status

As of 2026-06-01, I found the official ICLR master template repository listing `iclr2026` but not `iclr2027`. This folder therefore still uses the ICLR 2026 style files:

```text
iclr2026_conference.sty
iclr2026_conference.bst
math_commands.tex
fancyhdr.sty
natbib.sty
```

Official references checked:

```text
https://iclr.cc/Conferences/2026/AuthorGuide
https://github.com/ICLR/Master-Template
```

Switch to the official ICLR 2027 template as soon as it is published.

## Paper Structure

Current main-paper shape after the ICLR cleanup:

```text
1. Introduction: optimizer-specific rational FFN claim and contributions.
2. Background and Related Work: optimizer evidence standard and RLB-specific geometry.
3. Rational Matrix Policy Optimization: one compact method section, no main-text subsection sprawl.
4. Experiments: current three-seed real-corpus screen, mean +/- std curves, and paper-result requirements.
5. Discussion and Limitations: scope of the claim, missing tuned baselines, scale, speed, and mechanism evidence.
6. Conclusion.
```

Appendix should contain:

```text
full optimizer pseudocode
all hyperparameters and launch commands
gauge-conditioning theoretical note
all per-seed tables
all curves and AUC metrics
nonfinite/divergent rows
hardware and environment
```

## Build

Local renderer used in this workspace:

```bash
/home/mt872/autoresearch_attempt_1/.local/bin/tectonic main.tex
```

For Overleaf, upload this folder and compile `main.tex` with the included style files.
