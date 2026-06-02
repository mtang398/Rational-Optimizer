# ICLR Method Draft

This folder contains an Overleaf-ready ICLR-style method draft for RationalOPT. The draft should now be written around the 3-seed FineWeb/FineWeb-Edu result, not the older one-seed screen.

## Current Paper Position

The paper should make one focused claim:

```text
Rational FFNs expose optimizer-visible geometry, and an on-policy matrix optimizer that uses this geometry trains rational language models more robustly and to lower heldout loss than generic AdamW or Muon under the same base protocol.
```

Current source state:

```text
abstract, introduction, experiments, discussion, and conclusion are intentionally empty placeholders
method section is retained
conditional quotient-optimization theorem is retained in the appendix
appendix proof is organized as quotient geometry, canonicalized preconditioning, and contraction/stability
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

Current main-paper shape:

```text
1. Introduction: placeholder.
2. Background and Related Work: placeholder.
3. Rational Matrix Policy Optimization: compact method section.
4. Experiments: placeholder.
5. Discussion: placeholder.
6. Conclusion: placeholder.
```

Appendix should contain:

```text
full optimizer pseudocode
all hyperparameters and launch commands
conditional quotient-optimization separation proof
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
