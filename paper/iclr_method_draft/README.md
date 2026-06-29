# ICLR Method Draft

This folder contains the Overleaf-style ICLR method/appendix draft for RationalOPT. It is not yet the complete paper; non-method sections will be drafted after the current result narrative and citation set are locked.

The draft should use the manifest-first matched runs described in `../../experiments/ICLR_EXACT_RUN_PLAN.md`. Current E1 and E2 tables, curves, runtime summaries, and token-to-target readouts live in `../../experiments/ICLR_RUN_STATUS.md`, `../../experiments/results/iclr26_e1_figures/`, `../../experiments/results/iclr26_e1_token_savings_2026_06_12/`, the five `../../experiments/results/iclr26_e2_*` packages, and `../../experiments/results/iclr26_e2_figures/`. Exact submitted commands live in `../../experiments/ICLR_RUN_COMMANDS.md`. WikiText may remain a small demo anchor when useful.

## Current Paper Position

```text
abstract: method-facing abstract
method: rewritten paper-quality description of global-rational RLB and MatrixPolicy
appendix: rewritten exact properties, design rationale, active/inactive branches, and hyperparameters
experiments/data source: completed E1 plus completed E2 packages with global-rational/no-local-atom (`rlb_fused_global_rational`) MatrixPolicy and non-MatrixPolicy RLB-control overlays
not rendered here yet: introduction, related work, experiments, discussion, and conclusion
```

## Build

Use the included ICLR style files and compile `main.tex`. The local renderer used in this workspace is:

```bash
/home/mt872/autoresearch_attempt_1/.local/bin/tectonic main.tex
```
