# ICLR Method Draft

This folder contains the Overleaf-style ICLR draft for RationalOPT. It is a paper-shaped draft with abstract, introduction, related work, method, current completed M0 experiments, and a math-heavy appendix explaining the activation and optimizer design.

The draft uses the completed matched E1 and E2 result artifacts in `../../experiments/ICLR_RUN_STATUS.md`, `../../experiments/results/iclr26_e1_figures/`, `../../experiments/results/iclr26_e1_token_savings_2026_06_12/`, the five `../../experiments/results/iclr26_e2_*` packages, and `../../experiments/results/iclr26_e2_figures/`. The reported RLB results use the global-rational/no-local-atom (`rlb_fused_global_rational`) variant. MatrixPolicy method constants are recorded in `../../experiments/manifests/iclr26_global_rational_matrixpolicy_manifest.csv` and the completed JSONL config records.

## Current Paper Position

```text
abstract: aligned to the global-rational/no-local-atom result anchor
introduction/related work: motivation and positioning, included because the draft has an abstract
method: global-rational RLB and MatrixPolicy with defined notation, role-specific matrix signals, centered group policy, early Muon branch, and bounded positive rescaling
appendix: mathematical rationale for the activation and optimizer design, including rational derivatives, RLB group Jacobian, RMS-floor rescaling error, pressure proxy, centered policy, and rescaling contraction
experiments/data source: completed E1 plus completed E2 packages with global-rational/no-local-atom (`rlb_fused_global_rational`) MatrixPolicy and non-MatrixPolicy RLB-control overlays
not rendered here yet: larger scaling experiments and any final discussion/conclusion that depends on those results
```

## Build

Use the included ICLR style files and compile `main.tex`. The local renderer used in this workspace is:

```bash
/home/mt872/autoresearch_attempt_1/.local/bin/tectonic main.tex
```
