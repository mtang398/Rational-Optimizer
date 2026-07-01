# ICLR Method Draft

This folder contains the Overleaf-style ICLR draft for RationalOPT. It is a paper-shaped draft with abstract, introduction, related work, method, current fixed-scale experiments, conclusion, and a math-heavy appendix explaining the activation and optimizer design.

The draft uses the completed matched E1 and E2 result artifacts in `../../experiments/ICLR_RUN_STATUS.md`, `../../experiments/results/iclr26_e1_token_savings_2026_06_12/`, the completed E1/E2 JSONL run trees, the cleaned throughput summary in `../../experiments/results/iclr26_runtime_summary_2026_06_11/`, and the five `../../experiments/results/iclr26_e2_*` packages. The reported RLB rows use the audited `rlb_fused_global_rational` source rows, but the paper-facing method name is simply RLB. MatrixPolicy method constants are recorded in `../../experiments/manifests/iclr26_global_rational_matrixpolicy_manifest.csv` and the completed JSONL config records.

## Current Paper Position

```text
abstract: aligned to the RLB result anchor and E1 target-arrival claim
introduction/related work: motivation and positioning, included because the draft has an abstract
method: RLB and MatrixPolicy with defined notation, role-specific matrix signals, centered group policy, early gated Muon substep, and bounded positive pair balancing
main experiments: completed E1/E2 target-arrival table, one target-arrival map across common loss targets, and representative E1 loss/PPL/train-loss curves
reserved main-paper slots: larger-scale results and component ablations, with no current claim relying on missing results
appendix: shared protocol, E1/E2 support curves, E1 target-frontier controls, endpoint sanity table, and mathematical rationale for the activation and optimizer design
figures: `matrixpolicy_overview.pdf`, `matrixpolicy_signal_flow.pdf`, `target_arrival_evidence_matrix.pdf`, `e1_representative_silu_dynamics.pdf`, plus appendix E1/E2 support figures
tables: `tables/e1_e2_silu_summary_table.tex`, `tables/e1_target_time_table.tex`
```

## Build

Use the included ICLR style files and compile `main.tex` with the project-local TinyTeX renderer:

```bash
env PATH=/home/mt872/rationalOPT/.TinyTeX/bin/x86_64-linux:/usr/local/bin:/usr/bin:/bin /home/mt872/rationalOPT/.TinyTeX/bin/x86_64-linux/latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The figures and generated table are rebuilt with:

```bash
python3 generate_figures.py
```
