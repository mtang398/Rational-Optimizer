# ICLR Method Draft

This folder contains the Overleaf-style ICLR draft for RationalOPT. It is a paper-shaped draft with abstract, introduction, related work, method, current fixed-scale experiments, conclusion, and a math-heavy appendix explaining the activation and optimizer design.

The draft uses the completed matched E1 and E2 result artifacts in `../../experiments/ICLR_RUN_STATUS.md`, `../../experiments/results/iclr26_e1_token_savings_2026_06_12/`, the completed E1/E2 JSONL run trees, the cleaned throughput summary in `../../experiments/results/iclr26_runtime_summary_2026_06_11/`, and the five `../../experiments/results/iclr26_e2_*` packages. The reported RLB results use the global-rational/no-local-atom (`rlb_fused_global_rational`) variant. MatrixPolicy method constants are recorded in `../../experiments/manifests/iclr26_global_rational_matrixpolicy_manifest.csv` and the completed JSONL config records.

## Current Paper Position

```text
abstract: aligned to the global-rational/no-local-atom result anchor and E1 target-arrival claim
introduction/related work: motivation and positioning, included because the draft has an abstract
method: global-rational RLB and MatrixPolicy with defined notation, role-specific matrix signals, centered group policy, early gated Muon substep, and bounded positive pair balancing
main experiments: completed E1 validation trajectories, E1 target-arrival frontiers, and E1 token/time savings against SiLU+AdamW, RLB+AdamW, and RLB+Muon
reserved main-paper slots: larger-scale results and component ablations, with no current claim relying on missing results
appendix: shared protocol, E2 saturation curves, E1 companion train/PPL curves, endpoint sanity table, and mathematical rationale for the activation and optimizer design
figures: `matrixpolicy_overview.pdf`, `matrixpolicy_signal_flow.pdf`, `e1_validation_all_datasets.pdf`, `e1_target_frontiers.pdf`, `e1_multimetric_examples.pdf`, `e2_validation_dynamics.pdf`, `e2_perplexity_dynamics.pdf`, `e2_training_dynamics.pdf`
table: `tables/e1_target_time_table.tex`
```

## Build

Use the included ICLR style files and compile `main.tex`. The local renderer used in this workspace is:

```bash
/home/mt872/autoresearch_attempt_1/.local/bin/tectonic main.tex
```

The figures and generated table are rebuilt with:

```bash
python3 generate_figures.py
```
