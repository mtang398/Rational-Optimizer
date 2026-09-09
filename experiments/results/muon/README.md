# Muon results

This directory contains the SwiGLU/GRAIN activation comparison with Muon
across the published model scales, five datasets, and three seeds.

- `runs.csv`: run-level endpoint and timing records.
- `summary.csv`: unique dataset–budget–activation cells.
- `checkpoints.csv`: checkpoint-level validation records when available.

Both SwiGLU and GRAIN activation records use the same table schema.

The 18-layer, 1,024-wide suite trains for 3,050 steps and approximately
100M tokens. Its 30 Muon rows are currently pending. Completed 12-layer
results remain in the same tables.
