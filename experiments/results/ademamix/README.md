# ADeMaMix results

This directory contains the SwiGLU/GRAIN activation comparison with ADeMaMix
across the published model scales, five datasets, and three seeds.

- `runs.csv`: run-level endpoint and timing records.
- `summary.csv`: unique dataset–budget–activation cells.
- `checkpoints.csv`: checkpoint-level validation records when available.

The tables retain the large finite losses, non-finite losses, and early-stop
status produced by this optimizer. Numeric aggregates include only finite
losses from runs that reached the required endpoint.
