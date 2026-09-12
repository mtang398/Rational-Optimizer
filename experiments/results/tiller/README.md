# TILLER results

This directory contains TILLER results for the 12-layer and 18-layer endpoint
suites in one schema.

- `runs.csv`: 45 dataset–seed rows with endpoint status, loss, elapsed time,
  matched-control fields when available, and source hashes.
- `summary.csv`: fifteen model–budget–dataset cells aggregated from recorded
  finite endpoints.
- `checkpoints.csv`: recorded validation checkpoints along the available
  trajectories, including the complete 12-layer studies and the primary
  18-layer runs as they are collected.

Completed and pending rows share the same schema; pending measurements remain
empty until their required endpoint is available.

All 15 primary 18-layer, 100M-token TILLER rows are complete. They run for
3,050 steps and use SwiGLU + Muon as their matched comparison. Their formal
timing excludes training-time GRAIN diagnostic collection. The two-stage
optimizer has its own directory, [tiller_then_muon](../tiller_then_muon/).
