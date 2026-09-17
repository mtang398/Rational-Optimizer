# TILLER results

This directory contains TILLER results for the 12-layer and 18-layer endpoint
suites in one schema.

- `runs.csv`: 45 dataset–seed rows with endpoint status, loss, elapsed time,
  matched-control fields when available, and source hashes.
- `summary.csv`: fifteen model–budget–dataset cells aggregated from recorded
  finite endpoints.
- `checkpoints.csv`: recorded validation checkpoints along the available
  trajectories: the completed 12-layer 100M-token study and all 15 primary
  18-layer runs.

The 15 rows for the 12-layer 300M-token study remain pending, with empty
measurement fields.

All 15 primary 18-layer, 100M-token TILLER rows are complete. They run for
3,050 steps and use SwiGLU + Muon as their matched comparison. Their formal
timing excludes training-time GRAIN diagnostic collection. The two-stage
optimizer has its own directory, [tiller_then_muon](../tiller_then_muon/).
