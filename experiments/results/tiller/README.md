# TILLER results

This directory contains TILLER results for the 12-layer and 18-layer endpoint
suites in one schema.

- `runs.csv`: 45 dataset–seed rows with endpoint status, loss, elapsed time,
  matched-control fields when available, and source hashes.
- `summary.csv`: fifteen model–budget–dataset cells aggregated from recorded
  finite endpoints.
- `checkpoints.csv`: recorded validation checkpoints along the available
  trajectories, including all 930 observations from the 12-layer studies.

Completed and pending rows share the same schema; pending measurements remain
empty until their required endpoint is available.

The 15 new 18-layer, 100M-token TILLER rows run for 3,050 steps and are
currently pending. Their comparison is SwiGLU + Muon. The two-stage optimizer
has its own directory, [tiller_then_muon](../tiller_then_muon/).
