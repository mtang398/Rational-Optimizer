# TILLER results

This directory contains TILLER results for the 12-layer and 18-layer endpoint
suites in one schema.

- `runs.csv`: 45 dataset–seed rows with endpoint status, loss, elapsed time,
  matched-control fields when available, and source hashes.
- `summary.csv`: fifteen model–budget–dataset cells aggregated from recorded
  finite endpoints.
- `checkpoints.csv`: 1,975 recorded validation checkpoints from complete and
  incomplete trajectories.

Completed and pending rows share the same schema; pending measurements remain
empty until their required endpoint is available.
