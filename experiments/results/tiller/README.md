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

Five completed 18-layer, 300M-token TILLER rows now have their exact matched
SwiGLU + Muon endpoints. Their per-seed loss leads are recorded directly in
`runs.csv`; the paired timing-ratio field remains empty when the candidate and
control timing boundaries differ.
