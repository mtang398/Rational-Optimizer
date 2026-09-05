# Repository storage and candidate-deletion policy

The active optimizer campaign keeps only reusable core code, the exact data
required by the current protocol, active candidates, and the three eventual
finalists.

## Data retained

- Five exact 300,000,000-token training tensors: DCLM, FineWeb, FineWeb-Edu,
  Dolma sample, and C4-en.
- Their exact 8,000,000-token validation tensors.
- Tokenizer metadata needed to load those tensors offline.

Smaller train/validation caches and streaming/regenerated substitutes are
excluded from the repository so the 300M campaign cannot silently use them.

## Active-campaign limits

- At most ten candidate implementations and one shared experiment package.
- One source file per candidate slot; minor repairs overwrite it.
- One compact manifest, preregistration, validator, launcher, and analyzer for
  the campaign. Candidate-specific copies of the repository are forbidden.
- Resume checkpoints are temporary and deleted immediately after the signed
  trajectory validates. Runtime snapshots store hashes and environment facts,
  never complete source-tree copies.
- Non-cache campaign artifacts should remain below 1 GiB. A size audit runs
  after every completed or failed job.

## Rejection cleanup

When a mathematical candidate is rejected, cancel all descendants and delete
its implementation, derivation, tests, launcher rows, runs, logs, reports,
checkpoints, and cached compiled artifacts. Preserve only one compact registry
tombstone containing its opaque slot, source hash, mechanism fingerprint,
terminal metrics, and rejection reason. The tombstone blocks reuse without
retaining token-heavy artifacts.

## Final retention

After recursive ablation and runtime closure, retain only:

- core trainer and shared utilities;
- both matched Muon control trajectories and signed reports;
- the three final optimizer sources and derivations;
- their final full-method, direct leave-one-out, pruned-parent, seed, dataset,
  and runtime evidence;
- manifests, cache hashes, source hashes, analyzers, and compact scheduler
  provenance.

All other candidate material is removed after validation. Deletions are
permanent within this workspace; recovery requires an external backup or a
prior commit.
