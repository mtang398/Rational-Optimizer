# ScalingOPT literature and implementation review

Reviewed September 23, 2026. This is our analysis of the public
[ScalingOPT community](https://tianshijing.github.io/ScalingOpt/), not an official
community report or a reproduction of its experiments.

- [Detailed review](REVIEW.md): mechanisms, inspected implementation differences,
  compatibility with fixed-SwiGLU TILLER and limits of the evidence.
- [Tracked ideas](IDEAS.md): ten candidates, hypotheses, priorities and decisions.
- [Optimizer screening](optimizer-screening.csv): 89 catalog entries, with the
  depth of review explicitly distinguished from catalog triage.
- [Paper index](paper-index.json): bibliographic links for 146 entries.
- [Pinned author-source references](author-source-manifest.json): repository
  commits, source URLs and hashes for six inspected repositories. Third-party
  source files and downloaded papers are not redistributed here.
- [Frozen source comparison](frozen-source-comparison.json) and
  [synthetic polar illustration](synthetic-polar-geometry.json): scope-specific
  evidence, not production measurements or independent reproductions.

MuonEq and HTMuon are candidate parent-direction interventions; Nexus supplies
a separate generalization hypothesis. None is implemented in the published
SwiGLU TILLER optimizer, and these notes do not demonstrate a cause of its late
loss-gap narrowing. Existing ablations should guide a later controlled test.
