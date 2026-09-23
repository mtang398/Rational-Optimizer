# Fixed-SwiGLU TILLER research tracker

Updated September 23, 2026. These are research candidates, not authorized training
conditions. Keep existing jobs, frozen releases, scientific settings and ablations
unchanged. Evaluate the four existing 1.09B SwiGLU ablations before selecting a
new experiment. No GRAIN condition is proposed here.

Detailed source review: [ScalingOPT review](REVIEW.md).
The same directory contains the 89-entry screening CSV, 146-paper index, pinned
author-code hashes, frozen-source comparison and synthetic geometry illustration.
Coverage is explicitly distinguished from independent reproduction.

| ID | Idea | Priority / status | Specific question | Decision requirement |
|---|---|---|---|---|
| OPT-01 | [MuonEq](https://arxiv.org/html/2603.28254v2): conditioning before polar | First inexpensive candidate; awaiting ablations | Does balancing inside existing groups improve the parent direction? | Preserve grouping, NS5, momentum, scale compensation and auxiliary AdamW; distinguish new parent alone from parent plus coordination. |
| OPT-02 | [HTMuon](https://arxiv.org/html/2603.10067v1): spectral shaping | First late-stage spectral hypothesis; awaiting evidence | Does retaining within-block singular-value differences improve later finite-step loss? | Check real checkpoint directions and matched update budgets; a 1000-step early result cannot establish a late-training fix. |
| OPT-03 | [NorMuon](https://arxiv.org/html/2510.05491v1) / [Muon+](https://arxiv.org/html/2602.21545v3): post-polar balancing | Conditional | Is there meaningful row/column imbalance after our existing block polar? | Check axes and overlap first. NorMuon requires distinct polar-update history; do not reuse gradient-moment state. |
| OPT-04 | [Aurora](https://github.com/tilde-research/aurora-release): leverage balancing | Conditional, especially down projections | Do tall down blocks benefit from balanced row participation? | Author implementation leaves wide/square blocks on ordinary polar; account for multiple polar evaluations. |
| OPT-05 | [SOAP–Muon](https://nikhilvyas.github.io/SOAP_Muon.pdf) / [Newton–Muon](https://arxiv.org/html/2604.01472v1) | Later, more expensive | Does added local curvature or input geometry complement group allocation? | Account for covariance/basis states and computation; existing K32 probes are not automatically a sufficient covariance estimator. |
| OPT-06 | [MARS-M](https://github.com/AGI-Arena/MARS/tree/main/MARS_M) / [MONA](https://arxiv.org/html/2605.26842v2) | Conditional temporal direction | Can gradient-history correction improve the parent independently of coordination? | Distinguish exact/approximate gradient differences, extra state, clipping and raw-gradient acceptance semantics. |
| OPT-07 | [TEON](https://arxiv.org/html/2601.23261v2): cross-layer tensor geometry | Conditional on attention-parent ablation | Does alternative matrix assembly improve attention optimization? | Cross-layer tensorization differs from our head blocks and changes response coupling; important prior art. |
| OPT-08 | [Nexus](https://arxiv.org/html/2604.09258v1): microbatch gradient agreement | Longer-term generalization direction | Can downstream gains persist at comparable pretraining loss? | Inner parameter motion changes our fixed-state accumulation/probe assumptions; not a drop-in wrapper. |
| OPT-09 | [SGG](https://arxiv.org/html/2506.01049v1) / [Magma](https://arxiv.org/html/2602.15322v1): allocation/masking | Comparator before combination | Would a different allocation rule explain gains beyond parent geometry? | Overlaps with coordination; do not append masks/scales after acceptance or relax guards to raise acceptance. |
| OPT-10 | [Gram Newton–Schulz](https://dao-lab.ai/blog/2026/gram-newton-schulz/) | Systems candidate | Can the same selected polynomial run faster? | Verify BF16 direction/acceptance parity and stability; orthogonalization speedup is not whole-training speedup. |

Secondary reading retained in the full review: Muon², AdaMuon, SPECTRA,
Mano/weight-manifold methods, and hyperparameter transfer for blocked matrix
optimizers. They remain screened alternatives, with no implementation scheduled.

## Shared interpretation and integration rules

- Fixed-SwiGLU activation congruence is one; GRAIN-dependent drift rotation is
  zero. Grouped MLP and head-block attention parents still differ from native Muon
  when coordination rejects a proposal.
- Construct any new direction before functional measurements, surrogate rows,
  energies, decay cross terms and acceptance calculations. Protect the true loss
  gradient from in-place transformations used by third-party optimizers.
- Preserve auxiliary AdamW, data, initialization, learning-rate horizon and token
  accounting in a controlled comparison. Record any new history state and test
  checkpoint restoration explicitly.
- Author code contains non-obvious defaults and arithmetic differences; consult
  the pinned-code review rather than copying optimizer classes wholesale.
- Neither rejection rate nor spectra alone establish a cause or a useful fix.
  Compare actual loss outcomes. No root cause has been established by this review.

## Decision after the existing four-cell ablation

1. Identify the effects of grouped MLP, head-block attention, both parents and
   coordination relative to the reused native-Muon baseline.
2. Inspect all failed proposal predicates, including layer/role contractions,
   predicted gain, budget residual, hard cases and nonfinite arithmetic.
3. Select one complementary mechanism and one explicit question. Do not combine
   several candidates in the first test.
4. A claim that coordination adds value to a new parent needs that parent both
   alone and with coordination. A hybrid beating native Muon does not isolate it.

## Tracking history

- 2026-09-23: Initial literature/source review completed; ideas recorded at the
  user's request. All candidates are unimplemented/unsubmitted in this campaign.
  Existing ablation is the decision input, not a test of these new combinations.
