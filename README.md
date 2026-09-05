# RationalOPT

RationalOPT studies loss-aware coordination of structured update directions for
Rational Latent Basis Transformer feed-forward layers.

The active optimizer is **Factorized Every-Step Robust Finite-Difference
Gradient-Ledger Muon, version 1**. The repository carries the complete optimizer
implementation and its exact experiment entrypoint; no reduced proxy is used.

## Existing evidence

- M0/DCLM/3,050 steps, three-seed mean: `4.337422212` versus SwiGLU+AdamW
  `4.405600000`, for a final lead of `+0.068177788`.
- M1/FineWeb-Edu/9,150 steps: `3.649184942` versus SwiGLU+Muon `3.690944910`,
  for a final lead of `+0.041759968`.
- Qualifying matched end-to-end timing ratio: `1.049116783x`.

These are historical source records. The active campaign requires newly
completed matched endpoints for final transfer claims.

## Locked transfer campaign

- M0: DCLM, FineWeb-Edu, FineWeb, Dolma-sample, and C4; seeds `1337`, `2027`,
  and `3407`; 3,050 steps; control SwiGLU+AdamW.
- M1: DCLM, FineWeb-Edu, and C4; the same seeds; 9,150 steps; control
  SwiGLU+Muon.
- LR `3e-4`, minimum LR `3e-5`, warmup `200`, weight decay `0.1`, betas
  `(0.9, 0.95)`, epsilon `1e-8`, clipping `1.0`, initialization, schedule,
  model, batching, token order, evaluation, and diagnostics are inherited
  exactly from the corresponding original control cell.
- Candidate hyperparameters are frozen to the successful implementation. No
  transfer-specific tuning is permitted.

Every row reports endpoint loss, absolute matched lead, step-1,000 lead,
end-to-end total time, exact total-time ratio, and scalability evidence. A
candidate run is terminated if its matched lead is negative at step 1,000; a
partial trajectory never counts as endpoint evidence.

## Scalability contract

The method is owner-free. It requires no complete-layer ownership,
owner-local mathematics, state proportional to total activation positions,
dense `(LG) x (LG)` materialization, dense cubic solve in `LG`, or
parameter-sized selected-update publication. Its declared persistent state is
`O(LH + LGd + 64LG)`, with fixed-size transaction solves.

## Repository map

```text
activation/                 Rational Latent Basis activations and fused kernels
optimizer_design/           Complete optimizer implementation and audits
training/                   Matched language-model training harness
experiments/                Locked manifests, launchers, reports, and raw records
RLB_OPTIMIZER_FAIRNESS_CONTRACT.md
REPO_STORAGE_POLICY.md
```

Large token caches and virtual environments remain local and ignored. New
campaigns reuse shared token caches rather than duplicating them.
