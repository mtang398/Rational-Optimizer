# Experiments

The active experiment package validates **Factorized Every-Step Robust
Finite-Difference Gradient-Ledger Muon, version 1** across the locked M0 and M1
transfer suites.

## Evidence rules

- Build every candidate row from the original AdamW or Muon control cell.
- Permit differences only in activation, optimizer identity, run/output name,
  and the audit contract that binds the complete implementation.
- Keep LR, minimum LR, WD and parameter routing, betas, epsilon, clipping,
  schedule, initialization, data tokens and order, model, batching, evaluation,
  diagnostics, and seeds exactly matched.
- Use generic four-RTX-A6000 requests without a named-node pin.
- Verify the NVCC-built fused activation and enable peer-to-peer GPU
  communication.
- Record topology rather than selecting a particular node.
- Stop a candidate whose matched step-1,000 lead is negative.
- Count only completed endpoints as quality evidence.

## Matrix

- M0: five datasets x three seeds x two arms, 3,050 steps, AdamW control.
- M1: three datasets x three seeds x two arms, 9,150 steps, Muon control.
- Total: 24 paired cells and 48 endpoint trajectories.

Reports include endpoint loss, absolute lead, lead retention, exact end-to-end
total-time ratio, and scalability evidence for every row.
