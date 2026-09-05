# Active work

- [x] Remove the discarded optimizer implementation, experiments, results,
  tests, paper draft, and generated artifacts from the live worktree.
- [x] Preserve original AdamW and Muon control records as provenance.
- [x] Port and hash-audit the complete Factorized Every-Step Robust
  Finite-Difference Gradient-Ledger Muon implementation.
- [x] Freeze and audit the 24-cell M0/M1 transfer matrix.
- [x] Launch all matched endpoints with the NVCC-built fused activation and
  peer-to-peer GPU communication enabled on unpinned four-A6000 NVLink-capable
  allocations (array `452320`, aggregate `452321`).
- [ ] Terminate any candidate row whose same-seed matched step-1,000 lead is
  negative.
- [ ] Report every completed or failed row with endpoint loss, absolute lead,
  lead retention, exact end-to-end total-time ratio, and scalability evidence.
