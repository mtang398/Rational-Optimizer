# RLB Acceleration Worklog

This folder tracks same-method acceleration work for `RLB+MatrixPolicy`. The constraint is that changes should preserve the training rule: same Transformer, same RLB forward function, same MatrixPolicy update, same telemetry semantics unless a run explicitly changes a telemetry interval.

## CPU-Only Changes Started

- MatrixPolicy now skips stat-factor work when the scheduled Muon mix is exactly zero. This preserves the update because the old value was `0 * stat_factor`, but avoids needless policy/stat computations outside the active Muon window.
- MatrixPolicy reuses the Adam-side Muon fraction when setting the matching Muon optimizer group LR. The Adam and Muon group lists are constructed from the same parameter groups in the same order, so this avoids recomputing the same schedule/stat value.
- RLB optimizer-stat collection now caches the deterministic sample index for a fixed activation batch shape and sample count instead of rebuilding the same `torch.linspace(...).long()` index every stat pass.

These are implementation optimizations only. They do not change the RLB activation, MatrixPolicy formulas, LR schedule, or manifest rows.

## CPU-Only Profiling Helper

`summarize_phase_timing.py` reads existing JSONL train events and writes a phase timing CSV. It does not import torch and does not touch GPUs.

```bash
python3 experiments/rlb_acceleration/summarize_phase_timing.py \
  --phase E1_m0_100m \
  --phase E2_m0_300m
```

Default output:

```text
experiments/rlb_acceleration/phase_timing_summary.csv
```

## Next GPU Checks When GPUs Free

1. Run a short fixed-seed CPU/GPU parity smoke if a GPU is available: one RLB+MatrixPolicy row for a tiny token budget, comparing pre/post loss curves and optimizer telemetry.
2. Profile CUDA time by phase with synchronized timers already logged as `forward_backward_seconds` and `optimizer_step_seconds`.
3. If optimizer overhead remains high, inspect MatrixPolicy for additional CPU/GPU syncs from `.item()` calls in active policy paths.
4. If forward/backward dominates, profile the fused local-basis CUDA kernel and consider kernel-level changes behind a parity test.
