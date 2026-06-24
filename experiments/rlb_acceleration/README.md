# RLB Acceleration Worklog

This folder tracks same-method acceleration work for `RLB+MatrixPolicy`. The constraint is that changes should preserve the training rule: same Transformer, same RLB forward function, same MatrixPolicy update, same telemetry semantics unless a run explicitly changes a telemetry interval.

## Accepted Safe-Speed Change

The active same-method speed fix is in the production optimizer code: after every MatrixPolicy Muon group has permanently decayed to zero, `RationalMatrixPolicyOptimizer` skips the otherwise zero-LR Muon step. The early zero-LR warmup is not skipped, so the pre-active Muon state semantics are unchanged.

This is not a new optimizer method. It preserves the original `rational_matrix_policy_onpolicy` rule and only removes late no-op optimizer work.

## Verification Status

The fix passed the 500-step P0 speed check and then completed the full 15-row E1 safe-speed rerun on 2026-06-23. The E1 quality deltas versus the original MatrixPolicy table are within seed/dataset noise, while the clean JSONL runtime aggregate improved from the prior clean E1 RLB+MatrixPolicy row (`32.0` min, `0.6032` s/step, `55,759.6` tokens/s over 14 clean rows) to `27.3` min, `0.5102` s/step, and `67,078.3` tokens/s over 15 clean rows.

The larger `~0.09-0.10s` optimizer-step values in `phase_timing_summary.csv` are logged-step diagnostics. On log steps the training loop enables MatrixPolicy telemetry, and `optimizer_step_seconds` includes pre-step telemetry, weight snapshots, optimizer stepping, and post-step update telemetry. That field is useful for identifying telemetry/log-step cost, but it should not be read as average MatrixPolicy overhead.

E2 safe-speed timing is now queued separately under `E2_matrixpolicy_safe_speed_300m` as jobs `810092`-`810106`; once complete, the runtime table should replace the original E2 MatrixPolicy timing rows the same way E1 does.

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
2. Profile CUDA time by phase with non-telemetry microbenchmarks, because the existing logged `optimizer_step_seconds` field includes log-step telemetry overhead.
3. If optimizer overhead remains high, inspect MatrixPolicy for additional CPU/GPU syncs from `.item()` calls in active policy paths.
4. If forward/backward dominates, profile the fused local-basis CUDA kernel and consider kernel-level changes behind a parity test.
