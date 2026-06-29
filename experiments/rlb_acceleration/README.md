# RLB Acceleration Worklog

This folder tracks same-method acceleration work for `RLB+MatrixPolicy`. The constraint is that changes should preserve the training rule: same Transformer, same RLB forward function, same MatrixPolicy update, same telemetry semantics unless a run explicitly changes a telemetry interval.

## Accepted Safe-Speed Change

The active same-method speed fix is in the production optimizer code: after every MatrixPolicy Muon group has permanently decayed to zero, `RationalMatrixPolicyOptimizer` skips the otherwise zero-LR Muon step. The early zero-LR warmup is not skipped, so the pre-active Muon state semantics are unchanged.

This is not a new optimizer method. It preserves the original `rational_matrix_policy_onpolicy` rule and only removes late no-op optimizer work.

## Verification Status

The safe Muon-off implementation fix passed the 500-step P0 speed check, completed the full 15-row E1 safe-speed rerun on 2026-06-23, and completed the full 15-row E2 safe-speed MatrixPolicy timing replacement on 2026-06-24. Those rows are retained as implementation-speed validation for the original local-atom MatrixPolicy path.

The current paper-facing MatrixPolicy result rows are now the corrected global-rational/no-local-atom replacement rows. On E1, the global-rational MatrixPolicy aggregate is `24.5` min, `0.4563` s/step, and `72,344.1` tokens/s over 15 rows, versus the earlier local-atom safe-speed aggregate of `27.3` min, `0.5102` s/step, and `67,078.3` tokens/s. That is about a 10% mean runtime reduction on E1.

On E2, the typical row is faster but the mean is not uniformly lower because three corrected global-rational rows are slow outliers. The global-rational per-dataset MatrixPolicy means are DCLM `67.7` min, FineWeb-Edu `65.6` min, FineWeb `74.7` min, Dolma-sample `75.0` min, and C4 `73.4` min. The median row time is lower than the local-atom safe-speed median (`67.1` min versus `71.2` min), while the overall mean is higher (`71.3` min versus `68.5` min). Do not claim a uniform E2 wall-clock speedup without the per-dataset/outlier caveat.

The larger `~0.09-0.10s` optimizer-step values in `phase_timing_summary.csv` are logged-step diagnostics. On log steps the training loop enables MatrixPolicy telemetry, and `optimizer_step_seconds` includes pre-step telemetry, weight snapshots, optimizer stepping, and post-step update telemetry. That field is useful for identifying telemetry/log-step cost, but it should not be read as average MatrixPolicy overhead.

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
