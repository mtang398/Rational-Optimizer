# RLB Acceleration Worklog

This folder tracks same-method acceleration work for `RLB+MatrixPolicy`. The constraint is that changes should preserve the training rule: same Transformer, same RLB forward function, same MatrixPolicy update, same telemetry semantics unless a run explicitly changes a telemetry interval.

## Accepted Safe-Speed Change

The active same-method speed fix is in the production optimizer code: after every MatrixPolicy Muon group has permanently decayed to zero, `RationalMatrixPolicyOptimizer` skips the otherwise zero-LR Muon step. The early zero-LR warmup is not skipped, so the pre-active Muon state semantics are unchanged.

This is not a new optimizer method. It preserves the original `rational_matrix_policy_onpolicy` rule and only removes late no-op optimizer work.

## Verification Status

The fix passed the 500-step P0 speed check, completed the full 15-row E1 safe-speed rerun on 2026-06-23, and completed the full 15-row E2 safe-speed MatrixPolicy timing replacement on 2026-06-24. The E1 quality deltas versus the original MatrixPolicy table are within seed/dataset noise. The current paper-facing E1 runtime table now has 15 runs for every optimizer/activation combo; the safe-speed MatrixPolicy aggregate is `27.3` min, `0.5102` s/step, and `67,078.3` tokens/s.

The E2 safe-speed replacement completed with `Restarts=0` for all clean rows after the non-NVLink odd-chain row was cancelled and replaced. Current MatrixPolicy E2 runtime ranges from `67.0` to `69.6` minutes by dataset, with mean s/step from `0.4141` to `0.4304`. The generated runtime summary and all E2 dataset packages now use those safe-speed rows.

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
