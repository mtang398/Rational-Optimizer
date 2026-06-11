# RLB Acceleration Worklog

This folder tracks same-method acceleration work for `RLB+MatrixPolicy`. The constraint is that changes should preserve the training rule: same Transformer, same RLB forward function, same MatrixPolicy update, same telemetry semantics unless a run explicitly changes a telemetry interval.

## Candidate Ideas Only

The production activation and optimizer code has been restored. This folder is only a scratch area for analysis, timing summaries, and future candidate patches until a GPU A/B benchmark proves a change is safe and useful.

Candidate same-method ideas to revisit later:

- Skip MatrixPolicy stat-factor work only when the scheduled Muon mix is exactly zero.
- Reuse the Adam-side Muon fraction when setting the matching Muon optimizer group LR, if group ordering is explicitly asserted.
- Cache deterministic RLB optimizer-stat sample indices for fixed activation batch shapes.

None of these candidate implementation changes are active in the main method code right now.

The isolated candidate patch is:

```text
experiments/rlb_acceleration/candidate_patches/same_method_optimizer_overhead.patch
```

That patch is for future review/A-B testing only. Do not apply it to paper runs unless a GPU benchmark first verifies correctness and speed.

## Verification Status

No real throughput speedup is verified yet because no GPU benchmark was run. The existing runtime table is the trustworthy average-step source: clean E2 DCLM has `RLB+MatrixPolicy` at `0.5293 s/step` and `RLB+AdamW` at `0.5188 s/step`, about `+2.0%`. Clean E1 has `RLB+MatrixPolicy` at `0.6032 s/step` and `RLB+AdamW` at `0.5945 s/step`, about `+1.5%`.

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
