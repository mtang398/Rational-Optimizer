# MatrixPolicy Fast Pilot Protocol

Status: active pilot policy. `matrixpolicyV7`, `matrixpolicyV8_fastpulse`, and `matrixpolicyV9_approx_muon` P0 all failed on 2026-06-22 and were pruned from the active repo surface. V7 slightly improved loss but was slower; V8 tested no-new-code shorter Muon pulse schedules and worsened loss/AUC; V9 tested lower Muon Newton-Schulz accuracy inside the original MatrixPolicy window and worsened loss/AUC for only negligible same-node speed. Current active use: method-preserving safe-speed P0 jobs `727990`-`727992`, validating commit `02b85d9`, V10 switch-clean P0 jobs `728006`-`728007`, V11 state-adaptive beta2 P0 jobs `728025`-`728026`, and V12 selector-beta2 candidate job `728038`. V10-V12 keep full-quality early MatrixPolicy conditioning and test post-Muon Adam-state geometry, using existing optimizer flags only.

## Problem

The previous workflow sent a full E1 candidate test too early. A full candidate E1 costs 15 jobs at about 100M tokens each. That is acceptable for a serious final check, but it is too expensive for V3/V4/V5-style idea search.

The faster pilot should be paired against V1, short, and split into independent jobs so preemption does not destroy a long run.

## Stage P0: Mechanism Smoke

Purpose: prove the code path and telemetry work. This is not result evidence.

Shape:

```text
1 dataset: DCLM
1 seed: 1337
2 methods: V1 and candidate
300-500 steps
validation every 50 steps
one row per Slurm job
```

Pass only if:

```text
candidate JSONL logs all required mechanism telemetry
telemetry is not saturated or constant
runtime is <= 1.05x paired V1 unless the loss/AUC gain is obvious
no NaNs, no missing final validation, no skipped candidate path
```

## Stage P1: Short Paired Pilot

Purpose: reject weak ideas before full E1.

Shape:

```text
2 datasets: DCLM and FineWeb-Edu
2 seeds: 1337 and 2027
2 methods: V1 and candidate
600-800 steps, about 20M-26M tokens per row
validation every 50 steps
8 independent Slurm jobs
```

Why these two datasets: V5 only improved FineWeb-Edu while losing/tying elsewhere, so DCLM plus FineWeb-Edu is a quick check for both the robust anchor and the easiest false-positive dataset.

Primary gate:

```text
candidate validation-loss AUC <= paired V1 mean AUC on both datasets
candidate final validation loss <= paired V1 + 0.001 on both datasets
candidate beats V1 by at least 0.001 mean final loss or AUC on one dataset
mechanism telemetry matches the proposal's claimed behavior
```

Failing either dataset is enough to stop. Do not rationalize a full E1 from one lucky seed.

## Stage P2: Medium Confirmation

Run only after P1 passes.

Shape:

```text
5 datasets: DCLM, FineWeb-Edu, FineWeb, Dolma-sample, C4
1 seed: 1337
2 methods: V1 and candidate
1200-1600 steps, about 40M-52M tokens per row
10 independent Slurm jobs
```

Pass only if:

```text
candidate beats paired V1 AUC on at least 4/5 datasets
candidate final loss is not worse than V1 by more than 0.0015 on any dataset
runtime <= 1.05x V1, or token-to-target improvement clearly compensates
telemetry remains non-saturated after warmup
```

## Stage P3: Full E1

Run a full E1 candidate only after P2 passes.

Shape:

```text
5 datasets x 3 seeds candidate rows
100M tokens per row
one manifest row per Slurm job
no dependency chain longer than same-parity row groups
```

If V1 comparison noise is suspected, include fresh paired V1 rows for the affected dataset/seed instead of relying only on historical V1.

## Why This Is Faster

A bad idea should usually die in P0 or P1:

```text
P0: 2 short jobs
P1: 8 short jobs
P2: 10 medium jobs only if P1 passes
Full E1: 15 full jobs only if P2 passes
```

This reduces the common failure case from 15 full 100M-token jobs to 2-8 short jobs, and every job is independently requeueable. There should be no 36-hour monolithic run.

## Mandatory Rule

Do not queue a full E1 for V6, V7, or any later version until the proposal has a signed mathematical claim, telemetry proving the claim is active, and a paired P1 result that beats or ties V1 under the gates above.
