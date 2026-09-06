# Factorized Every-Step Robust Finite-Difference Gradient-Ledger Muon transfer

This is a fresh, matched endpoint campaign. It does not tune the method, LR,
weight decay, schedules, initialization, data order, or any shared training
hyperparameter.

## Locked matrix

- M0/100M: DCLM, FineWeb-Edu, FineWeb, Dolma-sample, and C4; seeds 1337,
  2027, and 3407; 3,050 steps; SwiGLU+AdamW control.
- M1/300M: DCLM, FineWeb-Edu, and C4; the same seeds; 9,150 steps;
  SwiGLU+Muon control.
- Both paired arms use LR `3e-4`, minimum LR `3e-5`, warmup `200`, weight
  decay `0.1`, betas `(0.9, 0.95)`, epsilon `1e-8`, gradient clip `1.0`, the
  same cosine schedule, initialization, token caches, batch order, evaluation
  cadence, and all other shared parsed arguments.
- M0 preserves the original GitHub diagnostic settings: probe batch `1` and
  matrix-spectrum interval `250`. M1 preserves the exact established
  9,150-step diagnostic settings: `0` and `0`. Both scales preserve telemetry
  cadence `4`. Diagnostics are identical between each paired control and
  candidate.

## Execution and acceptance

Each matrix row runs its fresh control and fresh candidate endpoint on one
exclusive four-RTX-A6000 allocation. The scheduler may choose any eligible
node; no node or exclusion list is named. The node must advertise NVLink, all
four selected ranks must have direct peer access, `NCCL_P2P_DISABLE=0`,
`NCCL_P2P_LEVEL=NVL`, and shared-memory transport enabled. This uses direct
P2P only across actual NVLink peers and shared memory across the remaining
pairwise paths, avoiding the cluster's hanging PCIe/CUMEM P2P route. The
NVCC-built rational extension must load with PyTorch fallback disabled.

This campaign does not repeat the already-passed short timing admission gate.
The exact end-to-end candidate/control total-training-time ratio comes from
the two complete endpoints. A ratio at or below `1.05x` is required. While the
candidate endpoint runs, its paired step-1,000 validation loss is checked; a
negative lead terminates only that candidate row and is recorded as a
scientific failure. There is no separate 1,000-step screening run.

## Storage and scale

Token caches remain shared in place. Each of the 48 trajectories has a unique
directory; interrupted attempts are archived and no cache is duplicated.
The exact method is owner-free, stores no activation-position-sized method
state, publishes no parameter-sized selected update, and creates no dense
`(LG)×(LG)` object. Its declared state is `O(LH + LGd + 64LG)` and its dense
operations have fixed maximum dimension 96.
