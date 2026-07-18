# MatrixPolicy Live-Statistic Correction Campaign

## Scope

Two implementation defects invalidate every result whose optimizer consumes
live RLB response/derivative gains under distributed training:

1. each rank previously applied its rank-local live gains after DDP had reduced
   gradients, allowing the RLB matrix replicas to diverge; and
2. validation forwards previously refreshed the optimizer-consumed cache, so a
   later training update could depend on held-out activations.

The corrected implementation uses training-only sufficient statistics,
globally sums squared responses, squared derivatives, and sample counts in one
fixed-layout collective, and reconstructs identical weighted RMS gains on every
rank. Other diagnostic moments remain rank-zero telemetry. Telemetry-only RLB
runs retain their previous behavior.

## Required Execution

| Block | Coverage | Jobs |
| --- | --- | ---: |
| Canonical main, 100M tokens | 5 datasets x 3 seeds | 15 |
| Canonical main, 300M tokens | 5 datasets x 3 seeds | 15 |
| E8 MatrixPolicy sensitivity | 5 datasets x 16 learning-rate/weight-decay cells | 80 |
| Corrected E9 preflight | A0-A9, 80 steps each | 10 |
| Corrected E9 scientific study | 10 arms x 5 datasets x 3 seeds | 150 |
| **Total executions** |  | **270** |

The defect directly changes 200 scientific identities: 30 main rows, 80 E8
rows, and the 90 E9 rows in A3/A5/A6/A7/A8/A9. The complete 150-row E9 study
is rerun because its protocol requires all arms to share one frozen runtime;
mixing controls from the old freeze with corrected treatments would violate
that contract.

The 210 ordinary global-RLB optimizer controls do not consume these live gains
and remain valid. Superseded E0/E3, safe-speed, V2-V6, and exploratory ablation
rows are not rerun. The duplicate rational-only manifest is not counted as an
additional result set.

## Execution Contract

The campaign uses a content-addressed, read-only runtime snapshot; isolated
manifests and output roots; persistent workers that process disjoint manifest
rows; four RTX A6000 GPUs on an NVLink-labelled node; and staged validation
barriers. There is no runtime-speed exclusion. Known unsuitable nodes are
excluded by identity.

Submission is gated in this order:

```text
four-GPU NCCL correctness gate
  -> ten corrected E9 preflights
  -> 30 main runs
  -> 80 E8 runs
  -> 150 E9 runs
  -> final coverage/provenance validation
```

Each stage validator checks coverage, frozen hashes, GPU/node metadata, final
attempt timing, finite metrics, token/index fingerprints, and expected live
stat scope. The preflight additionally checks rank equality of every RLB matrix
pair after each optimizer step and across an evaluation boundary.

## Status

The corrected CPU/Gloo tests pass. The first submitted freeze
`016d60f78ffeb3ecffce53420be14551462eb83554dca2c7e5a4dd39ff7b8e3b`
stopped at gate job `844708` before any scientific run started. Its environment
fingerprint included node-local system packages because `.venv-cu128` exposes
system site packages; consequently, a valid A6000 node produced a different
host-package inventory. The failed freeze and runtime are preserved under
`submissions/failed_freeze_016d60f_20260712_051448/`.

The corrected freeze contract records the complete virtualenv-local stack and
the recursive dependency closure of the training libraries, together with the
Python ABI and PyTorch CUDA version. Host utilities are excluded. The new
immutable freeze is
`925e3615588e61a26cf1de8f95d14375058c4e2c4c3d36196b4ad2285fab52a1`.
Submission on 2026-07-12 created 281 Slurm jobs, comprising 270 row jobs, one
NCCL gate, five validators, and five failure watchdogs. The exact ledger is
`submissions/submission_20260712_163526.csv`; gate job `848496` precedes the
campaign and final validator job `848771` closes it. All 274 row/validator
dependencies were verified after the four-chain rewire recorded in
`submissions/concurrency_rewire_16gpu_20260712_163654.csv`.

The 30 corrected main rows passed `validation/main.json` on 2026-07-14 with
complete E1/E2 coverage, finite metrics, frozen provenance, and zero Slurm
restarts. They are now the paper-facing MatrixPolicy source for all fixed-scale
tables, curves, target-arrival readouts, and observed runtime summaries.

The 80 corrected E8 rows passed `validation/e8.json` on 2026-07-15 with
complete five-dataset, four-learning-rate, four-weight-decay coverage and no row
errors. Workers `858357`-`858360` completed the sweep and validator
`858361` closed the stage. Forty final JSONLs report restart count zero and
forty report restart count one; all recorded runtimes are final-attempt harness
times. The paper sensitivity analysis uses these 80 MatrixPolicy trajectories
and only the 160 unaffected SiLU controls from the original E8 manifest.

The E9 stage remains incomplete. The cluster outage left 93 of 150 rows valid;
the exact 57-row recovery is recorded in
`submissions/e9_resume_20260717_221003.csv` and its JSON companion. Recovery
workers `9193`-`9196`, validator `9197`, final validator `9198`, and
watchdog `9199` are queued. No E9 result enters the paper before both
validators pass.

Gate job `848496` passed on `seo-compute-02` with four identical ranks,
training-only evaluation-cache behavior, and the expected frozen hashes. Three
80-step preflights completed in 73, 86, and 101 seconds before the queue was
converted from one-row allocations to persistent workers.

An intermediate conversion caused the old `afterany` validators (`848507`,
`848538`, `848619`, and `848770`) to run after their temporary worker
dependencies were cancelled. They correctly failed on missing coverage, and no
downstream scientific row started. These are orchestration failures rather
than experimental results.

The active autonomous DAG is recorded in
`submissions/persistent_worker_dag_20260712_225700.csv` and its JSON companion.
One preflight worker completes all ten rows, skipping the three terminal rows.
After its validator passes, four main workers process all 30 corrected main
rows without releasing their allocations between rows. Four E8 workers and
four E9 workers follow behind their own validators. Workers preserve terminal
rows across requeue, archive an interrupted current row, and rerun only that
row with final-attempt timing. The active worker-launcher SHA-256 is
`7195369d4b362e4b09c9c0a7904c84e5d00d8a3de2d34002bb53e5c9f60b4fbd`.
The DAG contains 22 jobs, never exceeds four concurrent four-GPU workers, and
has worker-aware failure watchdogs at every stage.

The persistent preflight worker `853056` started on `ellis-compute-02` at
05:17 EDT on 2026-07-13 and completed the seven missing rows in nine minutes.
All ten preflight rows pass row-level validation with finite metrics, matching
frozen provenance, and no terminal failure. CPU validator `853057` initially
failed before reading the rows because strict environment verification expanded
node-local system-package metadata on `snavely-cpu-09`. Its watchdog correctly
cancelled all downstream jobs before a main run started.

CPU-side validation now verifies the immutable runtime, manifest, cache, freeze,
and output provenance against the environment recorded in the freeze, while GPU
training launchers retain strict executing-host environment verification. The
corrected validator passed the full 10/10 preflight report on the same
`snavely-cpu-09` node in diagnostic job `858350`. The resumed downstream DAG is
recorded in `submissions/persistent_worker_dag_20260713_102809.csv` and its JSON
companion. Main workers `858352`-`858355` are immediately eligible; E8 and E9
remain behind their stage validators. Their pending allocation ceilings were
tightened to 8, 12, and 20 hours from observed row runtimes, as recorded in
`submissions/worker_timelimit_adjustment_20260713_103158.csv`; these are maximum
allocation lengths rather than delayed start times.

The user granted a one-time 16-GPU concurrency allowance for this campaign on
2026-07-12. The queued training DAG was consequently rewired from two to four
independent four-GPU chains without changing any runtime, manifest, output, or
validation artifact. The rewire was applied only while all affected jobs were
pending, and all 274 row/validator dependencies were verified afterward. Its
complete before/after record is
`submissions/concurrency_rewire_16gpu_20260712_163654.csv`. The resulting hard
cap is four simultaneous training jobs, or 16 GPUs; stage barriers and failure
watchdogs remain unchanged.
