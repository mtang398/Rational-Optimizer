# Global GPU Queue Contract

This scheduling contract supplements the source-frozen optimizer fairness
contract without changing any frozen method or experiment package.

- Every GPU job requests exactly four generic NVIDIA RTX A6000 GPUs.
- At most two GPU jobs may run at once: eight A6000 GPUs total.
- The complete user queue, not each package separately, is organized as two
  global dependency lanes.
- Every pending GPU job preserves its own scientific dependency and also has
  one earlier predecessor in its assigned global lane.
- Sibling package branches may not become independently runnable merely
  because their local predecessor completed.
- After every submission or cancellation, audit the pending GPU dependency
  graph.  It may have at most two roots, and those roots must be the
  successors of the two active lane heads.
- If a third four-GPU job starts, stop the newest job before scientific
  training, repair dependencies with `scontrol`, and resubmit the identical
  frozen command.
- Holds, owner-release helpers, exact-node pins, preferred nodes, excluded
  nodes, and NVLink requirements are forbidden.  Scheduling repair changes
  only dependencies, never optimizer math, seed, data, LR, WD, or allocation
  size.

On 2026-07-30, all 261 pending GPU jobs were topologically ordered while
preserving their existing `afterany` edges, then assigned one additional
predecessor in one of two global lanes rooted at jobs 633617 and 636876.
The post-repair graph had exactly two pending roots: 633619 and 636878.
After 636878 exited before training during the sealer-manifest
administrative repair, its identical replacement 639539 was inserted after
636880 and before 636882.  This preserves the second serial GPU lane without
changing the intervention.

Run the read-only check after every queue mutation:

```bash
.venv-cu128/bin/python experiments/audit_global_gpu_lanes.py
```
