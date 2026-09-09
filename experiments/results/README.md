# Experiment results

Results are grouped by optimizer. Each optimizer directory keeps all available
model scales, datasets, activation choices, token budgets, and seeds in the
same set of tables.

- `runs.csv` contains one row per training run.
- `summary.csv` contains one unique
  `(model_scale, dataset, train_tokens, activation, optimizer)` cell.
- `checkpoints.csv` is included when checkpoint-level validation measurements
  were retained.

Machine-readable identifiers and reader-facing names are stored separately in
`activation` / `activation_display_name`, `optimizer` /
`optimizer_display_name`, and `method` / `method_display_name`. Blank fields
mean that the source record did not contain that measurement. A non-finite
endpoint remains visible in `runs.csv` and is labelled `non_finite`; it is not
included in a numeric endpoint average.

The tables cover the 12-layer, 768-wide model at 100M and 300M training tokens
and the primary 18-layer, 1,024-wide model at 300M tokens. Unfinished endpoint
rows are represented with `pending` status and empty measurement fields.

The primary 225-cell inventory currently contains 34 complete runs, 2
incomplete runs, and 189 pending runs. The incomplete rows remain visible with
their completed-step counts and recorded validation checkpoints; they are
excluded from endpoint aggregates.

For both 300M-token TILLER studies, the comparison is GRAIN + TILLER against
SwiGLU + Muon. GRAIN + Muon is the activation-only decomposition baseline.
The 100M-token study uses SwiGLU + AdamW as its control.

The 100M per-seed losses are retained at the six-decimal precision of the
completed experiment ledger; the 300M table retains the full-precision values
from the per-seed result artifacts. The `time_scope` column identifies the
boundary of each timing measurement: `end_to_end_process` records the full
training process, while `training_loop` records the trainer's loop time.
The separate `training_loop_total_seconds` column retains the loop measurement
when full-process timing is available.
