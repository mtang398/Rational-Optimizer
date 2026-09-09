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
and the primary 18-layer, 1,024-wide model at 100M tokens and 3,050 steps. Unfinished endpoint
rows are represented with `pending` status and empty measurement fields.

The primary inventory contains 225 main runs and 15 runs with TILLER for
the first 1,000 updates followed by Muon. All 240 new rows are currently
pending. Their measurements will be updated as runs finish.

For the 18-layer study and the 12-layer 300M-token study, the comparison is
GRAIN + TILLER against SwiGLU + Muon. GRAIN + Muon is the activation-only
comparison. The 12-layer 100M-token study uses SwiGLU + AdamW as its control.

The 100M per-seed losses are retained at the six-decimal precision of the
completed experiment ledger; the 300M table retains the full-precision values
from the per-seed result artifacts. The `time_scope` column identifies the
boundary of each timing measurement: `end_to_end_process` records the full
training process, while `training_loop` records the trainer's loop time.
The separate `training_loop_total_seconds` column retains the loop measurement
when full-process timing is available.
