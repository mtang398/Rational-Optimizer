# Reproduction protocol

This directory is the executable specification for the activation–optimizer
suite and the paired TILLER evaluation.

## Experiment inventory

| Suite | Model | Datasets | Seeds | Tokens | Steps | Matched control |
|---|---|---|---|---:|---:|---|
| `12l_768d_100m_tokens_3050_steps` | 12 layers, $d=768$, 12 heads, $H=2048$ | DCLM, FineWeb-Edu, FineWeb, Dolma sample, C4 | 1337, 2027, 3407 | 100M | 3,050 | SwiGLU + AdamW |
| `12l_768d_300m_tokens_9150_steps` | 12 layers, $d=768$, 12 heads, $H=2048$ | DCLM, FineWeb-Edu, FineWeb, Dolma sample, C4 | 1337, 2027, 3407 | 300M | 9,150 | SwiGLU + Muon |
| `18l_1024d_300m_tokens_9150_steps` | 18 layers, $d=1024$, 16 heads, $H=3072$ | DCLM, FineWeb-Edu, FineWeb, Dolma sample, C4 | 1337, 2027, 3407 | 300M | 9,150 | SwiGLU + Muon |

All endpoint suites process 32,768 tokens per optimizer step with sequence
length 256 and four data-parallel ranks. The 12-layer suites use batch size 16
per GPU with two accumulation microbatches; the 18-layer suite uses batch size
8 per GPU with four accumulation microbatches.

## Shared training configuration

The selected TILLER rows use learning rate `3e-4`, minimum
learning rate `3e-5`, 200 warmup steps, cosine decay, weight decay `0.1`, betas
`(0.9, 0.95)`, epsilon `1e-8`, gradient clipping `1.0`, and initialization
standard deviation `0.02`. Evaluation runs every 50 steps for 10 batches.
GRAIN uses SiLU coefficient initialization, identity post-rational
initialization, group size 256, at most 32 groups, and statistics cadence 4.

Every activation comparison inherits the complete optimizer-specific row,
including its learning-rate schedule and optimizer arguments. The 100M-token
TILLER suite matches the established AdamW parameter routing, including decay
on the tied embedding. The 300M-token TILLER suite matches the established
Muon routing, with the tied embedding in the no-decay AdamW group.

The complete values are columns in `activation_optimizer_manifest.csv`; the
selected paired rows are materialized in `matrix.json`.

## Datasets

| Label | Dataset | Configuration | Validation source |
|---|---|---|---|
| DCLM | `mlfoundations/dclm-baseline-1.0` | `none` | training stream after the reserved token offset |
| FineWeb-Edu | `HuggingFaceFW/fineweb-edu` | `sample-10BT` | training stream after the reserved token offset |
| FineWeb | `HuggingFaceFW/fineweb` | `sample-10BT` | training stream after the reserved token offset |
| Dolma sample | `allenai/dolma` | `v1_6-sample` | training stream after the reserved token offset |
| C4 | `allenai/c4` | `en` | validation split |

For training-stream validation, the offset is 210M tokens in the 100M suite
and 610M tokens in the 300M suite. The 100M-token suite evaluates on 4M tokens;
the 300M-token suite evaluates on 8M tokens. The tokenizer is `gpt2`.

## Files

```text
build_activation_optimizer_manifest.py  creates the 640-row sweep manifest
activation_optimizer_manifest.csv       complete activation–optimizer rows
build_matrix.py                         selects the 45 TILLER/control pairs
matrix.json                             paired TILLER execution matrix
token_fingerprints.json                 expected token-cache sample hashes
run_activation_optimizer_sweep.sbatch   one manifest row per array task
run_activation_row.py                   manifest-to-trainer command builder
run_quality_row.sbatch                  paired control/TILLER endpoint runner
suite.py                                paired-row construction and identity checks
method_entrypoint.py                    TILLER composite optimizer wiring
collect_results.py                      compact result-table extraction
analyze_row.py                          matched endpoint and total-time analysis
verify_repository.py                    source, manifest, and result validation
test_suite.py                           protocol regression tests
```

## Validate the repository

```bash
.venv/bin/python -m experiments.protocol.build_activation_optimizer_manifest --print-summary
.venv/bin/python -m experiments.protocol.build_matrix
.venv/bin/python -m experiments.protocol.verify_repository
.venv/bin/python -m pytest
```

## Launch

The primary suite is partitioned into four stages. Each stage runs at most two
array tasks concurrently. Run the Muon activation pairs first, then TILLER,
then the AdamW activation pairs, followed by the remaining optimizers:

```bash
suite=18l_1024d_300m_tokens_9150_steps
sbatch --array=0-29%2 \
  experiments/protocol/run_activation_optimizer_sweep.sbatch muon "$suite"
```

After the Muon array is terminal, submit TILLER:

```bash
suite=18l_1024d_300m_tokens_9150_steps
sbatch --array=0-14%2 experiments/protocol/run_quality_row.sbatch "$suite"
```

Inspect the TILLER ledger before promoting the next stage. A recorded negative
step-1,000 screen is a terminal scientific result; an execution failure is
retryable.

```bash
.venv/bin/python -m experiments.protocol.analyze_all
```

Then submit AdamW, followed by the remaining optimizers after AdamW is
terminal:

```bash
suite=18l_1024d_300m_tokens_9150_steps
sbatch --array=0-29%2 \
  experiments/protocol/run_activation_optimizer_sweep.sbatch adamw "$suite"
```

```bash
suite=18l_1024d_300m_tokens_9150_steps
sbatch --array=0-149%2 \
  experiments/protocol/run_activation_optimizer_sweep.sbatch remaining "$suite"
```

The same stage commands reproduce the 12-layer, 300M-token suite after
changing `suite`. For the 12-layer, 100M-token suite, run the AdamW stage before
TILLER because AdamW is its selected control. The ten preflight rows use:

```bash
sbatch --array=0-9%2 experiments/protocol/run_activation_optimizer_sweep.sbatch all \
  12l_768d_preflight_2621440_tokens_80_steps
```

Timing rows request an exclusive allocation with four RTX A6000 GPUs and
NVLink, enable NCCL peer-to-peer transport, and do not select a named node.
The activation–optimizer stages produce the selected controls; each TILLER
task consumes its control trajectory and runs the candidate. Hardware,
topology, source hashes, terminal endpoint, total wall time, and the
step-1,000 screen are recorded with the row.

## Collect compact results

```bash
.venv/bin/python -m experiments.protocol.collect_results \
  --activation-runs experiments/runs/activation_optimizer \
  --tiller-runs experiments/protocol/runs \
  --tiller-analysis experiments/protocol/results \
  --output-root experiments/results
```

Raw JSONL logs, checkpoints, token caches, and compiled binaries are generated
locally. The collector pairs every GRAIN row with the SwiGLU row carrying the
same optimizer, dataset, seed, model, token budget, and training schedule. It
writes one compact run table, summary table, and checkpoint table per optimizer
under `experiments/results/`; each TILLER row is joined to the control selected
by the same matrix row.
