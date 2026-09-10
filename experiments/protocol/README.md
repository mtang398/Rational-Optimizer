# Reproduction protocol

This directory connects each activation–optimizer experiment to its data,
training command, checkpoints, endpoint loss, and measured training time.

## Experiment inventory

| Suite | Model | Nominal training tokens | Steps | TILLER comparison |
|---|---|---:|---:|---|
| `12l_768d_100m_tokens_3050_steps` | 12 layers, width 768, 12 heads, intermediate width 2048 | 100M | 3,050 | SwiGLU + AdamW |
| `12l_768d_300m_tokens_9150_steps` | 12 layers, width 768, 12 heads, intermediate width 2048 | 300M | 9,150 | SwiGLU + Muon |
| `18l_1024d_100m_tokens_3050_steps` | 18 layers, width 1024, 16 heads, intermediate width 3072 | 100M | 3,050 | SwiGLU + Muon |

Each endpoint suite covers DCLM, FineWeb-Edu, FineWeb, Dolma sample, and C4,
with seeds 1337, 2027, and 3407. The primary 18-layer suite contains 210 runs
of SwiGLU/GRAIN with seven baseline optimizers, 15 full TILLER runs, and
15 runs using TILLER for updates 1–1,000 followed by Muon for updates
1,001–3,050.

Every update processes 32,768 tokens at sequence length 256 across four
data-parallel ranks. The 12-layer model uses batch size 16 per GPU and two
accumulation microbatches; the 18-layer model uses batch size 8 and four
microbatches. Thus 3,050 steps process 99,942,400 training tokens. The 18-layer
SwiGLU and GRAIN models contain 296,867,840 and 296,871,080 parameters,
respectively.

## Configuration and data

`activation_optimizer_manifest.csv` records every shared training argument.
`matrix.json` selects the 60 TILLER/control pairs, including the two-stage
schedule. Learning rate, weight decay, initialization, data order, batching,
and evaluation follow the selected manifest row.

The TILLER rows use learning rate `3e-4`, minimum learning rate `3e-5`,
200 warmup steps, cosine decay, weight decay `0.1`, betas `(0.9, 0.95)`,
epsilon `1e-8`, gradient clipping `1.0`, and initialization standard deviation
`0.02`. Validation runs every 50 steps for 10 batches and at the endpoint.
The two-stage schedule preserves compatible momentum and AdamW state at the
switch and continues the same learning-rate schedule.

| Dataset label | Hugging Face dataset | Configuration |
|---|---|---|
| DCLM | `mlfoundations/dclm-baseline-1.0` | `none` |
| FineWeb-Edu | `HuggingFaceFW/fineweb-edu` | `sample-10BT` |
| FineWeb | `HuggingFaceFW/fineweb` | `sample-10BT` |
| Dolma sample | `allenai/dolma` | `v1_6-sample` |
| C4 | `allenai/c4` | `en` |

Dataset and GPT-2 tokenizer revisions are pinned in the manifest. The
100M-token suites use a 4M-token validation cache; the 300M-token suite uses
8M. Training-stream validation starts at token offsets 210M and 610M,
respectively. C4 uses its validation split.

## Prepare and verify

Run these commands from the repository root after the installation steps in
the [main README](../../README.md):

```bash
.venv/bin/python -m experiments.protocol.build_activation_optimizer_manifest
.venv/bin/python -m experiments.protocol.build_matrix
.venv/bin/python -m experiments.protocol.verify_repository
.venv/bin/python -m pytest
.venv/bin/python -m experiments.protocol.prepare_data \
  --suite 18l_1024d_100m_tokens_3050_steps
```

Data preparation runs once before training. Use `--dataset dclm` to prepare
one dataset; omit it to prepare all five. The same command accepts either
12-layer suite.

## Submit the primary suite

The five arrays below encode the execution order and allow up to three
concurrent four-GPU runs. Submit from the repository root:

```bash
suite=18l_1024d_100m_tokens_3050_steps
muon_job=$(sbatch --parsable --array=0-29%3 --time=02:00:00 \
  experiments/protocol/run_activation_optimizer_sweep.sbatch muon "$suite")
tiller_job=$(sbatch --parsable --array=0-14%3 --time=02:00:00 \
  --dependency="afterok:$muon_job" \
  experiments/protocol/run_quality_row.sbatch "$suite" tiller_v1)
switch_job=$(sbatch --parsable --array=0-14%3 --time=02:00:00 \
  --dependency="afterany:$tiller_job" \
  experiments/protocol/run_quality_row.sbatch "$suite" tiller_then_muon_v1)
adamw_job=$(sbatch --parsable --array=0-29%3 --time=02:00:00 \
  --dependency="afterany:$switch_job" \
  experiments/protocol/run_activation_optimizer_sweep.sbatch adamw "$suite")
sbatch --array=0-149%3 --time=03:00:00 \
  --dependency="afterany:$adamw_job" \
  experiments/protocol/run_activation_optimizer_sweep.sbatch remaining "$suite"
```

The 12-layer suites use the same launchers with their suite identifiers.
Run AdamW before TILLER for the 12-layer 100M-token suite, and Muon before
TILLER for the 12-layer 300M-token suite. Allocate a longer wall limit for
9,150-step runs.

Each launch requests four RTX A6000 GPUs on an NVLink-capable allocation
without selecting a named node. The runtime audit permits node co-residency
only while the GPU partition remains non-oversubscribed and the exact
CPU/GPU/RAM/NVLink contract is preserved. The runner records hardware and
topology, then measures the complete training process. TILLER consumes the
completed matched control and records its step-1,000 comparison and final
endpoint.
A candidate with a negative step-1,000 lead is stopped and retained with its
observed trajectory.

## Collect results

```bash
.venv/bin/python -m experiments.protocol.collect_results --include-pending \
  --activation-runs experiments/runs/activation_optimizer \
  --tiller-runs experiments/protocol/runs \
  --tiller-analysis experiments/protocol/results \
  --output-root experiments/results
```

Each optimizer directory contains `runs.csv`, `summary.csv`, and
`checkpoints.csv`. Model size, token budget, dataset, seed, activation, and
optimizer identify every result. The collector retains the published
12-layer results while updating new runs.

The main implementation files are `run_activation_row.py` (baseline command
construction), `suite.py` (paired command construction),
`method_entrypoint.py` (optimizer integration), `analyze_row.py` (paired
loss and timing), and `collect_results.py` (compact publication tables).
Large checkpoints and token caches are generated locally.
