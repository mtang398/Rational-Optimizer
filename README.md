# RationalOPT

RationalOPT contains two coupled Transformer components:

- **GRAIN** — Groupwise Rational Activation with Internal Normalization.
- **TILLER** — Tangents Informed by a Loss Ledger for Equal-budget Reweighting.

GRAIN replaces a SwiGLU feed-forward block with a parameter-matched grouped
rational block. TILLER uses loss responses collected across GRAIN groups to
coordinate their structured update directions under a fixed global update
budget.

## Repository map

```text
activation/          GRAIN and SwiGLU definitions, Python operators, CUDA kernels
optimizer_design/    TILLER mathematics and implementation
training/            Shared Transformer trainer and baseline optimizers
experiments/         Manifests, launchers, validation, and compact results
paper/               ICLR 2027 manuscript source, official style, and rendered PDF
```

Every experiment is generated from a manifest row. The row records the model,
dataset slice, seed, token budget, optimizer, activation, learning-rate
schedule, weight decay, batching, initialization, evaluation cadence, and all
method arguments used by the shared trainer.

## Install

The reference environment uses Python 3.12, PyTorch 2.11.0 with CUDA 12.8,
and four NVIDIA RTX A6000 GPUs for the reported distributed runs.

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
.venv/bin/python setup.py build_ext --inplace
```

The CUDA build creates `rational_opt._C` inside `activation/rational_opt/`.

## Verify

```bash
.venv/bin/python -m experiments.protocol.build_activation_optimizer_manifest --print-summary
.venv/bin/python -m experiments.protocol.build_matrix
.venv/bin/python -m experiments.protocol.verify_repository
.venv/bin/python -m pytest
```

## Reproduce

For the primary 18-layer suite, submit Muon activation pairs first, followed
by TILLER, AdamW, and the remaining activation–optimizer pairs:

```bash
suite=18l_1024d_300m_tokens_9150_steps
sbatch --array=0-29%2 \
  experiments/protocol/run_activation_optimizer_sweep.sbatch muon "$suite"
```

After that array is terminal, submit the TILLER, AdamW, and remaining stages
in order using the commands in
[experiments/protocol](experiments/protocol/). Review the recorded step-1,000
screens before advancing to AdamW.

Each TILLER row reuses the completed control selected by `matrix.json` and runs
the candidate under the same four-GPU RTX A6000/NVLink execution standard.
The launcher records the assigned hardware and topology, checks the source
manifest, and writes endpoint and timing records.
See [experiments/protocol](experiments/protocol/) for the complete command and
[experiments/results](experiments/results/) for the compact result tables.
The method paper and its TeX Live 2025/Overleaf build instructions are in
[paper](paper/).
