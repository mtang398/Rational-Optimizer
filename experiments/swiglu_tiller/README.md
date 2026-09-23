# SwiGLU + TILLER: code, results and ScalingOPT research

This folder publishes **`tiller_swiglu_fixed_v1`**, the GRAIN-free SwiGLU
specialization of TILLER, with a portable Muon comparison launcher and a
September 23, 2026 research snapshot. The model is a randomly initialized
**modern MHA decoder**, not a pretrained Qwen model. Qwen-derived implementation
and tokenizer provenance remain explicit.

| Contents | Description |
|---|---|
| [Optimizer](tiller_swiglu.py) / [numerical core](core.py) | Executed production optimizer, with local import paths |
| [Method specification](METHOD_FROZEN.md) | Joint gate/value role, response measurements, grouping, normalization and state |
| [Results](results/README.md) | Matched Muon/TILLER training loss, validation loss, perplexity and six-task benchmarks |
| [ScalingOPT review](scalingopt/README.md) | Literature/source review, 89-entry screening, 146-paper index and tracked ideas |
| [Source provenance](SOURCE_PROVENANCE.json) | Frozen source hashes and exact packaging changes |
| [Verification](VERIFICATION.md) | CPU checks, production evidence and portability limits |

## What this version does

SwiGLU has no learned GRAIN activation coefficients. Its current/reference
activation congruence is therefore **1**, and GRAIN-dependent response-drift
rotation is **0**. The Robust-FD loss-response ledger and equal-budget group
coordination remain active. This is an explicit specialization, not unchanged
GRAIN + TILLER and not a newly invented drift signal.

The MLP uses grouped gate/value/down parent directions, with gate and value
forming one incoming logical role. Attention uses the original public MHA
head-block optimizer, including four-head blocks and stacked-QKV `1/sqrt(3)`.
There is no GQA update, token-time correction, rational activation or GRAIN build
dependency. Rejected coordination falls back to these parents, which differ
from native whole-matrix Muon. Acceptance alone does not identify the source of
an observed training advantage.

Both the ledger and factorized histories use beta2 **0.95**; functional refreshes
occur at updates **1, 9, 17, ...**, with **K=32** and sketch rank **64**. Other
updates add gradient-surrogate rows. Parent momentum is **0.95**. Auxiliary
AdamW uses **(0.9, 0.95)**, epsilon **1e-8**, and no weight decay. Structural
weight decay is **0.1**; global clipping is applied once at norm **1**. The
optimizer core is copied from the accepted frozen source, not unpublished
working-copy changes. Candidate methods in the research notes are not enabled.

## Results included

Both reported **2,056,136,704-parameter** SwiGLU runs completed **11,445 updates /
3,000,238,080 loss-bearing tokens**. These are preliminary **single-seed** results;
the broader research and ablations remain incomplete.

| Final metric | SwiGLU + Muon | SwiGLU + fixed TILLER |
|---|---:|---:|
| Training-batch loss | 2.84995508 | 2.84553170 |
| Validation loss | 2.68069997 | 2.68043283 |
| Validation perplexity | 14.59530601 | 14.59140758 |

The earlier LM-loss lead largely disappears by 3B tokens. Downstream results
are mixed: several favor TILLER, while PIQA and ARC-Easy raw accuracy favor
Muon. All available raw/normalized metrics are retained. See the
[full tables](results/README.md), [per-100-step loss/PPL CSV](results/loss-ppl-every-100-steps.csv),
[every-step training CSV](results/training-every-step.csv), and
[both benchmark milestones](results/benchmarks.csv). PPL is exp(validation
cross-entropy); LAMBADA PPL is a different task-specific measurement.

![Learning curves](results/learning-curves.png)

## Environment and setup

The recorded production runtime used Python 3.12, PyTorch **2.11.0+cu128**, FP32
master weights and BF16 forward/backward autocast on four H100 GPUs. Dependencies
are pinned to that runtime; another PyTorch version can change native Muon or
compiled numerics. No activation CUDA extension is required for this package.

```bash
git clone --branch main https://github.com/mtang398/Rational-Optimizer.git
cd Rational-Optimizer
python3.12 -m venv .venv-swiglu
source .venv-swiglu/bin/activate
python -m pip install -r experiments/swiglu_tiller/requirements.txt
```

## Prepare data on CPU

The original experiments use pinned FineWeb-Edu `sample-100BT`, a pinned Qwen
tokenizer, and prestaged lossless little-endian uint32 caches. No pretrained
weights or live data streaming are used during training.

For exact data-order comparisons, supply the **same verified cache** and a JSON
manifest containing `dtype: "<u4"`, the pinned `tokenizer_revision`, and
`splits.train` / `splits.validation` entries with `path`, `tokens`, and `sha256`.
The tokenizer revision is in [config.json](config.json). Paths may be absolute
or relative to the manifest. Both conditions must use the same manifest.

```bash
python -m experiments.swiglu_tiller.prepare \
  --import-manifest /data/existing/manifest.json --output /data/swiglu
```

The preparation helper also accepts `--fineweb` or `--jsonl ...`. Those options
create a **new dataset selection**, not a reconstruction of the original cache.
They support deterministic document-hash splits, normalized exact deduplication
and optional overlap screening; they do not reproduce the campaign's full
near-duplicate filtering. Reproducing the published data identity requires the
original verified cache. Raw corpus documents and token caches are not uploaded.

## Launch training in an existing allocation

Default: 2.056B parameters, 32 layers, width 2048, sixteen equal Q/K/V heads,
head dimension 128, SwiGLU width 6144, context 2048, seed 1337. Four ranks ×
microbatch 2 × accumulation 16 give **262,144 loss tokens per update**. Tied
embeddings, Q/K normalization, RoPE and RMSNorm are retained.

Run separately, with distinct output directories:

```bash
bash experiments/swiglu_tiller/train_tiller.sh \
  --data /data/swiglu/manifest.json --output /runs/swiglu-tiller
bash experiments/swiglu_tiller/train_muon.sh \
  --data /data/swiglu/manifest.json --output /runs/swiglu-muon
```

These scripts invoke `torchrun`; they do not request or submit a cluster job.
The portable trainer is newly packaged CPU-tested code. The included GPU
results were produced by the frozen campaign trainer using the same numerical
optimizer, not by rerunning these launchers.

Both retain LR **3e-4**, **200-update warmup**, and the **45,776-update cosine
horizon** with minimum ratio **0.1**. Stopping at 3B tokens does not compress
the schedule. Validation runs every 100 updates and at the final update;
checkpoints run every 250 updates and at completion/continuation boundaries.
The two most recent periodic checkpoints are retained; final checkpoints remain.

For a short 1.092B-parameter / 1000-step run, use
`--config experiments/swiglu_tiller/configs/1b_1000_steps.json`. That config uses
28 layers, width 1536, head dimension 96, SwiGLU width 4608, microbatch 4 and
accumulation 8. It is a configuration for the same two optimizer choices, not
the four-cell production ablation suite.

## Checkpoints, resumption and limits

```bash
# Example chunk; select a walltime suitable for your already allocated resources.
bash experiments/swiglu_tiller/train_tiller.sh \
  --data /data/swiglu/manifest.json --output /runs/swiglu-tiller --max-hours 4
# Resume in the same source/environment/configuration and rank topology:
bash experiments/swiglu_tiller/train_tiller.sh \
  --data /data/swiglu/manifest.json --output /runs/swiglu-tiller \
  --resume latest --max-hours 4
```

The default margin is 15 minutes; actual filesystem checkpoint time must fit.
Checkpoints bind the source/config/data, model, all optimizer children, counters,
each rank's RNG and exact next-batch identity. Rank count and microbatch layout
cannot change across resume. A lock rejects concurrent writers. Abandoned log
records after a restored checkpoint are archived before replay. Load only your
own trusted checkpoints. Portable checkpoint envelopes differ from the campaign;
do not point these scripts at active campaign output directories.

Supported optimizer integration: this decoder, FP32 parameters, optional BF16
autocast and ordinary DDP. No FSDP, activation checkpointing/reentrant forwards,
GradScaler, GQA or arbitrary Hugging Face wrapper support is claimed. The public
optimizer constructor accepts a DDP-wrapped model and requires masks for the
whole update **before the first forward**. Its `backward()` performs normalization;
do not divide again by accumulation. See the method specification's example.

Training outputs include losses, actual tokens, validation PPL, child LRs,
proposal acceptance and coefficient statistics. Production per-guard rejection
reasons were not recorded; no cause is fabricated here. The new pending ablation
has separate guard instrumentation. Optional spectral telemetry behavior of this
published optimizer follows its original frozen source.

The portable launcher computes LM validation only. The six-task scores in this
folder come from the campaign's matched benchmark evaluator, with its complete
protocol/asset/checkpoint identities in the result JSON files. This launcher does
not automatically run that benchmark suite or open the sealed 12B LM test.

## CPU checks and plotting

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m unittest discover -s experiments/swiglu_tiller -t . -p 'test_*.py' -v
# Optional figure regeneration:
python -m pip install matplotlib
python -m experiments.swiglu_tiller.plot_results
```

The optimizer tests include the real recurrence, probe normalization, actual
mask changes, complete state restoration, original MHA attention composition,
and four CPU/Gloo ranks. CPU evidence does not promise bitwise replay across
different GPU architectures, versions or distributed execution orders.
