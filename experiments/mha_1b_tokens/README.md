# Modern MHA decoder: four 1B-token training runs

This is **1B training tokens, approximately 537M parameters**, not a 1B-parameter model.
The modern MHA decoder is randomly initialized. All four conditions use
28 layers, width 1024, 16 Q heads and 16 KV heads, head dimension 64, Q/K RMSNorm,
RoPE, RMSNorm, tied embeddings, and context 2048. Qwen is used only for tokenizer and implementation-reference provenance.

| Launcher | Model / optimizer | Exact parameter count |
|---|---|---:|
| `train_muon.sh` | SwiGLU + PyTorch Muon | 537,326,080 |
| `train_adamw.sh` | SwiGLU + AdamW | 537,326,080 |
| `train_tiller.sh` | GRAIN + original complete TILLER | 537,331,120 |
| `train_tiller100_muon.sh` | GRAIN + TILLER for updates 1–100, then Muon | 537,331,120 |

Counts come from actual module parameter inventories, excluding duplicate tied
embedding references. SwiGLU width is 3072; GRAIN width is 4608, 18 groups of 256.
Attention/norm/embedding initialization is explicitly copied and checked across
activations; both GRAIN runs have identical full initialization. Seed is 1337.

The MHA optimizer calls the repository's **unchanged `TILLERRouter` and
`TILLERAttentionOptimizer`**, including response-guided attention, four-head polar
blocks, stacked equal QKV and its `1/sqrt(3)` normalization. There is no GQA
extension and no token-time modification in this folder. Native Q/K/V parameters
are packed only for the reference optimizer and unpacked after its update.

This portable folder depends on `optimizer_design/` and `activation/` from the
**same repository commit**. Clone the repository, not just this folder. It has no
TACC paths, account, submission controller, or campaign ledger dependencies.
The shell scripts launch training in an allocation you already obtained; they do
not call `sbatch`. Existing Stampede3 campaign jobs must still use their own
reviewed submission bridge.

## Setup

Use Linux, Python 3.12+, a CUDA compiler compatible with the pinned PyTorch CUDA
12.8 build, and BF16-capable GPUs. Four H100s match the original batch geometry;
other hardware is not performance-validated. Build on a host with CUDA toolkit
access; set `CUDA_HOME` and `TORCH_CUDA_ARCH_LIST` as appropriate for your hardware.

```bash
git clone --branch portable-mha-1b-tokens https://github.com/mtang398/Rational-Optimizer.git
cd Rational-Optimizer
python3 -m venv .venv-mha
source .venv-mha/bin/activate
bash experiments/mha_1b_tokens/setup.sh
export PYTHONPATH="$PWD/activation:$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

Setup installs the repository's pinned dependencies and builds the rational CUDA
extension. It does not fetch pretrained model weights. CUDA GRAIN training refuses
the CPU rational fallback. CPU tests explicitly enable that reference path.

## Prepare local data before allocating GPUs

The tokenizer is `Qwen/Qwen3-0.6B-Base` at revision
`da87bfb608c14b7cf20ba1ce41287e8de496c0cd`.
The optional FineWeb-Edu source is `HuggingFaceFW/fineweb-edu`, `sample-100BT`,
revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`.
Preparation downloads tokenizer assets only, never pretrained weights.

For exact comparisons with an existing run, copy its verified token caches and
manifest, then import the manifest with its paths corrected to the local files:

```bash
python -m experiments.mha_1b_tokens.prepare \
  --import-manifest /data/existing/manifest.json --output /data/mha-import
```

Import verifies SHA256, uint32 size, token counts and vocabulary range. The imported
manifest must have `splits.train` and `splits.validation` entries with `path`,
`sha256`, and optionally `tokens`, and identify the pinned tokenizer with `tokenizer_revision` or `tokenizer.revision`. Import does
not retokenize, shuffle or copy the large cache. Keep those referenced files intact.

Alternatively, prepare a **new dataset selection** on CPU:

```bash
python -m experiments.mha_1b_tokens.prepare --fineweb --output /data/mha-1b
# Or use ordered UTF-8 JSONL files, each line containing {"text": "..."}:
python -m experiments.mha_1b_tokens.prepare \
  --jsonl /data/documents-001.jsonl /data/documents-002.jsonl --output /data/mha-local
```

This starter assigns normalized-content hash splits before selection, removes
normalized exact duplicates using SQLite, appends EOS, and writes lossless little
endian uint32 caches with checksums and ordered document identities. Every target,
including EOS, bears loss; packing allows attention across document boundaries.
The final selected document can be truncated to the requested cache length. Train
cache defaults to 1,000,079,361 tokens (one extra input token); fixed validation
uses 131,072 loss-bearing tokens. No test set is opened by training.

**New preparation is not the campaign's full curated data pipeline.** It does not
remove near-duplicates. Supply `--exclude-jsonl` with downstream evaluation texts
for normalized 13-word overlap screening; otherwise overlap screening is absent.
These limitations and removal counts are recorded in the manifest. Use the same
imported campaign caches for direct comparisons to campaign results. Do not claim
campaign-identical data or uncontaminated benchmarks from a fresh starter cache.
A failed preparation leaves partial files and will not overwrite them; retry in a
new output directory. Tokenizer/network access is needed only during preparation.

## Start each condition

Inside a four-GPU allocation, with the Python environment active:

```bash
bash experiments/mha_1b_tokens/train_muon.sh \
  --data /data/mha-1b/manifest.json --output /runs/mha/swiglu_muon
bash experiments/mha_1b_tokens/train_adamw.sh \
  --data /data/mha-1b/manifest.json --output /runs/mha/swiglu_adamw
bash experiments/mha_1b_tokens/train_tiller.sh \
  --data /data/mha-1b/manifest.json --output /runs/mha/grain_tiller
bash experiments/mha_1b_tokens/train_tiller100_muon.sh \
  --data /data/mha-1b/manifest.json --output /runs/mha/grain_tiller100_muon
```

Each command is one separate logical run and uses four GPUs by default. Run them
sequentially in one allocation or in separate allocations. Do not run multiple
launchers concurrently on the same GPUs/output directory. A trainer lock and
identity checks reject conflicting output reuse.

`NPROC_PER_NODE` changes the process count. For example, one GPU with the same
global batch needs `NPROC_PER_NODE=1` and `--accumulation 16` with microbatch 8.
`--microbatch`/`--accumulation` must preserve 262,144 loss tokens per update.
Changing that decomposition changes the run identity and is not supported across
a checkpoint resume. Memory fit and throughput must be measured on your hardware.

Default target: **3815 updates = 1,000,079,360 loss-bearing tokens**. Defaults retain
the 45,776-update full-horizon cosine schedule, 200-update warmup, LR 0.0003 and
minimum LR ratio 0.1. This is the first 1B-token prefix of the 12B schedule, **not**
a cosine schedule compressed to 1B. All conditions use that same schedule.

Structural matrices have weight decay 0.1. Muon uses momentum 0.95, five
Newton–Schulz iterations and `match_rms_adamw`. Original TILLER retains beta2 0.95,
32 functional probes, sketch rank 64 and refreshes at updates 1, 9, 17, ...;
non-refresh updates still add gradient-surrogate rows to the ledger. Auxiliary
AdamW uses betas (0.9, 0.95), eps 1e-8, and no decay for embeddings, normalization
vectors and GRAIN coefficients. Global gradient norm is clipped once to 1.0.

The switch run keeps GRAIN throughout. At update 101 it transfers exact parent
momentum, including splitting packed QKV into native matrices, retains auxiliary
AdamW moments/counters, and removes only TILLER capture hooks. It does not reset
weights, data, RNG, LR or the schedule. Checkpoints record the optimizer phase.

## Checkpoint and resume

```bash
# Four-hour useful chunk with a 15-minute checkpoint margin:
bash experiments/mha_1b_tokens/train_tiller.sh \
  --data /data/mha-1b/manifest.json --output /runs/mha/grain_tiller --max-hours 4
# Next allocation, same environment, source, config, data and number of ranks:
bash experiments/mha_1b_tokens/train_tiller.sh \
  --data /data/mha-1b/manifest.json --output /runs/mha/grain_tiller \
  --resume latest --max-hours 4
```

Checkpoints are written to temporary directories, SHA256-verified, and atomically
published. They include model, every optimizer child, TILLER state, rank RNG,
completed update, exact next batch hash and source/config/data identities. Only
completed updates count. On resume, any logged records after the restored update
(and a truncated final log line) are preserved in `metrics-abandoned-*.jsonl` and
removed from the canonical metrics prefix before replay. Resume restores the same full-horizon schedule. Keep the
output directory with its identity, resolved config and latest pointer; load only
trusted local checkpoints. No automatic scheduler submission/retry occurs.

Periodic checkpoints are every 250 updates; two recent periodic checkpoints are
retained. Switch checkpoints 100/101 and final checkpoints are permanent. The
walltime margin must exceed the measured time to finish an update and write a
checkpoint on your filesystem. A hard kill can replay work after the last durable
checkpoint; such work is not counted as completed progress.

## Outputs and validation

`metrics.jsonl` records each step, actual/cumulative loss-bearing tokens, training
loss, gradient norm, LR for each optimizer child, phase, functional-refresh flag,
optimizer time and GPU peak memory. Validation runs every 100 updates and at the
final update, using a fixed local prefix; TILLER capture is disabled during
validation and RNG restored. `startup.json` records exact parameter counts and
shared initialization hashes. `result.json` distinguishes a partial chunk from a
completed 1B target.

Validation is next-token LM loss, not a six-task benchmark suite. This starter does
not automatically download or run HellaSwag/PIQA/ARC/WinoGrande/BoolQ/LAMBADA.
Use a GRAIN-aware model adapter if adding benchmarks; loading GRAIN into an
unchanged Hugging Face SwiGLU model is invalid. No benchmark scores are included.

## Tests and scope of verification

```bash
RATIONAL_OPT_TORCH_FALLBACK=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m unittest discover -s experiments/mha_1b_tokens -t . -p 'test_*.py' -v
# Four CPU/Gloo ranks; tiny model/data, no GPU allocation:
python -m experiments.mha_1b_tokens.test_distributed
```

Tests cover upstream same-config SwiGLU forward/loss/gradient parity, an independent
GRAIN functional reference, original attention polar normalization, complete TILLER
routing/cadence, exact first-100 switch equivalence, resume at 99/100/101, auxiliary
state, verified checkpoint corruption detection, token-ID round trips and
cross-rank data positions. CPU evidence does not establish CUDA extension or
hardware performance on another system. No GPU job was submitted for packaging.

Single-process resume tests require bitwise equality. Four-process resume checks
require exact counters/RNG/next-batch identities and floating tensors within
`atol=1e-7, rtol=2e-5`; distributed reductions can change floating-point operation
order after restart. All optimizer replicas within a run must agree exactly.

For the Robust-FD persistent score factor only, distributed resume compares
`S.T @ S`, not arbitrary eigendecomposition row signs. Diagonal/tail history and
the decay cross term are checked separately; no optimizer state is dropped.

The four-process exact-state diagnostic uses a CPU-only rank-ordered gradient
reduction helper (`test_worker.py`) to control floating-point summation after DDP
reconstruction. Production uses ordinary DDP. Uncontrolled Gloo comparisons
showed up to about 3.1e-6 weight drift after the switch, with unchanged data and
refresh counters; bitwise replay across distributed restarts is not promised.
No reduction helper or CPU rational fallback is used for CUDA training.
