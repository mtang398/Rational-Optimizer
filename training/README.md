# Training

This directory contains the shared causal-Transformer training path used by
every manifest row.

```text
train.py                 model, data, optimization, evaluation, and JSONL logging
baseline_optimizers.py   Lion, SOAP, AdEMAMix, CAME, and Schedule-Free AdamW
exact_resume.py          atomic distributed checkpoint and trajectory recovery
run.sbatch               general four-GPU activation–optimizer launcher
aggregate_results.py     per-seed and aggregate result-table builder
```

`train.py` also provides AdamW and Muon. GRAIN and SwiGLU select different
feed-forward modules while sharing the same attention, normalization,
embedding, data, evaluation, and logging code.

## Prepare data

The trainer streams the dataset named on the command line, tokenizes it with
the specified tokenizer, and stores an `int32` cache under `experiments/cache/`.

```bash
.venv/bin/python training/train.py \
  --prepare-only \
  --dataset-name mlfoundations/dclm-baseline-1.0 \
  --dataset-config none \
  --dataset-streaming \
  --dataset-text-column text \
  --train-split train \
  --validation-split train \
  --max-train-tokens 100000000 \
  --max-val-tokens 4000000 \
  --validation-skip-tokens 210000000 \
  --tokenizer gpt2
```

The experiment launchers build the complete command from
`experiments/protocol/activation_optimizer_manifest.csv` or
`experiments/protocol/matrix.json`; these are the reference entrypoints for
reproducing reported rows.

## Aggregate a run directory

```bash
.venv/bin/python training/aggregate_results.py \
  --run-dir experiments/runs/activation_optimizer/<phase>/<dataset> \
  --out-dir /tmp/rationalopt-summary
```
