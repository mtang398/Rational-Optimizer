# Experiments

Generated experiment artifacts live here.

## Layout

```text
cache/    Hugging Face dataset cache and tokenized WikiText-103 tensors
runs/     active JSONL run files and Slurm logs
results/  aggregate CSV/JSON summaries produced after runs finish
```

## Active Run

The active comparison directory is:

```text
experiments/runs/wikitext103/rlb_optimizer_empirical_ngram_full/
```

Completed jobs in this run:

```text
763059   baseline and first optimizer sweep
813929   rational_quotient_onpolicy extension
821187   rational_jacobian_onpolicy extension
```

Important logs:

```text
experiments/runs/logs/ract-wt103-opt-763059.out
experiments/runs/logs/ract-wt103-opt-813929.out
experiments/runs/logs/ract-wt103-opt-821187.out
```

Aggregate output:

```text
experiments/results/rlb_optimizer_empirical_ngram_full/
```

Current best aggregate row:

```text
rational_jacobian_onpolicy + rlb_fused_fixed_strong_ffn
mean loss 3.605394, PPL 36.800, sec/step 0.204885
```

## Cleanup Policy

Keep dataset/token caches while active experiments need them:

```text
experiments/cache/
```

Keep active run JSONL files, Slurm logs, and aggregate results:

```text
experiments/runs/logs/
experiments/runs/wikitext103/rlb_optimizer_empirical_ngram_full/
experiments/results/rlb_optimizer_empirical_ngram_full/
```

Remove repo-local generated artifacts after compile/test runs:

```text
build/
training/__pycache__/
optimizer_design/__pycache__/
activation/rational_opt/__pycache__/
```
