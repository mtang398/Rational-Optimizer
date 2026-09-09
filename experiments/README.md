# Experiments

This directory connects declarative experiment rows to the shared trainer and
stores compact, reviewable results.

```text
protocol/   manifests, matrix builders, launchers, checks, and collectors
results/    checkpoints, endpoints, timing, and summaries grouped by optimizer
```

The activation–optimizer manifest covers AdamW, Muon, Lion, SOAP, AdEMAMix,
CAME, Schedule-Free AdamW, TILLER, and the TILLER-to-Muon schedule. Each optimizer has its
own directory under `results/`; model scale, dataset, seed, activation, and
training budget remain columns rather than directory levels, so later suites
can extend the same schema.

The published inventory spans a 12-layer, 768-wide model at 100M and 300M
training tokens, plus the primary 18-layer, 1,024-wide model at 100M tokens
and 3,050 steps.
Every endpoint suite covers DCLM, FineWeb-Edu, FineWeb, Dolma sample, and C4
with seeds 1337, 2027, and 3407. The protocol README gives the exact inventory
and launch commands.
