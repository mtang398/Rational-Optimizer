# ICLR Optimizer Experiment Blueprint

This blueprint explains the evidence standard. The exact run matrix is `ICLR_EXACT_RUN_PLAN.md`.

The paper should copy the experiment style of accepted optimizer papers: Sophia, SOAP, AdamW, Lion, Adam-mini, GaLore, CAME, Schedule-Free, AdEMAMix, and Fantastic Pretraining Optimizers. The center is optimizer evidence:

```text
speed-to-target
final-budget loss
model/data scaling
batch-size and overhead behavior
memory and throughput
cross-corpus transfer
long-horizon and forgetting behavior
strong baseline protocol-lock and sensitivity maps as reviewer defense
mechanism diagnostics as support
ablations last
```

## Claim

```text
MatrixPolicy is a stronger optimizer for RLB Transformer training than generic AdamW/Muon and modern optimizer-family baselines, giving a better loss-vs-compute frontier at academic LM-pretraining scale while remaining stable, transferable, and efficient enough to justify its overhead.
```

The current 3-seed FineWeb/FineWeb-Edu result is a pilot signal. It motivates the paper, but the paper must be won by the new experiments in `ICLR_EXACT_RUN_PLAN.md`.

## Accepted-Paper Standards

| paper | venue | standard RationalOPT must match |
| --- | --- | --- |
| AdamW, Loshchilov & Hutter | ICLR 2019 | LR/WD landscapes and clear separation of optimizer, schedule, and regularization effects. |
| Lion, Chen et al. | NeurIPS 2023 | Broad task transfer, batch-size behavior, compute/memory behavior, and limitations. |
| CAME, Luo et al. | ACL 2023 | Convergence, stability, and memory footprint for memory-efficient adaptive baselines. |
| Sophia, Liu et al. | ICLR 2024 | GPT-style pretraining at multiple scales; steps/tokens/compute/wall-clock to target loss. |
| GaLore, Zhao et al. | ICML 2024 oral | Optimizer-state memory, feasibility, and training curves at meaningful model scale. |
| Schedule-Free, Defazio et al. | NeurIPS 2024 | Horizon robustness and comparisons that do not rely on one stopping point. |
| SOAP, Vyas et al. | ICLR 2025 | LM pretraining, matrix/preconditioner baselines, batch-size/precondition-frequency sensitivity, wall-clock. |
| Adam-mini, Zhang et al. | ICLR 2025 | Memory, throughput, role/block behavior, and optimizer-state footprint. |
| AdEMAMix, Pagliardini et al. | ICLR 2025 | Long-horizon token efficiency and forgetting/distribution-shift behavior. |
| Cautious Optimizers, Liang et al. | ICLR 2026 | Consistent gains under controlled optimizer comparisons. |
| Fantastic Pretraining Optimizers, Wen et al. | ICLR 2026 | Final-budget comparisons across model scales and data ratios; ranking-flip checks; fair baseline sensitivity. |

Primary source links:

```text
FineWeb: https://proceedings.neurips.cc/paper_files/paper/2024/hash/370df50ccfdf8bde18f8f9c2d9151bda-Abstract-Datasets_and_Benchmarks_Track.html
DCLM: https://proceedings.neurips.cc/paper_files/paper/2024/hash/19e4ea30dded58259665db375885e412-Abstract-Datasets_and_Benchmarks_Track.html
Dolma: https://arxiv.org/abs/2402.00159
AdamW: https://openreview.net/forum?id=Bkg6RiCqY7
Sophia: https://proceedings.iclr.cc/paper_files/paper/2024/hash/06960915ba8674c7a898ec0b472b80ff-Abstract-Conference.html
GaLore: https://openreview.net/forum?id=hYHsrKDiX7
SOAP: https://openreview.net/forum?id=IDxZhXrpNf
Adam-mini: https://proceedings.iclr.cc/paper_files/paper/2025/hash/45ae878717399e6f62d57c65f052cd46-Abstract-Conference.html
Lion: https://papers.neurips.cc/paper_files/paper/2023/hash/9a39b4925e35cf447ccba8757137d84f-Abstract-Conference.html
Schedule-Free: https://neurips.cc/virtual/2024/poster/96925
AdEMAMix: https://iclr.cc/virtual/2025/poster/28625
CAME: https://aclanthology.org/2023.acl-long.243/
Cautious Optimizers: https://openreview.net/forum?id=zBPZeRjfgu
Fantastic Pretraining Optimizers: https://openreview.net/forum?id=2J51qUZ0iG
```

## Exact Experiments

The exact experiments are specified in `ICLR_EXACT_RUN_PLAN.md`. The corrected design uses:

```text
comparability anchors/checks: C4-EN main, OpenWebText/Pile loader or appendix only where useful
modern 2026-grade corpora: FineWeb-Edu, FineWeb, DCLM, Dolma sample
```

FineWeb remains in the paper as the modern web stress test. It is not described as a dataset copied from accepted optimizer papers.

The run blocks are:

```text
0. loader/model smokes
1. baseline protocol lock
2. M0 main loss-vs-compute benchmark suite
3. 600M long-horizon frontier
4. M1/M2 model-scale study
5. batch, throughput, memory, and optimizer-state accounting
6. cross-corpus transfer
7. corpus-shift continued training
8. sensitivity maps
9. mechanism diagnostics from main logs
10. method ablations last
```

## Operational Constraints

```text
max 4 A6000 GPUs per job
max 8 A6000 GPUs active at once
repo size below 200G
commit compact summaries and JSONL traces only
never commit checkpoints, caches, datasets, or Slurm logs
```
