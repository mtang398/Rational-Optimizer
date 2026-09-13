# ADeMaMix results

This directory contains the SwiGLU/GRAIN activation comparison with ADeMaMix
across the published model scales, five datasets, and three seeds.

- `runs.csv`: run-level endpoint and timing records.
- `summary.csv`: unique dataset–budget–activation cells.
- `checkpoints.csv`: checkpoint-level validation records when available.

The tables retain the large finite losses, non-finite losses, and early-stop
status produced by this optimizer. Numeric aggregates include only finite
losses from runs that reached the required endpoint.


## 12-layer model, 300M training tokens

Validation loss for the 12-layer, 768-wide model at six training checkpoints.
Cells report the mean ± sample standard deviation across three seeds, at the
four-decimal precision of the recorded checkpoint summary. A smaller finite
seed count is shown as `n=1` or `n=2`; `—` indicates that no finite measurement
was reported at that step. Individual endpoint losses and training times are
in [runs.csv](runs.csv).

| Dataset | Activation | Step 1,000 | Step 2,000 | Step 4,000 | Step 6,000 | Step 8,000 | Step 9,150 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DCLM | SwiGLU | 1709.6780 ± 585.9580 | 2931159.8867 ± 4767520.6794 | 11241.5732 ± 0.0000 (n=1) | 196344.0312 ± 0.0000 (n=1) | 34853138432.0000 ± 0.0000 (n=1) | — |
| DCLM | GRAIN | 7522.7668 ± 12918.9198 | — | — | — | — | — |
| FineWeb-Edu | SwiGLU | 1388.9467 ± 438.1649 | 929392.9245 ± 1048745.5316 | 1370518.8750 ± 0.0000 (n=1) | — | — | — |
| FineWeb-Edu | GRAIN | 27698.5969 ± 47596.7828 | 4962161216.0000 ± 6009468964.3189 (n=2) | — | — | — | — |
| FineWeb | SwiGLU | 1046.3267 ± 385.1340 | 386.1834 ± 120.1352 | 18508.4055 ± 25038.3317 (n=2) | 1100.8079 ± 0.0000 (n=1) | 2696.9573 ± 0.0000 (n=1) | 1361.4141 ± 0.0000 (n=1) |
| FineWeb | GRAIN | 257.9755 ± 250.1753 | 1633877367.1458 ± 2533289655.0567 | — | — | — | — |
| Dolma-sample | SwiGLU | 1127.4188 ± 977.4469 | 1608323.0966 ± 2563013.6932 | 70147874816.0000 ± 0.0000 (n=1) | — | — | — |
| Dolma-sample | GRAIN | 28371.7812 ± 2941.2476 | 2482927104.0000 ± 2466098821.8411 (n=2) | — | — | — | — |
| C4 | SwiGLU | 1104.9656 ± 565.1435 | 180791.1562 ± 0.0000 (n=1) | — | — | — | — |
| C4 | GRAIN | 414.2971 ± 77.7608 | 553095678.5990 ± 943240014.0699 | 1687480172544.0000 ± 0.0000 (n=1) | — | — | — |

[Recorded checkpoint summary](https://github.com/mtang398/Rational-Optimizer/blob/dd97429cd9458c359dfe0d7e576d81aafe334bec/experiments/results/iclr26_e2_figures/README.md).
