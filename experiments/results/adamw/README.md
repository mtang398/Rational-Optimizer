# AdamW results

This directory contains the SwiGLU/GRAIN activation comparison with AdamW
across the published model scales, five datasets, and three seeds.

- `runs.csv`: run-level endpoint and timing records.
- `summary.csv`: unique dataset–budget–activation cells.
- `checkpoints.csv`: checkpoint-level validation records when available.

Both SwiGLU and GRAIN activation records use the same table schema.


## 12-layer model, 300M training tokens

Validation loss for the 12-layer, 768-wide model at six training checkpoints.
Cells report the mean ± sample standard deviation across three seeds, at the
four-decimal precision of the recorded checkpoint summary. A smaller finite
seed count is shown as `n=1` or `n=2`; `—` indicates that no finite measurement
was reported at that step. Individual endpoint losses and training times are
in [runs.csv](runs.csv).

| Dataset | Activation | Step 1,000 | Step 2,000 | Step 4,000 | Step 6,000 | Step 8,000 | Step 9,150 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DCLM | SwiGLU | 4.9903 ± 0.0383 | 4.5538 ± 0.0272 | 4.2691 ± 0.0242 | 4.1383 ± 0.0272 | 4.0667 ± 0.0268 | 4.0493 ± 0.0275 |
| DCLM | GRAIN | 4.9860 ± 0.0322 | 4.5477 ± 0.0292 | 4.2697 ± 0.0283 | 4.1394 ± 0.0315 | 4.0676 ± 0.0299 | 4.0496 ± 0.0298 |
| FineWeb-Edu | SwiGLU | 4.8639 ± 0.0168 | 4.3679 ± 0.0200 | 4.0543 ± 0.0197 | 3.9046 ± 0.0184 | 3.8255 ± 0.0186 | 3.8035 ± 0.0182 |
| FineWeb-Edu | GRAIN | 4.8515 ± 0.0270 | 4.3616 ± 0.0263 | 4.0497 ± 0.0226 | 3.9023 ± 0.0199 | 3.8238 ± 0.0209 | 3.8024 ± 0.0206 |
| FineWeb | SwiGLU | 5.0527 ± 0.0192 | 4.6013 ± 0.0095 | 4.2990 ± 0.0101 | 4.1566 ± 0.0097 | 4.0805 ± 0.0098 | 4.0612 ± 0.0101 |
| FineWeb | GRAIN | 5.0480 ± 0.0115 | 4.5930 ± 0.0068 | 4.2973 ± 0.0115 | 4.1554 ± 0.0105 | 4.0795 ± 0.0102 | 4.0601 ± 0.0110 |
| Dolma-sample | SwiGLU | 4.8563 ± 0.0014 | 4.4240 ± 0.0099 | 4.1350 ± 0.0106 | 3.9963 ± 0.0097 | 3.9227 ± 0.0091 | 3.9037 ± 0.0091 |
| Dolma-sample | GRAIN | 4.8475 ± 0.0112 | 4.4228 ± 0.0070 | 4.1366 ± 0.0099 | 3.9964 ± 0.0087 | 3.9231 ± 0.0078 | 3.9045 ± 0.0082 |
| C4 | SwiGLU | 5.0066 ± 0.0246 | 4.5330 ± 0.0144 | 4.2225 ± 0.0137 | 4.0777 ± 0.0133 | 4.0005 ± 0.0114 | 3.9811 ± 0.0128 |
| C4 | GRAIN | 5.0046 ± 0.0293 | 4.5251 ± 0.0107 | 4.2188 ± 0.0097 | 4.0755 ± 0.0115 | 3.9980 ± 0.0101 | 3.9787 ± 0.0098 |

[Recorded checkpoint summary](https://github.com/mtang398/Rational-Optimizer/blob/dd97429cd9458c359dfe0d7e576d81aafe334bec/experiments/results/iclr26_e2_figures/README.md).
