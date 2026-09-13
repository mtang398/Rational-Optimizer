# Lion results

This directory contains the SwiGLU/GRAIN activation comparison with Lion
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
| DCLM | SwiGLU | 4.9565 ± 0.0352 | 4.5005 ± 0.0231 | 4.2217 ± 0.0219 | 4.0876 ± 0.0241 | 4.0126 ± 0.0228 | 3.9934 ± 0.0230 |
| DCLM | GRAIN | 4.9103 ± 0.0427 | 4.4743 ± 0.0318 | 4.2123 ± 0.0265 | 4.0805 ± 0.0294 | 4.0068 ± 0.0295 | 3.9887 ± 0.0295 |
| FineWeb-Edu | SwiGLU | 4.8099 ± 0.0075 | 4.3046 ± 0.0220 | 3.9982 ± 0.0231 | 3.8478 ± 0.0196 | 3.7659 ± 0.0215 | 3.7440 ± 0.0208 |
| FineWeb-Edu | GRAIN | 4.7597 ± 0.0237 | 4.2879 ± 0.0256 | 3.9920 ± 0.0233 | 3.8432 ± 0.0209 | 3.7636 ± 0.0227 | 3.7416 ± 0.0214 |
| FineWeb | SwiGLU | 5.0084 ± 0.0139 | 4.5399 ± 0.0078 | 4.2458 ± 0.0089 | 4.1009 ± 0.0102 | 4.0221 ± 0.0085 | 4.0015 ± 0.0085 |
| FineWeb | GRAIN | 4.9539 ± 0.0114 | 4.5179 ± 0.0112 | 4.2366 ± 0.0122 | 4.0943 ± 0.0107 | 4.0161 ± 0.0103 | 3.9960 ± 0.0105 |
| Dolma-sample | SwiGLU | 4.8041 ± 0.0041 | 4.3710 ± 0.0090 | 4.0847 ± 0.0104 | 3.9425 ± 0.0101 | 3.8670 ± 0.0094 | 3.8475 ± 0.0094 |
| Dolma-sample | GRAIN | 4.7624 ± 0.0186 | 4.3478 ± 0.0144 | 4.0769 ± 0.0121 | 3.9361 ± 0.0106 | 3.8607 ± 0.0089 | 3.8412 ± 0.0085 |
| C4 | SwiGLU | 4.9530 ± 0.0226 | 4.4680 ± 0.0107 | 4.1678 ± 0.0108 | 4.0211 ± 0.0103 | 3.9407 ± 0.0113 | 3.9213 ± 0.0105 |
| C4 | GRAIN | 4.8989 ± 0.0205 | 4.4407 ± 0.0113 | 4.1554 ± 0.0124 | 4.0137 ± 0.0148 | 3.9327 ± 0.0132 | 3.9132 ± 0.0139 |

[Recorded checkpoint summary](https://github.com/mtang398/Rational-Optimizer/blob/dd97429cd9458c359dfe0d7e576d81aafe334bec/experiments/results/iclr26_e2_figures/README.md).
