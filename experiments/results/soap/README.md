# SOAP results

This directory contains the SwiGLU/GRAIN activation comparison with SOAP
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
| DCLM | SwiGLU | 5.1360 ± 0.0184 | 4.6909 ± 0.0570 | 4.3574 ± 0.0355 | 4.1959 ± 0.0296 | 4.1163 ± 0.0290 | 4.0964 ± 0.0300 |
| DCLM | GRAIN | 4.9971 ± 0.0416 | 4.5865 ± 0.0608 | 4.2984 ± 0.0078 | 4.1464 ± 0.0341 | 4.0754 ± 0.0281 | 4.0608 ± 0.0331 |
| FineWeb-Edu | SwiGLU | 5.0606 ± 0.0646 | 4.5418 ± 0.0216 | 4.1519 ± 0.0262 | 3.9789 ± 0.0195 | 3.8866 ± 0.0212 | 3.8629 ± 0.0201 |
| FineWeb-Edu | GRAIN | 4.8982 ± 0.0420 | 4.4264 ± 0.0336 | 4.0819 ± 0.0121 | 3.9314 ± 0.0240 | 3.8468 ± 0.0192 | 3.8240 ± 0.0128 |
| FineWeb | SwiGLU | 5.2775 ± 0.0437 | 4.7942 ± 0.1152 | 4.4121 ± 0.0185 | 4.2201 ± 0.0089 | 4.1358 ± 0.0106 | 4.1139 ± 0.0101 |
| FineWeb | GRAIN | 5.1606 ± 0.0137 | 4.6923 ± 0.0454 | 4.3622 ± 0.0324 | 4.1973 ± 0.0154 | 4.1192 ± 0.0103 | 4.1081 ± 0.0320 |
| Dolma-sample | SwiGLU | 5.0463 ± 0.0622 | 4.5589 ± 0.0265 | 4.2314 ± 0.0103 | 4.0647 ± 0.0121 | 3.9780 ± 0.0081 | 3.9568 ± 0.0093 |
| Dolma-sample | GRAIN | 4.8764 ± 0.0334 | 4.4635 ± 0.0252 | 4.2149 ± 0.0260 | 4.0424 ± 0.0406 | 3.9457 ± 0.0099 | 3.9269 ± 0.0125 |
| C4 | SwiGLU | 5.1948 ± 0.0604 | 4.6889 ± 0.0405 | 4.3065 ± 0.0112 | 4.1452 ± 0.0118 | 4.0716 ± 0.0224 | 4.0349 ± 0.0108 |
| C4 | GRAIN | 5.1604 ± 0.2205 | 4.5515 ± 0.0133 | 4.2526 ± 0.0263 | 4.1011 ± 0.0219 | 4.0193 ± 0.0176 | 4.0024 ± 0.0194 |

[Recorded checkpoint summary](https://github.com/mtang398/Rational-Optimizer/blob/dd97429cd9458c359dfe0d7e576d81aafe334bec/experiments/results/iclr26_e2_figures/README.md).
