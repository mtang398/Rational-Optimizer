# Muon results

This directory contains the SwiGLU/GRAIN activation comparison with Muon
across the published model scales, five datasets, and three seeds.

- `runs.csv`: run-level endpoint and timing records.
- `summary.csv`: unique dataset–budget–activation cells.
- `checkpoints.csv`: checkpoint-level validation records when available.

Both SwiGLU and GRAIN activation records use the same table schema.

The 18-layer, 1,024-wide suite trains for 3,050 steps and approximately
100M tokens. All 30 rows in this suite are complete: 15 SwiGLU + Muon runs
and their 15 GRAIN + Muon counterparts. The completed 12-layer results remain
in the same tables.


## 12-layer model, 300M training tokens

Validation loss for the 12-layer, 768-wide model at six training checkpoints.
Cells report the mean ± sample standard deviation across three seeds, at the
four-decimal precision of the recorded checkpoint summary. A smaller finite
seed count is shown as `n=1` or `n=2`; `—` indicates that no finite measurement
was reported at that step. Individual endpoint losses and training times are
in [runs.csv](runs.csv).

| Dataset | Activation | Step 1,000 | Step 2,000 | Step 4,000 | Step 6,000 | Step 8,000 | Step 9,150 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DCLM | SwiGLU | 5.1356 ± 0.0339 | 4.5702 ± 0.0325 | 4.2298 ± 0.0306 | 4.0879 ± 0.0310 | 4.0158 ± 0.0305 | 3.9973 ± 0.0305 |
| DCLM | GRAIN | 5.1313 ± 0.0302 | 4.5676 ± 0.0317 | 4.2230 ± 0.0267 | 4.0829 ± 0.0269 | 4.0088 ± 0.0266 | 3.9913 ± 0.0260 |
| FineWeb-Edu | SwiGLU | 4.9953 ± 0.0177 | 4.3557 ± 0.0228 | 4.0009 ± 0.0183 | 3.8475 ± 0.0161 | 3.7663 ± 0.0175 | 3.7454 ± 0.0170 |
| FineWeb-Edu | GRAIN | 4.9903 ± 0.0154 | 4.3571 ± 0.0185 | 3.9970 ± 0.0185 | 3.8406 ± 0.0175 | 3.7586 ± 0.0184 | 3.7373 ± 0.0187 |
| FineWeb | SwiGLU | 5.1938 ± 0.0130 | 4.5962 ± 0.0095 | 4.2524 ± 0.0119 | 4.1027 ± 0.0121 | 4.0266 ± 0.0117 | 4.0066 ± 0.0128 |
| FineWeb | GRAIN | 5.1855 ± 0.0153 | 4.5923 ± 0.0183 | 4.2471 ± 0.0138 | 4.0966 ± 0.0107 | 4.0200 ± 0.0102 | 3.9991 ± 0.0110 |
| Dolma-sample | SwiGLU | 4.9932 ± 0.0054 | 4.4333 ± 0.0060 | 4.1013 ± 0.0120 | 3.9535 ± 0.0103 | 3.8776 ± 0.0095 | 3.8581 ± 0.0101 |
| Dolma-sample | GRAIN | 4.9914 ± 0.0078 | 4.4326 ± 0.0103 | 4.0917 ± 0.0076 | 3.9430 ± 0.0047 | 3.8677 ± 0.0046 | 3.8478 ± 0.0047 |
| C4 | SwiGLU | 5.1609 ± 0.0341 | 4.5248 ± 0.0117 | 4.1777 ± 0.0099 | 4.0254 ± 0.0140 | 3.9448 ± 0.0125 | 3.9251 ± 0.0134 |
| C4 | GRAIN | 5.1513 ± 0.0273 | 4.5259 ± 0.0109 | 4.1730 ± 0.0110 | 4.0181 ± 0.0146 | 3.9381 ± 0.0140 | 3.9189 ± 0.0149 |

[Recorded checkpoint summary](https://github.com/mtang398/Rational-Optimizer/blob/dd97429cd9458c359dfe0d7e576d81aafe334bec/experiments/results/iclr26_e2_figures/README.md).
