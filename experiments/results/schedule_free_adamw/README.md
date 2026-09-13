# Schedule-Free AdamW results

This directory contains the SwiGLU/GRAIN activation comparison with
Schedule-Free AdamW across the published model scales, five datasets, and
three seeds.

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
| DCLM | SwiGLU | 5.4545 ± 0.0291 | 5.0363 ± 0.0325 | 4.6521 ± 0.0334 | 4.4730 ± 0.0313 | 4.3908 ± 0.0298 | 4.3657 ± 0.0298 |
| DCLM | GRAIN | 5.4438 ± 0.0343 | 5.0255 ± 0.0361 | 4.6505 ± 0.0395 | 4.4681 ± 0.0363 | 4.3863 ± 0.0341 | 4.3607 ± 0.0344 |
| FineWeb-Edu | SwiGLU | 5.4788 ± 0.0185 | 4.9453 ± 0.0231 | 4.4643 ± 0.0258 | 4.2687 ± 0.0247 | 4.1826 ± 0.0246 | 4.1559 ± 0.0238 |
| FineWeb-Edu | GRAIN | 5.4615 ± 0.0119 | 4.9220 ± 0.0147 | 4.4509 ± 0.0261 | 4.2553 ± 0.0232 | 4.1689 ± 0.0220 | 4.1421 ± 0.0217 |
| FineWeb | SwiGLU | 5.5684 ± 0.0117 | 5.1193 ± 0.0137 | 4.6901 ± 0.0124 | 4.5060 ± 0.0105 | 4.4236 ± 0.0103 | 4.3979 ± 0.0106 |
| FineWeb | GRAIN | 5.5567 ± 0.0163 | 5.1058 ± 0.0172 | 4.6764 ± 0.0084 | 4.4933 ± 0.0078 | 4.4117 ± 0.0080 | 4.3864 ± 0.0081 |
| Dolma-sample | SwiGLU | 5.3512 ± 0.0087 | 4.8985 ± 0.0056 | 4.4895 ± 0.0060 | 4.3161 ± 0.0042 | 4.2391 ± 0.0055 | 4.2151 ± 0.0053 |
| Dolma-sample | GRAIN | 5.3440 ± 0.0087 | 4.8938 ± 0.0061 | 4.4897 ± 0.0102 | 4.3132 ± 0.0109 | 4.2360 ± 0.0112 | 4.2121 ± 0.0112 |
| C4 | SwiGLU | 5.5598 ± 0.0344 | 5.0728 ± 0.0242 | 4.6171 ± 0.0131 | 4.4261 ± 0.0100 | 4.3422 ± 0.0104 | 4.3163 ± 0.0107 |
| C4 | GRAIN | 5.5390 ± 0.0303 | 5.0615 ± 0.0202 | 4.6063 ± 0.0090 | 4.4169 ± 0.0077 | 4.3340 ± 0.0075 | 4.3084 ± 0.0071 |

[Recorded checkpoint summary](https://github.com/mtang398/Rational-Optimizer/blob/dd97429cd9458c359dfe0d7e576d81aafe334bec/experiments/results/iclr26_e2_figures/README.md).
