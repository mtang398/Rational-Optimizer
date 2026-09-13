# CAME results

This directory contains the SwiGLU/GRAIN activation comparison with CAME
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
| DCLM | SwiGLU | 5.5228 ± 0.0322 | 5.1088 ± 0.0252 | 4.6850 ± 0.0205 | 4.4791 ± 0.0221 | 4.3907 ± 0.0223 | 4.3682 ± 0.0226 |
| DCLM | GRAIN | 5.5213 ± 0.0384 | 5.1190 ± 0.0288 | 4.7592 ± 0.0273 | 4.5640 ± 0.0381 | 4.4736 ± 0.0416 | 4.4503 ± 0.0426 |
| FineWeb-Edu | SwiGLU | 5.5271 ± 0.0210 | 5.0004 ± 0.0184 | 4.5080 ± 0.0339 | 4.2755 ± 0.0247 | 4.1761 ± 0.0218 | 4.1503 ± 0.0211 |
| FineWeb-Edu | GRAIN | 5.5313 ± 0.0176 | 5.0050 ± 0.0209 | 4.5633 ± 0.0268 | 4.3561 ± 0.0283 | 4.2529 ± 0.0348 | 4.2255 ± 0.0358 |
| FineWeb | SwiGLU | 5.6395 ± 0.0158 | 5.1935 ± 0.0228 | 4.7432 ± 0.0503 | 4.5220 ± 0.0254 | 4.4301 ± 0.0193 | 4.4060 ± 0.0189 |
| FineWeb | GRAIN | 5.6293 ± 0.0197 | 5.1998 ± 0.0126 | 4.8045 ± 0.0092 | 4.6015 ± 0.0077 | 4.5053 ± 0.0061 | 4.4804 ± 0.0064 |
| Dolma-sample | SwiGLU | 5.4110 ± 0.0084 | 4.9728 ± 0.0149 | 4.5654 ± 0.0444 | 4.3689 ± 0.0414 | 4.2734 ± 0.0294 | 4.2492 ± 0.0270 |
| Dolma-sample | GRAIN | 5.4015 ± 0.0110 | 4.9762 ± 0.0145 | 4.5928 ± 0.0262 | 4.3822 ± 0.0414 | 4.2907 ± 0.0422 | 4.2665 ± 0.0409 |
| C4 | SwiGLU | 5.6174 ± 0.0280 | 5.1482 ± 0.0327 | 4.6857 ± 0.0411 | 4.4522 ± 0.0187 | 4.3543 ± 0.0153 | 4.3298 ± 0.0148 |
| C4 | GRAIN | 5.6076 ± 0.0244 | 5.1431 ± 0.0247 | 4.7251 ± 0.0416 | 4.4954 ± 0.0600 | 4.3902 ± 0.0582 | 4.3651 ± 0.0582 |

[Recorded checkpoint summary](https://github.com/mtang398/Rational-Optimizer/blob/dd97429cd9458c359dfe0d7e576d81aafe334bec/experiments/results/iclr26_e2_figures/README.md).
