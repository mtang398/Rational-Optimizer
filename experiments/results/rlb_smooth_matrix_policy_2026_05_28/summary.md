| row | final loss | final PPL | gap vs winner loss | gap vs winner PPL |
| --- | ---: | ---: | ---: | ---: |
| RLB Smooth-MatrixPolicy (same LR) | 3.493210 | 32.89 | 0.000000 | 0.00 |
| RLB beta2=0.999 explicit | 3.493851 | 32.91 | 0.000641 | 0.02 |
| RLB early-off transport | 3.548505 | 34.76 | 0.055295 | 1.87 |
| RLB MatrixPolicy-Y (same LR) | 3.548665 | 34.77 | 0.055455 | 1.88 |
| SiLU/SwiGLU + AdamW beta2=0.999 | 3.549346 | 34.79 | 0.056136 | 1.90 |
| RLB + AdamW beta2=0.999 | 3.550018 | 34.81 | 0.056808 | 1.92 |
| RLB + Jacobian | 3.614862 | 37.15 | 0.121652 | 4.25 |
| RLB + AdamW | 3.617501 | 37.24 | 0.124291 | 4.35 |
| SiLU/SwiGLU + AdamW | 3.621982 | 37.41 | 0.128772 | 4.52 |
