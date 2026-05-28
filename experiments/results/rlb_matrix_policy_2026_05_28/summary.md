| row | final loss | final PPL | gap vs winner loss | gap vs winner PPL |
| --- | ---: | ---: | ---: | ---: |
| RLB MatrixPolicy-Y (same LR) | 3.548665 | 34.77 | 0.000000 | 0.00 |
| RLB + Jacobian | 3.614862 | 37.15 | 0.066197 | 2.38 |
| RLB + AdamW | 3.617501 | 37.24 | 0.068836 | 2.48 |
| SiLU/SwiGLU + AdamW | 3.621982 | 37.41 | 0.073316 | 2.64 |

| 1250-step row | loss | PPL |
| --- | ---: | ---: |
| Y 1250 same-LR | 4.126142 | 61.94 |
| RLB + Jacobian 1250 | 4.236546 | 69.17 |
| RLB + AdamW 1250 | 4.245049 | 69.76 |
| SiLU/SwiGLU + AdamW 1250 | 4.254405 | 70.41 |
