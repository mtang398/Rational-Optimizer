# Real LM Screen, 2026-05-30

All rows use the same 100M-token training budget, 4M-token heldout slice after a 110M-token stream offset, and the same base LR schedule.
PPL plots omit divergent/nonfinite runs; zoomed validation plots start at step 1000.

## FineWeb

| method | last finite validation loss | last finite PPL | AUC <= 1000 | AUC <= 2000 | note |
| --- | ---: | ---: | ---: | ---: | --- |
| SiLU+AdamW | 4.504617 | 90.43 | 5.993426 | 5.401559 | complete |
| RLB+AdamW | 4.493013 | 89.39 | 5.954484 | 5.373016 | complete |
| SiLU+Muon | 4.535766 | 93.29 | 6.664512 | 5.786310 | complete |
| RLB+Muon | 4.548868 | 94.53 | 6.585091 | 5.752002 | complete |
| RLB+MatrixPolicy (group-stat) | 4.344150 | 77.03 | 5.850945 | 5.262783 | complete |

## FineWeb-Edu

| method | last finite validation loss | last finite PPL | AUC <= 1000 | AUC <= 2000 | note |
| --- | ---: | ---: | ---: | ---: | --- |
| SiLU+AdamW | 4.225019 | 68.38 | 5.835354 | 5.186270 | complete |
| RLB+AdamW | 8.411884 | 4500.23 | 9.684973 | 9.684973 | train nonfinite at step 80; validation nonfinite at step 100 |
| SiLU+Muon | 4.252612 | 70.29 | 6.505154 | 5.563970 | complete |
| RLB+Muon | 4.271556 | 71.63 | 6.425483 | 5.529865 | complete |
| RLB+MatrixPolicy (group-stat) | 4.072055 | 58.68 | 5.670071 | 5.041694 | complete |
