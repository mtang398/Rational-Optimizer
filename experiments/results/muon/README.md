# Muon results

This directory contains the SwiGLU/GRAIN activation comparison with Muon
across the published model scales, five datasets, and three seeds.

- `runs.csv`: run-level endpoint and timing records.
- `summary.csv`: unique dataset–budget–activation cells.
- `checkpoints.csv`: checkpoint-level validation records when available.

Both SwiGLU and GRAIN activation records use the same table schema.

For the 18-layer, 1,024-wide, 300M-token suite, 25 of the 30 Muon runs have
complete 9,150-step endpoints, two interrupted runs retain their partial
trajectories, and three rows are pending. Twenty-three completed rows carry
verified end-to-end process timing; two retained endpoints carry their
original training-loop timing scope.
