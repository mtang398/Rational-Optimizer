# TILLER followed by Muon

This directory contains results for GRAIN trained with TILLER for updates
1–1,000 and ordinary Muon for updates 1,001–3,050. The optimizer retains
compatible momentum and AdamW state across the switch.

The suite covers the 296.87M-parameter model on five datasets and three seeds
at approximately 100M training tokens. Its matched comparison is SwiGLU + Muon.
All 15 dataset–seed runs have reached the 3,050-step endpoint.

`runs.csv` records individual runs, `summary.csv` groups seed results, and
`checkpoints.csv` contains validation measurements along each trajectory.
Pending rows receive measurements as the runs finish. The exact commands are
in [the reproduction protocol](../../protocol/README.md).
