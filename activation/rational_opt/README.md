# GRAIN Python operator

`rational.py` provides the `GRAIN` module, connects it to the fused CUDA
forward/backward operator, and records the activation statistics consumed by
TILLER.

Its learned tensors have shapes `G × 6` for numerator coefficients and
`G × 4` for denominator coefficients. The package exports `GRAIN` and its
low-level `rational_local_basis` autograd helper.
