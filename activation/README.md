# Rational Latent Basis activations

This directory contains the Rational Latent Basis activation implementations
and the NVCC-built fused CUDA extension used by matched language-model runs.

The active campaign uses `rlb_fused_global_rational` and requires
`RATIONAL_OPT_TORCH_FALLBACK=0`. Candidate and control rows retain the original
activation initialization and all shared model settings. Compilation is an
environment preparation step; end-to-end timing follows the same preparation
standard for both arms.
