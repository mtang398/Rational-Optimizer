# Activations

This directory contains the two feed-forward activation designs used by the
reported experiments: SwiGLU and GRAIN.

## SwiGLU

For model width $d$ and intermediate width $H$, SwiGLU uses

$$
\operatorname{SwiGLU}(x)
=W_o\!\left(\operatorname{SiLU}(W_gx)\odot W_vx\right),
$$

with two $d\times H$ input projections and one $H\times d$ output projection.

## GRAIN

**GRAIN** stands for **Groupwise Rational Activation with Internal
Normalization**. It uses one expanded projection, a learned rational map per
channel group, and one output projection.

Let $z=Ax\in\mathbb{R}^{3H/2}$ and split its final dimension into $G$ groups.
For group $g$,

$$
\rho_g=\sqrt{\operatorname{mean}(z_g^2)+10^{-6}},
\qquad u_g=z_g/\rho_g.
$$

GRAIN applies a positive-denominator P5/Q4 map

$$
r_g(u)=
\frac{\sum_{k=0}^{5}a_{gk}u^k}
{1+\sum_{k=1}^{4}|b_{gk}|\,|u|^k},
$$

restores the group scale, and mixes the result:

$$
\operatorname{GRAIN}(x)=B\,[\rho_1r_1(u_1),\ldots,\rho_Gr_G(u_G)].
$$

The trainer selects this activation with `--activation grain`. Its projection
width is $3H/2$, so its two dense matrices contain the same $3dH$ weights as
the three SwiGLU matrices. The group count is the largest divisor of $3H/2$
not exceeding both the requested group cap and
$\lceil(3H/2)/256\rceil$. This gives 12 groups for the 3,072-wide GRAIN
projection and 18 groups for the 4,608-wide projection used in the reported
models.

The rational coefficients are initialized from a SiLU fit on $[-5,5]$. Each
group learns six numerator and four denominator coefficients.

## Code

```text
rational_opt/rational.py       GRAIN module and its autograd binding
csrc/rational_ext.cpp          extension interface
csrc/rational_cuda_kernel.cu   fused CUDA forward/backward/statistics kernels
```

The public Python API is deliberately small:

```python
from rational_opt import GRAIN

activation = GRAIN(hidden_dim=3072, groups=12)
```

Build the extension from the repository root:

```bash
.venv/bin/python setup.py build_ext --inplace
```

Run the extension smoke test:

```bash
.venv/bin/python -m experiments.protocol.smoke_cuda_extension
```
