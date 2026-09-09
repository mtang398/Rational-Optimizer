# TILLER

TILLER stands for **Tangents Informed by a Loss Ledger for Equal-budget
Reweighting**. It coordinates the update directions of all rational groups by
asking a global question: given the loss responses observed across the model,
how should a fixed update budget be redistributed among the structured
directions available at this step?

The public API is [`tiller.py`](tiller.py), with the numerical implementation
in [`_tiller/`](_tiller/).

## Structured directions

Let $C=LG$ index the $G$ rational groups in each of $L$ Transformer
blocks. For coordinate $i$, TILLER starts with a Nesterov–Muon polar
direction $p_i$. Row and column second moments of the current gradient define
a factorized adaptive direction $a_i$. Its component parallel to $p_i$ is
removed:

\[
a_i^\perp = a_i-
\frac{\langle a_i,p_i\rangle}{\lVert p_i\rVert^2}p_i.
\]

The rational response supplies a participation value \(\pi_i\) and a
congruence value \(\kappa_i\). Their combination

\[
e_i=\pi_i(1-\kappa_i^2)
\]

allocates angular energy to the adaptive tangent. The resulting chord keeps
the native direction norm:

\[
q_i=\sqrt{1-e_i}\,p_i+
\sigma_i\sqrt{e_i}\,
\frac{\lVert p_i\rVert}{\lVert a_i^\perp\rVert}a_i^\perp,
\]

where $\sigma_i\in\{-1,1\}$ orients the tangent toward descent. A coordinate falls back to
$p_i$ whenever the chord is non-finite, degenerate, or loses first-order
descent.

## Response information

For an incoming group matrix, GRAIN supplies the exact group Jacobian $J$.
For an outgoing group matrix, it supplies the feature vector $h$. TILLER uses
two complementary summaries of these responses. The incoming participation

\[
\operatorname{pr}_A(J)=
\frac{[\operatorname{tr}(JJ^\top)]^2}
{w\lVert JJ^\top\rVert_{\mathrm F}^2}
\]

is a normalized inverse participation ratio of the squared singular values of
$J$. The outgoing participation

\[
\operatorname{pr}_B(h)=
\frac{\lVert h\rVert_2^4}{w\lVert h\rVert_4^4}
\]

measures how broadly feature energy is distributed across the $w$ coordinates.
Both range from $1/w$ for a maximally concentrated nonzero response to one for
uniform participation.

Congruence compares the live response kernel with the kernel produced by the
initial rational function at the current normalized input. For role
$r\in\{A,B\}$, with $\Gamma^A=JJ^\top$ and $\Gamma^B=hh^\top$,

\[
\kappa_i^r=
\frac{\sum_n\alpha_{in}^r
\langle\Gamma_{in}^r,\Gamma_{in}^{r,0}\rangle}
{\sqrt{(\sum_n\alpha_{in}^r\lVert\Gamma_{in}^r\rVert_{\mathrm F}^2)
(\sum_n\alpha_{in}^r\lVert\Gamma_{in}^{r,0}\rVert_{\mathrm F}^2)}}.
\]

The weights $\alpha_{in}^r$ are downstream cotangent energies, so responses
matter in proportion to their effect on the current loss. Participation asks
whether a response supports motion across many dimensions; $1-(\kappa_i^r)^2$
asks how far its kernel has rotated from initialization. Their product is the
fraction of direction energy assigned to the adaptive tangent.

## Loss ledger

Every eighth step, 32 fixed functional probes measure how each coordinate's
structured direction changes the downstream loss. These rows form
$S_t\in\mathbb{R}^{32\times C}$. For probe $k$ and coordinate $i$, the
directional score is

\[
S_{t,k,i}
=\langle q_i^A x_{k,\ell},J_{k,i}^\top B_i^\top\xi_{k,\ell}\rangle
+\langle q_i^B h_{k,i},\xi_{k,\ell}\rangle,
\]

where $\xi_{k,\ell}$ is the downstream loss cotangent. A row therefore records
how the same loss probe responds to every layer--group direction; its outer
product couples directions that influence that probe together. Between
functional refreshes, TILLER uses
the exact directional gradient scores from the two matrix roles, sums their
additive shard contributions, and matches their norm to the most recent
functional row norm. Repeating that calibrated score creates a 32-row
surrogate, so the temporal geometry advances on every optimizer step.

The ledger maintains a rank-64 factor with a Robust Frequent Directions
update. For the current score rows, it forms

\[
\widetilde B_t=
\begin{bmatrix}
\sqrt{\beta_2}B_{t-1}\\
\sqrt{1-\beta_2}S_t
\end{bmatrix},
\]

whose live row count is at most 96. The eigendecomposition is performed on the
row Gram matrix \(\widetilde B_t\widetilde B_t^\top\). Frequent-Directions
shrinkage retains 64 rows, while the discarded midpoint energy is accumulated
as an isotropic tail. The represented loss geometry is therefore a compact
low-rank-plus-isotropic metric over all layer/group coordinates.

## Equal-budget transaction

The ledger metric scores global coefficient vectors $c\in\mathbb{R}^{C}$.
TILLER constructs a 32-dimensional loss-Krylov subspace and solves the
equal-budget quadratic

\[
\min_c\; -\eta s_t^\top c+
\frac{\eta^2}{2}\left[
K^{-1}\lVert\mathcal A_t c\rVert_2^2+
\delta_t^\top(c\odot c)+2r_t^\top c\right],
\qquad
\sum_i\omega_i c_i^2=\sum_i\omega_i,
\]

where $s_t$ is the current directional derivative, $\mathcal A_t$ and
$\delta_t$ are the low-rank-plus-diagonal response geometry, $r_t$ is the
score--decay cross term, and $\omega_i$ is the squared norm of the paired
direction. The linear term rewards immediate descent; the quadratic term
penalizes directions whose loss responses repeatedly reinforce one another.
The selected coefficients may have either sign:
they can amplify, attenuate, or reverse individual structured directions while
the global transaction preserves the parent update budget. The candidate is
accepted after finite-value, budget, and descent checks; otherwise the parent
coefficient vector is used.

The same factorized tangent construction is applied to the attention matrix
roles. Other two-dimensional matrices follow Muon, and scalar, normalization,
embedding, and remaining parameters follow AdamW through the composite
training entrypoint.

## Distributed execution

Parameter-space arithmetic stays on the shards that hold each tensor. Loss
scores and factorized row/column statistics are additive, so arbitrary tensor,
FSDP, data, and pipeline partitions contribute through reductions of compact
statistics. Each rank reconstructs its local structured update from the shared
coefficient result. This design is independent of complete-layer placement and
of the number of activation positions processed by a step.

For $C=LG$, the persistent optimizer state scales as

\[
O(LH+LGd+64LG),
\]

and the TILLER ledger itself stores $66C+4$ elements. Its maximum live
temporal factor stores $96C$ elements. The temporal eigendecomposition is at
most $96\times96$, and the equal-budget transaction is at most
$32\times32$.

Here $H_r$ is the expanded GRAIN width consumed by TILLER; the corresponding
SwiGLU intermediate widths are 2,048 and 3,072 in the first two rows.

| Configuration | $L$ | $G$ | $H_r$ | $d$ | Ledger state | Total persistent state | Maximum live factor |
|---|---:|---:|---:|---:|---:|---:|---:|
| 12 layers, width 768 | 12 | 12 | 3,072 | 768 | 9,508 | 366,398 | 13,824 |
| 18 layers, width 1,024 | 18 | 18 | 4,608 | 1,024 | 21,388 | 976,394 | 31,104 |
| Scale example | 96 | 64 | 24,576 | 8,192 | 405,508 | 110,788,838 | 589,824 |

The scale example remains unchanged when total activation positions vary from
one to 1,050,000.

## API

- `TILLERRouter`: rational feed-forward matrix coordination.
- `TILLERAttentionOptimizer`: attention matrix update path.
- `tiller_scaling_formula(...)`: concrete state, communication, and temporary-storage
  accounting for a model shape.
- `experiments/protocol/method_entrypoint.py`: composite optimizer wiring used
  by the language-model experiments.
