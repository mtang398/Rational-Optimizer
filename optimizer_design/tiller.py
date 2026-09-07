"""Public API for TILLER.

TILLER stands for Tangents Informed by a Loss Ledger for Equal-budget
Reweighting. Its implementation is self-contained under
``optimizer_design._tiller``.
"""

from ._tiller import (
    FAMILY_ID,
    PREFIX,
    TILLERAttentionOptimizer,
    TILLERRouter,
    configure_microbatch_count,
    tiller_scaling_formula,
)

__all__ = (
    "FAMILY_ID",
    "PREFIX",
    "TILLERAttentionOptimizer",
    "TILLERRouter",
    "configure_microbatch_count",
    "tiller_scaling_formula",
)
