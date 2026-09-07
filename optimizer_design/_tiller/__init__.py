"""Numerical implementation of the TILLER optimizer."""

from . import core as _core
from .core import (
    FAMILY_ID,
    PREFIX,
    TILLERAttentionOptimizer,
    TILLERRouter,
    tiller_scaling_formula,
)


def configure_microbatch_count(count: int) -> None:
    """Configure the number of gradient-accumulation microbatches.

    Call this before constructing a :class:`TILLERRouter`. Repeating the call
    with the same or a different positive value is supported for test and
    experiment processes that construct optimizers sequentially.
    """

    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("TILLER microbatch count must be a positive integer")
    _core._probe_loss_image__EXPECTED_MICROBATCHES = count
    _core._fixed_transaction_base__import_EXPECTED_MICROBATCHES = count


__all__ = (
    "FAMILY_ID",
    "PREFIX",
    "TILLERAttentionOptimizer",
    "TILLERRouter",
    "configure_microbatch_count",
    "tiller_scaling_formula",
)
