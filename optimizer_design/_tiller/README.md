# TILLER numerical core

`core.py` contains the structured tangent construction, loss-ledger update,
equal-budget transaction, attention path, and distributed reductions used by
TILLER. `__init__.py` exposes the optimizer classes, scaling formula, and
gradient-accumulation configuration to the public API one directory above.
