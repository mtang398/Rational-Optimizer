"""Optimizer research components for rational activation experiments."""

from .function_space_rational_optimizer import FunctionSpaceRationalOptimizer
from .factored_adamw import FactoredAdamW
from .switching_rational_optimizer import SwitchingRationalOptimizer
from .onpolicy_balance_optimizer import RationalOnPolicyBalanceOptimizer
from .quotient_onpolicy_optimizer import RationalQuotientOnPolicyOptimizer
from .jacobian_onpolicy_optimizer import RationalJacobianOnPolicyOptimizer
from .quotient_jacobian_onpolicy_optimizer import RationalQuotientJacobianOnPolicyOptimizer
from .adaptive_metric_onpolicy_optimizer import RationalAdaptiveMetricOnPolicyOptimizer
from .transport_onpolicy_optimizer import RationalTransportOnPolicyOptimizer
from .functional_trust_onpolicy_optimizer import RationalFunctionalTrustOnPolicyOptimizer
from .matrix_policy_optimizer import RationalMatrixPolicyOptimizer

__all__ = [
    "FunctionSpaceRationalOptimizer",
    "FactoredAdamW",
    "SwitchingRationalOptimizer",
    "RationalOnPolicyBalanceOptimizer",
    "RationalQuotientOnPolicyOptimizer",
    "RationalJacobianOnPolicyOptimizer",
    "RationalQuotientJacobianOnPolicyOptimizer",
    "RationalAdaptiveMetricOnPolicyOptimizer",
    "RationalTransportOnPolicyOptimizer",
    "RationalFunctionalTrustOnPolicyOptimizer",
    "RationalMatrixPolicyOptimizer",
]
