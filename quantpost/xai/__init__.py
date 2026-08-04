"""Explainability: attribution methods and reservoir-specific probes.

    from quantpost.xai import attribution, reservoir_probes
"""

from . import attribution, reservoir_probes
from .attribution import (
                          Attribution,
                          block_permutation_importance,
                          conditional_permutation_importance,
                          efficiency_check,
                          kernel_shapley,
                          linear_shapley,
                          permutation_importance,
)

__all__ = ["attribution", "reservoir_probes", "Attribution",
           "permutation_importance", "block_permutation_importance",
           "conditional_permutation_importance", "linear_shapley",
           "kernel_shapley", "efficiency_check"]
