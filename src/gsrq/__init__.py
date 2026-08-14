"""From-paper GSRQ reproduction primitives.

This package is intentionally isolated from the accepted FACQ implementation.
It implements only the geometry stated in arXiv:2607.01065 and records inferred
choices in the baseline reproduction plan.
"""

from .gskm import GSKMResult, assign_codewords, fit_gskm
from .residual_quantizer import (
    ResidualQuantizerModel,
    fit_residual_quantizer,
)

__all__ = [
    "GSKMResult",
    "ResidualQuantizerModel",
    "assign_codewords",
    "fit_gskm",
    "fit_residual_quantizer",
]
