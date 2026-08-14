"""Fisher-whitened Gain--Shape Residual Quantization."""

from .codec import (
    FWResidualQuantizerModel,
    WhiteningTransform,
    build_fixed_whitener,
    build_spectral_fisher_whitener,
    build_sqrt_fisher_whitener,
    fit_fw_residual_quantizer,
    fit_scfw_residual_quantizer,
    fold_whitener_into_projection,
    inverse_transform_rows,
    roundtrip_selector_codebooks,
    transform_rows,
)

__all__ = [
    "FWResidualQuantizerModel",
    "WhiteningTransform",
    "build_fixed_whitener",
    "build_spectral_fisher_whitener",
    "build_sqrt_fisher_whitener",
    "fit_fw_residual_quantizer",
    "fit_scfw_residual_quantizer",
    "fold_whitener_into_projection",
    "inverse_transform_rows",
    "roundtrip_selector_codebooks",
    "transform_rows",
]
