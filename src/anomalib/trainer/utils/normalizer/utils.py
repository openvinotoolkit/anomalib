"""Utils for normalizers.

This is responsible for setting up the normalization method.
"""

from __future__ import annotations

import anomalib.trainer as trainer  # to avoid circular import
from anomalib.post_processing import NormalizationMethod

from .base import BaseNormalizer
from .cdf import CDFNormalizer
from .min_max import MinMaxNormalizer


def get_normalizer(
    normalization_method: NormalizationMethod, trainer: trainer.AnomalibTrainer
) -> BaseNormalizer | None:
    """Returns the normalizer class based on the normalization method.

    Args:
        normalization_method (NormalizationMethod): Normalization method. Defaults to None
        trainer (AnomalibTrainer): Trainer object.

    Returns:
        BaseNormalizer: Normalizer class or None if no normalization is used.
    """
    normalizer: BaseNormalizer | None = None
    if normalization_method == NormalizationMethod.MIN_MAX:
        normalizer = MinMaxNormalizer(trainer)
    elif normalization_method == NormalizationMethod.CDF:
        normalizer = CDFNormalizer(trainer)
    elif normalization_method != NormalizationMethod.NONE:
        raise ValueError(f"Normalization method {normalization_method} is not supported.")
    return normalizer
