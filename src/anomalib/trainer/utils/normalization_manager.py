"""Normalizer used in AnomalibTrainer.

This is responsible for setting up the normalization method.
"""

from __future__ import annotations

from pytorch_lightning.utilities.types import STEP_OUTPUT

import anomalib.trainer as core
from anomalib.post_processing import NormalizationMethod

from .normalizer import BaseNormalizer, CDFNormalizer, MinMaxNormalizer


class NormalizationManager:
    """The normalizer class is instantiated by the trainer.

    This is responsible for updating the normalization values and normalizing the outputs.

    Args:
        normalization_method (NormalizationMethod): Normalization method. Defaults to None
    """

    def __init__(
        self, trainer: core.AnomalibTrainer, normalization_method: NormalizationMethod = NormalizationMethod.NONE
    ):
        self.normalization_method: NormalizationMethod = normalization_method
        self.trainer = trainer
        self.normalizer: BaseNormalizer | None = self._get_normalizer()

    def _get_normalizer(self) -> BaseNormalizer | None:
        """Returns the normalizer class based on the normalization method.

        Returns:
            BaseNormalizer: Normalizer class
        """
        normalization_method: BaseNormalizer | None = None
        if self.normalization_method == NormalizationMethod.MIN_MAX:
            normalization_method = MinMaxNormalizer(self.trainer)
        elif self.normalization_method == NormalizationMethod.CDF:
            normalization_method = CDFNormalizer(self.trainer)
        elif self.normalization_method == NormalizationMethod.NONE:
            raise ValueError(f"Normalization method {self.normalization_method} is not supported.")
        return normalization_method

    def update(self, outputs: STEP_OUTPUT):
        """Update values

        Args:
            outputs (STEP_OUTPUT): Outputs used for gathering normalization metrics.
        """
        if self.normalizer is not None:
            self.normalizer.update(outputs)

    def normalize(self, outputs: STEP_OUTPUT) -> None:
        """Normalize the outputs.

        Args:
            outputs (STEP_OUTPUT): outputs to normalize
        """
        if self.normalizer is not None:
            self.normalizer.normalize(outputs)
