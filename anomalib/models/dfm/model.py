"""
DFM: Deep Feature Kernel Density Estimation
"""

from typing import Any, Dict, List, Union

import torch
from omegaconf import DictConfig, ListConfig
from torch import Tensor
from torchvision.models import resnet18

from anomalib.core.model.feature_extractor import FeatureExtractor
from anomalib.core.results import ClassificationResults
from anomalib.models.base.lightning_modules import ClassificationModule
from anomalib.models.dfm.dfm_model import DFMModel


class DfmLightning(ClassificationModule):
    """
    DFM: Deep Featured Kernel Density Estimation
    """

    def __init__(self, hparams: Union[DictConfig, ListConfig]):
        super().__init__(hparams)
        self.save_hyperparameters(hparams)
        self.threshold_steepness = 0.05
        self.threshold_offset = 12

        self.feature_extractor = FeatureExtractor(backbone=resnet18(pretrained=True), layers=["avgpool"]).eval()

        self.dfm_model = DFMModel(n_comps=hparams.model.pca_level, score_type=hparams.model.score_type)
        self.results = ClassificationResults()
        self.automatic_optimization = False

    @staticmethod
    def configure_optimizers():
        """
        DFM doesn't require optimization, therefore returns no optimizers.
        """
        return None

    def training_step(self, batch, _):  # pylint: disable=arguments-differ
        """Training Step of DFM.
        For each batch, features are extracted from the CNN.

        Args:
          batch: Dict: Input batch
          batch_idx: int: Index of the batch.

        Returns:
          Deep CNN features.

        """

        self.feature_extractor.eval()
        layer_outputs = self.feature_extractor(batch["image"])
        feature_vector = torch.hstack(list(layer_outputs.values())).detach().squeeze()
        return {"feature_vector": feature_vector}

    def training_epoch_end(self, outputs: List[Union[Tensor, Dict[str, Any]]]) -> None:
        """Fit a KDE model on deep CNN features.

        Args:
          outputs: Batch of outputs from the training step
          outputs: dict:

        Returns:

        """

        feature_stack = torch.vstack([output["feature_vector"] for output in outputs])
        self.dfm_model.fit(feature_stack)

    def validation_step(self, batch, _):  # pylint: disable=arguments-differ
        """Validation Step of DFM.
            Similar to the training step, features
            are extracted from the CNN for each batch.

        Args:
          batch: Dict: Input batch
          batch_idx: int: Index of the batch.

        Returns:
          Dictionary containing FRE anomaly scores and ground-truth.

        """

        self.feature_extractor.eval()
        layer_outputs = self.feature_extractor(batch["image"])
        feature_vector = torch.hstack(list(layer_outputs.values())).detach()
        batch["pred_scores"] = torch.from_numpy(self.dfm_model.score(feature_vector.view(feature_vector.shape[:2])))
        return batch
