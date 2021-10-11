"""
DFKDE: Deep Feature Kernel Density Estimation
"""

import os
from typing import Any, Dict, List, Union

import numpy as np
import torch
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from sklearn.metrics import roc_auc_score
from torch import Tensor
from torchvision.models import resnet50

from anomalib.core.callbacks.model_loader import LoadModelCallback
from anomalib.core.model.feature_extractor import FeatureExtractor
from anomalib.models.base.lightning_modules import ClassificationModule
from anomalib.models.dfkde.normality_model import NormalityModel


class Callbacks:
    """
    DFKDE-specific callbacks
    """

    def __init__(self, config: Union[DictConfig, ListConfig]):
        self.config = config

    def get_callbacks(self) -> List[Callback]:
        """
        Get PADIM model callbacks.
        """
        callbacks: List[Callback] = []
        checkpoint = ModelCheckpoint(
            dirpath=os.path.join(self.config.project.path, "weights"),
            filename="model",
        )
        callbacks.append(checkpoint)

        if "weight_file" in self.config.model.keys():
            model_loader = LoadModelCallback(os.path.join(self.config.project.path, self.config.model.weight_file))
            callbacks.append(model_loader)

        return callbacks

    def __call__(self):
        return self.get_callbacks()


class DfkdeLightning(ClassificationModule):
    """
    DFKDE: Deep Featured Kernel Density Estimation
    """

    def __init__(self, hparams: Union[DictConfig, ListConfig]):
        super().__init__(hparams)
        self.threshold_steepness = 0.05
        self.threshold_offset = 12

        self.feature_extractor = FeatureExtractor(backbone=resnet50(pretrained=True), layers=["avgpool"]).eval()

        self.normality_model = NormalityModel(
            filter_count=hparams.model.max_training_points,
            threshold_steepness=self.threshold_steepness,
            threshold_offset=self.threshold_offset,
        )
        self.callbacks = Callbacks(hparams)()
        self.automatic_optimization = False

    @staticmethod
    def configure_optimizers():
        """
        DFKDE doesn't require optimization, therefore returns no optimizers.
        """
        return None

    def training_step(self, batch, _):  # pylint: disable=arguments-differ
        """Training Step of DFKDE.
        For each batch, features are extracted from the CNN.

        Args:
          batch: Input batch
          _: Index of the batch.

        Returns:
          Deep CNN features.

        """

        self.feature_extractor.eval()
        layer_outputs = self.feature_extractor(batch["image"])
        feature_vector = torch.hstack(list(layer_outputs.values())).detach().squeeze()
        return {"feature_vector": feature_vector}

    def training_epoch_end(self, outputs: List[Dict[str, Any]]) -> None:
        """Fit a KDE model on deep CNN features.

        Args:
          outputs: Batch of outputs from the training step
          outputs: dict:

        Returns:

        """

        feature_stack = torch.vstack([output["feature_vector"] for output in outputs])
        self.normality_model.fit(feature_stack)

    def validation_step(self, batch, _):  # pylint: disable=arguments-differ
        """Validation Step of DFKDE.
            Similar to the training step, features
            are extracted from the CNN for each batch.

        Args:
          batch: Input batch
          _: Index of the batch.

        Returns:
          Dictionary containing probability, prediction and ground truth values.

        """

        self.feature_extractor.eval()
        images, label = batch["image"], batch["label"]
        layer_outputs = self.feature_extractor(images)
        feature_vector = torch.hstack(list(layer_outputs.values())).detach()
        probability = self.normality_model.predict(feature_vector.view(feature_vector.shape[:2]))
        return {"probability": probability, "label": label.cpu().numpy()}

    def validation_epoch_end(self, outputs: List[Union[Tensor, Dict[str, Any]]]) -> None:
        """Compute anomaly classification scores based on probability scores.

        Args:
          outputs: Batch of outputs from the validation step
          outputs: dict:

        Returns:

        """
        pred_labels = np.hstack([output["probability"] for output in outputs])
        true_labels = np.hstack([output["label"] for output in outputs])
        self.results.performance["image_roc_auc"] = roc_auc_score(true_labels, pred_labels)
        self.log(name="auc", value=self.results.performance["image_roc_auc"], on_epoch=True, prog_bar=True)

    def test_step(self, batch, _):
        """Test Step of DFKDE.
            Similar to the training and validation steps,
            features are extracted from the CNN for each batch.

        Args:
          batch: Input batch
          batch_idx: Index of the batch.
          batch: dict:
          batch_idx: dict:

        Returns:

        """

        return self.validation_step(batch, _)
