"""
DFM: Deep Feature Kernel Density Estimation
"""

import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pytorch_lightning as pl
import torch
from attrdict import AttrDict
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from sklearn.metrics import roc_auc_score
from torchvision.models import resnet18

from anomalib.core.callbacks.model_loader import LoadModelCallback
from anomalib.core.model.feature_extractor import FeatureExtractor
from anomalib.models.dfm.pca_model import PCAModel


class Callbacks:
    """
    DFM-specific callbacks
    """

    def __init__(self, config: DictConfig):
        self.config = config

    def get_callbacks(self) -> Sequence[Callback]:
        """
        Get PADIM model callbacks.
        """
        checkpoint = ModelCheckpoint(
            dirpath=os.path.join(self.config.project.path, "weights"),
            filename="model",
        )
        callbacks = [checkpoint]

        if "weight_file" in self.config.keys():
            model_loader = LoadModelCallback(os.path.join(self.config.project.path, self.config.weight_file))
            callbacks.append(model_loader)

        return callbacks

    def __call__(self):
        return self.get_callbacks()


class DFMLightning(pl.LightningModule):
    """
    DFM: Deep Featured Kernel Density Estimation
    """

    def __init__(self, hparams: AttrDict):
        super().__init__()
        self.save_hyperparameters(hparams)        
        self.threshold_steepness = 0.05
        self.threshold_offset = 12

        self.feature_extractor = FeatureExtractor(backbone=resnet18(pretrained=True), layers=["avgpool"]).eval()

        self.pca_model = PCAModel(
          n_comps=hparams.model.pca_level
        )
        self.callbacks = Callbacks(hparams)()
        self.image_roc_auc: Optional[float] = None
        self.automatic_optimization = False

    @staticmethod
    def configure_optimizers():
        """
        DFM doesn't require optimization, therefore returns no optimizers.
        """
        return None

    def training_step(self, batch, _):
        """Training Step of DFM.
        For each batch, features are extracted from the CNN.

        Args:
          batch: Input batch
          batch_idx: Index of the batch.
          batch: dict:
          batch_idx: int:

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
        self.pca_model.fit(feature_stack)

    def validation_step(self, batch, _):
        """Validation Step of DFM.
            Similar to the training step, features
            are extracted from the CNN for each batch.

        Args:
          batch: Input batch
          batch_idx: Index of the batch.
          batch: dict:
          batch_idx: int:

        Returns:
          Dictionary containing probability, prediction and ground truth values.

        """

        self.feature_extractor.eval()
        images, mask = batch["image"], batch["mask"]
        layer_outputs = self.feature_extractor(images)
        feature_vector = torch.hstack(list(layer_outputs.values())).detach()
        feature_reconstruction_error = self.pca_model.score(feature_vector.view(feature_vector.shape[:2]))
        ground_truth = np.any(mask.cpu().numpy(), axis=(1, 2, 3)).astype(int)
        return {"fre_scores": feature_reconstruction_error, "ground_truth": ground_truth}

    def validation_epoch_end(self, outputs: List[Dict[str, Any]]) -> None:
        """Compute anomaly classification scores based on probability scores.

        Args:
          outputs: Batch of outputs from the validation step
          outputs: dict:

        Returns:

        """
        pred_labels = np.hstack([output["fre_scores"] for output in outputs])
        true_labels = np.hstack([output["ground_truth"] for output in outputs])
        self.image_roc_auc = roc_auc_score(true_labels, pred_labels)
        self.log(name="auc", value=self.image_roc_auc, on_epoch=True, prog_bar=True)

    def test_step(self, batch, _):
        """Test Step of DFM.
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

    def test_epoch_end(self, outputs: List[Dict[str, Any]]) -> None:
        """Compute anomaly classification scores based on probability scores.

        Args:
          outputs: Batch of outputs from the validation step
          outputs: dict:

        Returns:

        """
        self.validation_epoch_end(outputs)
