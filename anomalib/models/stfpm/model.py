"""
STFPM: Student-Teacher Feature Pyramid Matching for Unsupervised Anomaly Detection
https://arxiv.org/abs/2103.04257
"""
import os.path
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from sklearn.metrics import roc_auc_score
from torch import Tensor, nn, optim

from anomalib.core.callbacks.compress import CompressModelCallback
from anomalib.core.callbacks.model_loader import LoadModelCallback
from anomalib.core.callbacks.nncf_callback import NNCFCallback
from anomalib.core.callbacks.timer import TimerCallback
from anomalib.core.model.feature_extractor import FeatureExtractor
from anomalib.core.utils.anomaly_map_generator import BaseAnomalyMapGenerator
from anomalib.models.base.model import BaseAnomalySegmentationLightning

__all__ = ["Loss", "AnomalyMapGenerator", "STFPMModel", "STFPMLightning"]


class Loss(nn.Module):
    """
    Feature Pyramid Loss
    This class implmenents the feature pyramid loss function proposed in STFPM [1] paper.

    Example:

    >>> from anomalib.core.model.feature_extractor import FeatureExtractor
    >>> from anomalib.models.stfpm.model import Loss
    >>> from torchvision.models import resnet18

    >>> layers = ['layer1', 'layer2', 'layer3']
    >>> teacher_model = FeatureExtractor(model=resnet18(pretrained=True), layers=layers)
    >>> student_model = FeatureExtractor(model=resnet18(pretrained=False), layers=layers)
    >>> loss = Loss()

    >>> inp = torch.rand((4, 3, 256, 256))
    >>> teacher_features = teacher_model(inp)
    >>> student_features = student_model(inp)
    >>> loss(student_features, teacher_features)
        tensor(51.2015, grad_fn=<SumBackward0>)
    """

    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss(reduction="sum")

    def compute_layer_loss(self, teacher_feats: Tensor, student_feats: Tensor) -> Tensor:
        """Compute layer loss based on Equation (1) in Section 3.2 of the paper.

        Args:
          teacher_feats: Teacher features
          student_feats: Student features
          teacher_feats: Tensor:
          student_feats: Tensor:

        Returns:
          L2 distance between teacher and student features.

        """

        height, width = teacher_feats.shape[2:]

        norm_teacher_features = F.normalize(teacher_feats)
        norm_student_features = F.normalize(student_feats)
        layer_loss = (0.5 / (width * height)) * self.mse_loss(norm_teacher_features, norm_student_features)

        return layer_loss

    def forward(self, teacher_features: Dict[str, Tensor], student_features: Dict[str, Tensor]) -> Tensor:
        """Compute the overall loss via the weighted average of
        the layer losses computed by the cosine similarity.

        Args:
          teacher_features: Teacher features
          student_features: Student features
          teacher_features: Dict[str:
          Tensor]:
          student_features: Dict[str:

        Returns:
          Total loss, which is the weighted average of the layer losses.

        """

        layer_losses: List[Tensor] = []
        for layer in teacher_features.keys():
            loss = self.compute_layer_loss(teacher_features[layer], student_features[layer])
            layer_losses.append(loss)

        total_loss = torch.stack(layer_losses).sum()

        return total_loss


class Callbacks:
    """STFPM-specific callbacks"""

    def __init__(self, config: DictConfig):
        self.config = config

    def get_callbacks(self) -> List[Callback]:
        """Get STFPM model callbacks."""
        checkpoint = ModelCheckpoint(
            dirpath=os.path.join(self.config.project.path, "weights"),
            filename="model",
        )
        early_stopping = EarlyStopping(monitor=self.config.model.metric, patience=self.config.model.patience)
        callbacks = [checkpoint, early_stopping, TimerCallback()]

        if self.config.optimization.nncf.apply:
            callbacks.append(
                NNCFCallback(
                    config=self.config,
                    dirpath=os.path.join(self.config.project.path, "compressed"),
                    filename="compressed_model",
                )
            )
        if self.config.optimization.compression.apply:
            callbacks.append(
                CompressModelCallback(
                    config=self.config,
                    dirpath=os.path.join(self.config.project.path, "compressed"),
                    filename="compressed_model",
                )
            )
        if "weight_file" in self.config.keys():
            model_loader = LoadModelCallback(os.path.join(self.config.project.path, self.config.weight_file))
            callbacks.append(model_loader)

        return callbacks

    def __call__(self):
        return self.get_callbacks()


class AnomalyMapGenerator(BaseAnomalyMapGenerator):
    """Generate Anomaly Heatmap"""

    def __init__(self, batch_size: int = 1, image_size: int = 256, alpha: float = 0.4, gamma: int = 0, sigma: int = 4):
        super().__init__(alpha=alpha, gamma=gamma, sigma=sigma)
        self.distance = torch.nn.PairwiseDistance(p=2, keepdim=True)
        self.image_size = image_size
        self.batch_size = batch_size

    def compute_layer_map(self, teacher_features: Tensor, student_features: Tensor) -> Tensor:
        """Compute the layer map based on cosine similarity.

        Args:
          teacher_features: Teacher features
          student_features: Student features
          teacher_features: Tensor:
          student_features: Tensor:

        Returns:
          Anomaly score based on cosine similarity.

        """
        norm_teacher_features = F.normalize(teacher_features)
        norm_student_features = F.normalize(student_features)

        layer_map = 0.5 * self.distance(norm_student_features, norm_teacher_features) ** 2
        layer_map = F.interpolate(layer_map, size=self.image_size, align_corners=False, mode="bilinear")
        return layer_map

    def compute_anomaly_map(
        self, teacher_features: Dict[str, Tensor], student_features: Dict[str, Tensor]
    ) -> np.ndarray:
        """
        Compute the overall anomaly map via element-wise production the interpolated anomaly maps.

        Args:
          teacher_features: Teacher features
          student_features: Student features
          teacher_features: Dict[str: Tensor]:
          student_features: Dict[str: Tensor]:

        Returns:
          Final anomaly map
        """
        anomaly_map = np.ones([self.image_size, self.image_size])
        for layer in teacher_features.keys():
            layer_map = self.compute_layer_map(teacher_features[layer], student_features[layer])
            layer_map = layer_map[0, 0, :, :]
            anomaly_map *= layer_map.cpu().detach().numpy()

        return anomaly_map

    def __call__(self, teacher_features: Dict[str, Tensor], student_features: Dict[str, Tensor]) -> np.ndarray:
        return self.compute_anomaly_map(teacher_features, student_features)


class STFPMModel(nn.Module):
    """
    STFPM: Student-Teacher Feature Pyramid Matching for Unsupervised Anomaly Detection
    """

    def __init__(self, hparams):
        super().__init__()
        self.backbone = getattr(torchvision.models, hparams.model.backbone)
        self.layers = hparams.model.layers

        self.teacher_model = FeatureExtractor(backbone=self.backbone(pretrained=True), layers=self.layers)
        self.student_model = FeatureExtractor(backbone=self.backbone(pretrained=False), layers=self.layers)

        # teacher model is fixed
        for parameters in self.teacher_model.parameters():
            parameters.requires_grad = False

        self.loss = Loss()
        self.anomaly_map_generator = AnomalyMapGenerator(batch_size=1, image_size=hparams.dataset.crop_size)

    def forward(self, images):
        """Forward-pass images into the network to extract teacher and student network.

        Args:
          images: Batch of images.

        Returns:
          Teacher and student features.

        """
        teacher_features: Dict[str, Tensor] = self.teacher_model(images)
        student_features: Dict[str, Tensor] = self.student_model(images)
        return teacher_features, student_features


class STFPMLightning(BaseAnomalySegmentationLightning):
    """
    PL Lightning Module for the STFPM algorithm.
    """

    def __init__(self, hparams):
        super().__init__(hparams)
        self.callbacks = Callbacks(hparams)()

        self.model = STFPMModel(hparams)
        self.loss_val = 0
        self.anomaly_map_generator = AnomalyMapGenerator(batch_size=1, image_size=hparams.dataset.crop_size)

    def configure_optimizers(self):
        """Configure optimizers by creating an SGD optimizer.

        :return: SGD optimizer

        Args:

        Returns:

        """
        return optim.SGD(
            params=self.model.student_model.parameters(),
            lr=self.hparams.model.lr,
            momentum=self.hparams.model.momentum,
            weight_decay=self.hparams.model.weight_decay,
        )

    def training_step(self, batch, _):
        """Training Step of STFPM..
        For each batch, teacher and student and teacher features
            are extracted from the CNN.

        Args:
          batch: Input batch
          _: Index of the batch.

        Returns:
          Hierarchical feature map

        """
        self.model.teacher_model.eval()
        teacher_features, student_features = self.model.forward(batch["image"])
        loss = self.loss_val + self.model.loss(teacher_features, student_features)
        self.loss_val = 0
        return {"loss": loss}

    def validation_step(self, batch, _):
        """Validation Step of STFPM.
            Similar to the training step, student/teacher features
            are extracted from the CNN for each batch, and anomaly
            map is computed.

        Args:
          batch: Input batch
          _: Index of the batch.

        Returns:
          Dictionary containing images, anomaly maps, true labels and masks.
          These are required in `validation_epoch_end` for feature concatenation.

        """
        filenames, images, labels, masks = batch["image_path"], batch["image"], batch["label"], batch["mask"]
        teacher_features, student_features = self.model.forward(images)
        anomaly_maps = self.model.anomaly_map_generator(teacher_features, student_features)

        return {
            "filenames": filenames,
            "images": images,
            "true_labels": labels.cpu().numpy(),
            "true_masks": masks.squeeze().cpu().numpy(),
            "anomaly_maps": anomaly_maps,
        }

    def test_step(self, batch, _):
        """Test Step of STFPM.
            Similar to the training and validation step, student/teacher
            features are extracted from the CNN for each batch, and anomaly
            map is computed.

        Args:
          batch: Input batch
          _: Index of the batch.

        Returns:
          Dictionary containing images, features, true labels and masks.
          These are required in `validation_epoch_end` for feature concatenation.

        """
        return self.validation_step(batch, _)

    def validation_epoch_end(self, outputs):
        """Compute image and pixel level roc scores.

        Args:
          outputs: Batch of outputs from the validation step

        Returns:

        """

        self.filenames = [Path(f) for x in outputs for f in x["filenames"]]
        self.images = [x["images"] for x in outputs]

        self.true_masks = np.stack([output["true_masks"] for output in outputs])
        self.anomaly_maps = np.stack([output["anomaly_maps"] for output in outputs])

        self.true_labels = np.stack([output["true_labels"] for output in outputs])
        self.pred_labels = self.anomaly_maps.reshape(self.anomaly_maps.shape[0], -1).max(axis=1)

        self.image_roc_auc = roc_auc_score(self.true_labels, self.pred_labels)
        self.pixel_roc_auc = roc_auc_score(self.true_masks.flatten(), self.anomaly_maps.flatten())

        self.log(name="Image-Level AUC", value=self.image_roc_auc, on_epoch=True, prog_bar=True)
        self.log(name="Pixel-Level AUC", value=self.pixel_roc_auc, on_epoch=True, prog_bar=True)
