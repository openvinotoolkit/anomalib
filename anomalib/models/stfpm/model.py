"""
STFPM: Student-Teacher Feature Pyramid Matching for Unsupervised Anomaly Detection
https://arxiv.org/abs/2103.04257
"""

from typing import Dict, List, Tuple, Union

import torch
import torch.nn.functional as F
import torchvision
from omegaconf import ListConfig
from pytorch_lightning.callbacks import Callback, EarlyStopping
from torch import Tensor, nn, optim

from anomalib.core.model import AnomalyModule
from anomalib.core.model.feature_extractor import FeatureExtractor
from anomalib.datasets.tiler import Tiler

__all__ = ["Loss", "AnomalyMapGenerator", "STFPMModel", "StfpmLightning"]


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


class AnomalyMapGenerator:
    """Generate Anomaly Heatmap"""

    def __init__(
        self,
        image_size: Union[ListConfig, Tuple],
    ):
        self.distance = torch.nn.PairwiseDistance(p=2, keepdim=True)
        self.image_size = image_size if isinstance(image_size, tuple) else tuple(image_size)

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

        layer_map = 0.5 * torch.norm(norm_teacher_features - norm_student_features, p=2, dim=-3, keepdim=True) ** 2
        layer_map = F.interpolate(layer_map, size=self.image_size, align_corners=False, mode="bilinear")
        return layer_map

    def compute_anomaly_map(
        self, teacher_features: Dict[str, Tensor], student_features: Dict[str, Tensor]
    ) -> torch.Tensor:
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
        batch_size = list(teacher_features.values())[0].shape[0]
        anomaly_map = torch.ones(batch_size, 1, self.image_size[0], self.image_size[1])
        for layer in teacher_features.keys():
            layer_map = self.compute_layer_map(teacher_features[layer], student_features[layer])
            anomaly_map = anomaly_map.to(layer_map.device)
            anomaly_map *= layer_map

        return anomaly_map

    def __call__(self, **kwds: Dict[str, Tensor]) -> torch.Tensor:
        """
        Returns anomaly_map.
        Expects `teach_features` and `student_features` keywords to be passed explicitly

        Example
        >>> anomaly_map_generator = AnomalyMapGenerator(image_size=tuple(hparams.model.input_size))
        >>> output = self.anomaly_map_generator(teacher_features=teacher_features, student_features=student_features)

        Raises:
            ValueError: `teach_features` and `student_features` keys are not found

        Returns:
            torch.Tensor: anomaly map
        """

        if not ("teacher_features" in kwds and "student_features" in kwds):
            raise ValueError(f"Expected keys `teacher_features` and `student_features. Found {kwds.keys()}")

        teacher_features: Dict[str, Tensor] = kwds["teacher_features"]
        student_features: Dict[str, Tensor] = kwds["student_features"]

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
        self.hparams = hparams
        if hparams.dataset.tiling.apply:
            self.tiler = Tiler(hparams.dataset.tiling.tile_size, hparams.dataset.tiling.stride)
            self.anomaly_map_generator = AnomalyMapGenerator(image_size=tuple(hparams.dataset.tiling.tile_size))
        else:
            self.anomaly_map_generator = AnomalyMapGenerator(image_size=tuple(hparams.model.input_size))

    def forward(self, images):
        """
        Forward-pass images into the network. During the training mode
        the model extracts the features from the teacher and student networks.
        During the evaluation mode, it returns the predicted anomaly map.

        Args:
          images: Batch of images.

        Returns:
          Teacher and student features when in training mode, otherwise the predicted anomaly maps.

        """
        if self.hparams.dataset.tiling.apply:
            images = self.tiler.tile(images)
        teacher_features: Dict[str, Tensor] = self.teacher_model(images)
        student_features: Dict[str, Tensor] = self.student_model(images)
        if self.training:
            output = teacher_features, student_features
        else:
            output = self.anomaly_map_generator(teacher_features=teacher_features, student_features=student_features)
            if self.hparams.dataset.tiling.apply:
                output = self.tiler.untile(output)

        return output


class StfpmLightning(AnomalyModule):
    """
    PL Lightning Module for the STFPM algorithm.
    """

    def __init__(self, hparams):
        super().__init__(hparams)

        self.model = STFPMModel(hparams)
        self.loss_val = 0
        self.callbacks: List[Callback] = [
            EarlyStopping(
                monitor=self.hparams.model.early_stopping.metric,
                patience=self.hparams.model.early_stopping.patience,
                mode=self.hparams.model.early_stopping.mode,
            )
        ]

    def configure_optimizers(self):
        """
        Configure optimizers by creating an SGD optimizer.

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

    def training_step(self, batch, _):  # pylint: disable=arguments-differ
        """
        Training Step of STFPM..
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

    def validation_step(self, batch, _):  # pylint: disable=arguments-differ
        """
        Validation Step of STFPM. Similar to the training step, student/teacher
        features are extracted from the CNN for each batch, and anomaly map is computed.

        Args:
          batch: Input batch
          _: Index of the batch.

        Returns:
          Dictionary containing images, anomaly maps, true labels and masks.
          These are required in `validation_epoch_end` for feature concatenation.

        """
        batch["anomaly_maps"] = self.model(batch["image"])

        return batch
