from argparse import Namespace
from typing import Callable, Dict, Iterable, List, Optional

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision
from torch import Tensor, nn, optim

__all__ = ["FeatureExtractor", "FeaturePyramidLoss", "AnomalyMapGenerator", "StudentTeacherFeaturePyramidMatching"]


class FeatureExtractor(nn.Module):
    """
    FeatureExtractor [summary]

    :param nn: [description]
    :type nn: [type]

    :Example:

    >>> import torch
    >>> import torchvision
    >>> from model import FeatureExtractor

    >>> model = FeatureExtractor(model=torchvision.models.resnet18(), layers=['layer1', 'layer2', 'layer3'])
    >>> input = torch.rand((32, 3, 256, 256))
    >>> features = model(input)

    >>> [layer for layer in features.keys()]
        ['layer1', 'layer2', 'layer3']
    >>> [feature.shape for feature in features.values()]
        [torch.Size([32, 64, 64, 64]), torch.Size([32, 128, 32, 32]), torch.Size([32, 256, 16, 16])]
    """

    def __init__(self, model: nn.Module, layers: Iterable[str]):
        super().__init__()
        self.model = model
        self.layers = layers
        self._features = {layer: torch.empty(0) for layer in self.layers}
        # self._features: List[Tensor] = []

        for layer_id in layers:
            layer = dict([*self.model.named_modules()])[layer_id]
            layer.register_forward_hook(self.get_features(layer_id))

    def get_features(self, layer_id: str) -> Callable:
        def hook(_, __, output):
            self._features[layer_id] = output
            # self._features.append(output)

        return hook

    # def forward(self, x: Tensor) -> List[Tensor]:
    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        self._features = {layer: torch.empty(0) for layer in self.layers}
        # self._features: List[Tensor] = []
        _ = self.model(x)
        return self._features


class FeaturePyramidLoss(nn.Module):
    """
    Feature Pyramid Loss
    This class implmenents the feature pyramid loss function proposed in STFPM [1] paper.

    :Example:

    >>> from model import FeatureExtractor
    >>> from model import FeaturePyramidLoss
    >>> from torchvision.models import resnet18

    >>> layers = ['layer1', 'layer2', 'layer3']
    >>> teacher_model = FeatureExtractor(model=resnet18(pretrained=True), layers=layers)
    >>> student_model = FeatureExtractor(model=resnet18(pretrained=False), layers=layers)
    >>> loss = FeaturePyramidLoss()

    >>> input = torch.rand((4, 3, 256, 256))
    >>> teacher_features = teacher_model(input)
    >>> student_features = student_model(input)
    >>> loss(student_features, teacher_features)
        tensor(51.2015, grad_fn=<SumBackward0>)
    """

    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss(reduction="sum")
        # self.cosine_similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-08)

    def compute_layer_loss(self, teacher_feats: Tensor, student_feats: Tensor) -> Tensor:
        height, width = teacher_feats.shape[2:]

        norm_teacher_features = F.normalize(teacher_feats)
        norm_student_features = F.normalize(student_feats)
        layer_loss = (0.5 / (width * height)) * self.mse_loss(norm_teacher_features, norm_student_features)

        return layer_loss
        # layer_loss = self.cosine_similarity(teacher_feats, student_feats)  # Eq (1)
        # layer_loss = torch.mean(layer_loss, dim=[1, 2])  # Eq (2)
        #
        # return layer_loss

    def forward(self, teacher_features: Dict[str, Tensor], student_features: Dict[str, Tensor]) -> Tensor:
        layer_losses: List[Tensor] = []
        for layer in teacher_features.keys():
            loss = self.compute_layer_loss(teacher_features[layer], student_features[layer])
            layer_losses.append(loss)

        total_loss = sum(layer_losses)

        # layer_losses = torch.stack(losses, dim=1)  # Eq (3): (NxL)
        # total_loss = layer_losses.sum()

        return total_loss


class AnomalyMapGenerator:
    def __init__(self, batch_size: int = 1, image_size: int = 256, alpha: float = 0.4, gamma: int = 0):
        super(AnomalyMapGenerator, self).__init__()
        self.distance = torch.nn.PairwiseDistance(p=2, keepdim=True)
        self.image_size = image_size
        self.batch_size = batch_size

        self.alpha = alpha
        self.beta = 1 - self.alpha
        self.gamma = gamma

    def compute_layer_map(self, teacher_features: Tensor, student_features: Tensor) -> Tensor:
        norm_teacher_features = F.normalize(teacher_features)
        norm_student_features = F.normalize(student_features)

        layer_map = 0.5 * self.distance(norm_student_features, norm_teacher_features) ** 2
        layer_map = F.interpolate(layer_map, size=self.image_size, align_corners=False, mode="bilinear")
        return layer_map

    def compute_anomaly_map(self, teacher_features: Dict[str, Tensor], student_features: Dict[str, Tensor]):
        # TODO: Reshape anomaly_map to handle batch_size > 1
        anomaly_map = np.ones([self.image_size, self.image_size])
        for layer in teacher_features.keys():
            layer_map = self.compute_layer_map(teacher_features[layer], student_features[layer])
            layer_map = layer_map[0, 0, :, :].to("cpu").detach().numpy()  # check

            anomaly_map *= layer_map
        return anomaly_map

    @staticmethod
    def compute_heatmap(anomaly_map: np.ndarray):
        anomaly_map = (anomaly_map - anomaly_map.min()) / np.ptp(anomaly_map)
        anomaly_map = anomaly_map * 255
        anomaly_map = anomaly_map.astype(np.uint8)

        heatmap = cv2.applyColorMap(anomaly_map, cv2.COLORMAP_JET)
        return heatmap

    def apply_heatmap_on_image(self, heatmap, image):
        return cv2.addWeighted(heatmap, self.alpha, image, self.beta, self.gamma)


class StudentTeacherFeaturePyramidMatching(pl.LightningModule):
    def __init__(self, hparams: Namespace, model: Optional[Callable] = None, layers: Optional[List[str]] = None):
        super().__init__()
        self.hparams = hparams
        # TODO: model and layers are init parameters.
        # self.model = getattr(torchvision.models, hparams.model)
        self.model = torchvision.models.resnet18
        self.layers = ["layer1", "layer2", "layer3"]

        self.teacher_model = FeatureExtractor(model=self.model(pretrained=True), layers=self.layers)
        self.student_model = FeatureExtractor(model=self.model(pretrained=False), layers=self.layers)
        self.loss = FeaturePyramidLoss()

        # teacher model is fixed
        for parameters in self.teacher_model.parameters():
            parameters.requires_grad = False

    def forward(self, images):
        return self.student_model(images)

    def configure_optimizers(self):
        return optim.SGD(
            params=self.student_model.parameters(),
            lr=self.hparams.lr,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay,
        )

    def training_step(self, batch, batch_idx):
        teacher_features: Dict[str, Tensor] = self.teacher_model(batch["image"])
        student_features: Dict[str, Tensor] = self.forward(batch["image"])
        loss = self.loss(teacher_features, student_features)
        self.log(name="loss", value=loss, on_step=False, on_epoch=True, prog_bar=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        images, mask = batch["image"], batch["mask"]
        teacher_features = self.teacher_model(images)
        student_features = self.forward(images)
        loss = self.loss(teacher_features, student_features)
        self.log(name="val_loss", value=loss, on_step=False, on_epoch=True, prog_bar=True)
        # return loss
        return {"val_loss": loss, "log": {"val_loss": loss}}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        log = {"train_loss": avg_loss}
        self.log(name="avg_train_loss", value=avg_loss, prog_bar=True)
        # return {"avg_train_loss": avg_loss, "log": log}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        self.log(name="val_loss", value=avg_loss)
        log = {"val_loss": avg_loss}
        # return {"avg_val_loss": avg_loss, "log": log}

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitModel")
        parser.add_argument("--model", type=str, default="resnet18")
        parser.add_argument("--layers", nargs="+", default=["layer1", "layer2", "layer3"])
        parser.add_argument("--num_epochs", type=int, default=100)
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--num_workers", type=int, default=36)
        parser.add_argument("--lr", type=float, default=0.4)
        parser.add_argument("--momentum", type=float, default=0.9)
        parser.add_argument("--weight_decay", type=float, default=1e-4)
        return parent_parser
