import os
import os.path
from typing import Callable, Dict, Iterable
from typing import List

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from sklearn.metrics import roc_auc_score
from torch import Tensor, nn, optim

__all__ = ["FeatureExtractor", "Loss", "AnomalyMapGenerator", "STFPMModel"]


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

    def __init__(self, backbone: nn.Module, layers: Iterable[str]):
        super().__init__()
        self.backbone = backbone
        self.layers = layers
        self._features = {layer: torch.empty(0) for layer in self.layers}

        for layer_id in layers:
            layer = dict([*self.backbone.named_modules()])[layer_id]
            layer.register_forward_hook(self.get_features(layer_id))

    def get_features(self, layer_id: str) -> Callable:
        def hook(_, __, output):
            self._features[layer_id] = output

        return hook

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        self._features = {layer: torch.empty(0) for layer in self.layers}
        _ = self.backbone(x)
        return self._features


class Loss(nn.Module):
    """
    Feature Pyramid Loss
    This class implmenents the feature pyramid loss function proposed in STFPM [1] paper.

    :Example:

    >>> from model import FeatureExtractor
    >>> from model import Loss
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

    def compute_layer_loss(self, teacher_feats: Tensor, student_feats: Tensor) -> Tensor:
        height, width = teacher_feats.shape[2:]

        norm_teacher_features = F.normalize(teacher_feats)
        norm_student_features = F.normalize(student_feats)
        layer_loss = (0.5 / (width * height)) * self.mse_loss(norm_teacher_features, norm_student_features)

        return layer_loss

    def forward(self, teacher_features: Dict[str, Tensor], student_features: Dict[str, Tensor]) -> Tensor:
        layer_losses: List[Tensor] = []
        for layer in teacher_features.keys():
            loss = self.compute_layer_loss(teacher_features[layer], student_features[layer])
            layer_losses.append(loss)

        total_loss = torch.stack(layer_losses).sum()

        return total_loss


class Callbacks:
    def __init__(self, args):
        self.args = args

    def get_callbacks(self) -> List[Callback]:
        checkpoint = ModelCheckpoint(
            dirpath=os.path.join(self.args.project_path, "weights"),
            # dirpath=os.path.join(self.args.project_path, self.args.dataset, os.path.split(self.args.dataset_path)[-1]),
            # filename="model-epoch{epoch:02d}-val_loss{val_loss:.2f}",
            filename="model",
            monitor=self.args.metric,
        )
        # checkpoint = ModelCheckpoint()
        early_stopping = EarlyStopping(monitor=self.args.metric, patience=self.args.patience)
        callbacks = [checkpoint, early_stopping]
        return callbacks

    def __call__(self):
        return self.get_callbacks()


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

    def compute_anomaly_map(self, teacher_features: Dict[str, Tensor], student_features: Dict[str, Tensor]) -> Tensor:
        # TODO: Reshape anomaly_map to handle batch_size > 1
        # TODO: Use torch tensor instead
        anomaly_map = np.ones([self.image_size, self.image_size])
        # device = list(teacher_features.values())[0].device
        # anomaly_map = torch.empty((self.image_size, self.image_size), device=device)
        for layer in teacher_features.keys():
            layer_map = self.compute_layer_map(teacher_features[layer], student_features[layer])
            layer_map = layer_map[0, 0, :, :]
            anomaly_map *= layer_map.cpu().detach().numpy()
            # anomaly_map *= layer_map

        return anomaly_map

    @staticmethod
    def compute_heatmap(anomaly_map: np.ndarray) -> np.ndarray:
        anomaly_map = (anomaly_map - anomaly_map.min()) / np.ptp(anomaly_map)
        anomaly_map = anomaly_map * 255
        anomaly_map = anomaly_map.astype(np.uint8)

        heatmap = cv2.applyColorMap(anomaly_map, cv2.COLORMAP_JET)
        return heatmap

    def apply_heatmap_on_image(self, anomaly_map: np.ndarray, image: np.ndarray) -> np.ndarray:
        heatmap = self.compute_heatmap(anomaly_map)
        heatmap_on_image = cv2.addWeighted(heatmap, self.alpha, image, self.beta, self.gamma)
        return heatmap_on_image

    def __call__(self, teacher_features: Dict[str, Tensor], student_features: Dict[str, Tensor]) -> np.ndarray:
        return self.compute_anomaly_map(teacher_features, student_features)


class STFPMModel(pl.LightningModule):
    def __init__(self, hparams):

        super().__init__()
        self.save_hyperparameters(hparams)
        self.backbone = getattr(torchvision.models, hparams.backbone)
        self.layers = hparams.layers

        self.teacher_model = FeatureExtractor(backbone=self.backbone(pretrained=True), layers=self.layers)
        self.student_model = FeatureExtractor(backbone=self.backbone(pretrained=False), layers=self.layers)

        # teacher model is fixed
        for parameters in self.teacher_model.parameters():
            parameters.requires_grad = False

        self.loss = Loss()
        self.anomaly_map_generator = AnomalyMapGenerator(batch_size=1, image_size=256)
        self.callbacks = Callbacks(hparams)()

    def forward(self, images):
        self.teacher_model.eval()
        teacher_features: Dict[str, Tensor] = self.teacher_model(images)
        student_features: Dict[str, Tensor] = self.student_model(images)
        return teacher_features, student_features

    def configure_optimizers(self):
        return optim.SGD(
            params=self.student_model.parameters(),
            lr=self.hparams.lr,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay,
        )

    def training_step(self, batch, batch_idx):
        teacher_features, student_features = self.forward(batch["image"])
        loss = self.loss(teacher_features, student_features)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        images, mask = batch["image"], batch["mask"]

        teacher_features, student_features = self.forward(images)
        loss = self.loss(teacher_features, student_features)

        anomaly_map = self.anomaly_map_generator(teacher_features, student_features)
        auc = roc_auc_score(mask.cpu().numpy().ravel(), anomaly_map.ravel())
        # auc = roc_auc_score(mask.cpu().numpy().ravel(), anomaly_map.cpu().numpy().ravel())

        # image_path, mask_path = batch["image_path"][0], batch["mask_path"][0]
        # images, masks = batch["image"], batch["mask"]
        #
        # defect_type = Path(image_path).parent.name
        # image_filename = Path(image_path).stem
        #
        # original_image = cv2.imread(image_path)
        # original_image = cv2.resize(original_image, (256, 256))
        #
        # heatmap_on_image = self.anomaly_map_generator.apply_heatmap_on_image(anomaly_map, original_image)
        #
        # cv2.imwrite(str(Path("./results") / f"{defect_type}_{image_filename}.jpg"), original_image)
        # cv2.imwrite(str(Path("./results") / f"{defect_type}_{image_filename}_heatmap.jpg"), heatmap_on_image)
        # cv2.imwrite(str(Path("./results") / f"{defect_type}_{image_filename}_mask.jpg"), masks.cpu().numpy())

        return {"val_loss": loss, "auc": auc}

    def validation_epoch_end(self, outputs):
        loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        auc = np.stack([x["auc"] for x in outputs]).mean()
        self.log(name="val_loss", value=loss, on_epoch=True, prog_bar=True)
        self.log(name="auc", value=auc, on_epoch=True, prog_bar=True)

    # @staticmethod
    # def add_model_specific_args(parent_parser):
    #     parser = parent_parser.add_argument_group("LitModel")
    #     parser.add_argument("--backbone", type=str, default="resnet18")
    #     parser.add_argument("--layers", nargs="+", default=["layer1", "layer2", "layer3"])
    #     parser.add_argument("--num_epochs", type=int, default=100)
    #     parser.add_argument("--batch_size", type=int, default=32)
    #     parser.add_argument("--num_workers", type=int, default=36)
    #     parser.add_argument("--lr", type=float, default=0.4)
    #     parser.add_argument("--momentum", type=float, default=0.9)
    #     parser.add_argument("--weight_decay", type=float, default=1e-4)
    #     parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    #     return parent_parser
