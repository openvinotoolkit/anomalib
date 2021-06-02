import argparse
import os
import os.path
from pathlib import Path
from random import sample
from typing import Dict, Optional, Union, List, Sequence

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from scipy.ndimage import gaussian_filter
from skimage import morphology
from skimage.segmentation import mark_boundaries
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from torch import Tensor

from anomalib.core.model.feature_extractor import FeatureExtractor
from anomalib.core.model.multi_variate_gaussian import MultiVariateGaussian
from anomalib.datasets.utils import Denormalize
from anomalib.utils.visualizer import Visualizer

__all__ = ["PADIMModel"]


def parse_args():
    parser = argparse.ArgumentParser("PaDiM")
    parser.add_argument("--data_path", type=str, default="/home/sakcay/Projects/ote/anomalib/datasets/MVTec")
    parser.add_argument("--save_path", type=str, default="./results")
    parser.add_argument("--arch", type=str, choices=["resnet18", "wide_resnet50_2"], default="resnet18")
    return parser.parse_args()


DIMS = {"resnet18": {"t_d": 448, "d": 100}, "wide_resnet50_2": {"t_d": 1792, "d": 550}}


class Padim(torch.nn.Module):
    def __init__(self, backbone: str, layers: List[str]):
        super().__init__()
        self.backbone = getattr(torchvision.models, backbone)
        self.layers = layers
        self.feature_extractor = FeatureExtractor(backbone=self.backbone(pretrained=True), layers=self.layers)
        self.gaussian = MultiVariateGaussian()
        self.dims = DIMS[backbone]
        self.idx = torch.tensor(sample(range(0, DIMS[backbone]["t_d"]), DIMS[backbone]["d"]))

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """
        Forward-pass image-batch (N, C, H, W) into model to extract features.

        :param x: Image-batch (N, C, H, W)
        :return: Features from single/multiple layers.

        :Example:

        >>> x = torch.randn(32, 3, 224, 224)
        >>> features = self.extract_features(x)
        >>> features.keys()
        dict_keys(['layer1', 'layer2', 'layer3'])

        >>> [v.shape for v in features.values()]
        [torch.Size([32, 64, 56, 56]),
         torch.Size([32, 128, 28, 28]),
         torch.Size([32, 256, 14, 14])]
        """
        with torch.no_grad():
            features = self.feature_extractor(x)

        return features

    def append_features(self, outputs: List[Dict[str, Tensor]]) -> Dict[str, List[Tensor]]:
        features: Dict[str, List[Tensor]] = {layer: [] for layer in self.layers}
        for batch in outputs:
            for layer in self.layers:
                features[layer].append(batch["features"][layer].detach())

        return features

    @staticmethod
    def concat_features(features: Dict[str, List[Tensor]]) -> Dict[str, Tensor]:
        concatenated_features: Dict[str, Tensor] = {}
        for layer, feature_list in features.items():
            concatenated_features[layer] = torch.cat(tensors=feature_list, dim=0)

        return concatenated_features

    def generate_embedding(self, features: Dict[str, Tensor]) -> Tensor:
        def __generate_patch_embedding(x: Tensor, y: Tensor) -> Tensor:
            device = x.device
            batch_x, channel_x, height_x, width_x = x.size()
            _, channel_y, height_y, width_y = y.size()
            stride = height_x // height_y
            x = F.unfold(x, kernel_size=stride, stride=stride)
            x = x.view(batch_x, channel_x, -1, height_y, width_y)
            z = torch.zeros(size=(batch_x, channel_x + channel_y, x.size(2), height_y, width_y), device=device)

            for i in range(x.size(2)):
                z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
            z = z.view(batch_x, -1, height_y * width_y)
            z = F.fold(z, kernel_size=stride, output_size=(height_x, width_x), stride=stride)

            return z

        def __reduce_embedding_dimension(embedding: Tensor, idx: Tensor) -> Tensor:
            idx = idx.to(embedding.device)
            embedding = torch.index_select(embedding, 1, idx)
            return embedding

        embedding_vectors = features[self.layers[0]]
        for layer in self.layers[1:]:
            embedding_vectors = __generate_patch_embedding(embedding_vectors, features[layer])

        embedding_vectors = __reduce_embedding_dimension(embedding_vectors, self.idx)

        return embedding_vectors


class Callbacks:
    def __init__(self, config: DictConfig):
        self.config = config

    def get_callbacks(self) -> Sequence:
        checkpoint = ModelCheckpoint(
            dirpath=os.path.join(self.config.project.path, "weights"),
            filename="model",
        )
        callbacks = [checkpoint]

        return callbacks

    def __call__(self):
        return self.get_callbacks()


class AnomalyMapGenerator:
    def __init__(self, image_size: int = 224, alpha: float = 0.4, gamma: int = 0, sigma: int = 4, kernel_size: int = 4):
        self.image_size = image_size
        self.sigma = sigma
        self.kernel_size = kernel_size
        self.alpha = alpha
        self.beta = 1 - self.alpha
        self.gamma = gamma

    @staticmethod
    def compute_distance(embedding: Tensor, stats: List[Tensor]) -> Tensor:
        def _mahalanobis(u: Tensor, v: Tensor, inv_cov: Tensor) -> Tensor:
            """
            Compute the Mahalanobis distance between two 1-D arrays.
            The Mahalanobis distance between 1-D arrays `u` and `v`, is defined as

            .. math::
            \\sqrt{ (u-v) V^{-1} (u-v)^T }

            where ``V`` is the covariance matrix.  Note that the argument `VI`
            is the inverse of ``V``.
            :param u:  Input array
            :param v:  Input array
            :param inv_cov: Inverse covariance matrix
            :return: Mahalanobis distance of the inputs.
            """
            delta = u - v
            mahalanobis_distance = torch.dot(torch.matmul(delta, inv_cov), delta)
            return torch.sqrt(mahalanobis_distance)

        batch, channel, height, width = embedding.shape
        embedding = embedding.reshape(batch, channel, height * width)

        distance_list = []
        for i in range(height * width):
            mean = stats[0][:, i]
            inverse_covariance = torch.linalg.inv(stats[1][:, :, i])
            distance = [_mahalanobis(emb[:, i], mean, inverse_covariance) for emb in embedding]
            distance_list.append(distance)

        distance_tensor = torch.tensor(distance_list).permute(1, 0).reshape(batch, height, width)
        return distance_tensor

    def up_sample(self, distance: Tensor) -> np.ndarray:
        score_map = (
            F.interpolate(distance.unsqueeze(1), size=self.image_size, mode="bilinear", align_corners=False)
            .squeeze()
            .cpu()
            .numpy()
        )
        return score_map

    def smooth_anomaly_map(self, anomaly_map: np.ndarray) -> np.ndarray:
        for i in range(anomaly_map.shape[0]):
            anomaly_map[i] = gaussian_filter(anomaly_map[i], sigma=self.sigma)

        return anomaly_map

    def compute_anomaly_map(self, embedding: Tensor, mean: Tensor, covariance: Tensor) -> np.ndarray:
        score_map = self.compute_distance(embedding, stats=[mean, covariance])
        up_sampled_score_map = self.up_sample(score_map)
        smoothed_anomaly_map = self.smooth_anomaly_map(up_sampled_score_map)

        return smoothed_anomaly_map

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
        heatmap_on_image = cv2.cvtColor(heatmap_on_image, cv2.COLOR_BGR2RGB)
        return heatmap_on_image

    @staticmethod
    def compute_adaptive_threshold(true_masks, anomaly_map):
        precision, recall, thresholds = precision_recall_curve(true_masks.flatten(), anomaly_map.flatten())
        a = 2 * precision * recall
        b = precision + recall
        f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
        threshold = thresholds[np.argmax(f1)]

        return threshold

    def compute_mask(self, anomaly_map: np.ndarray, threshold: float) -> np.ndarray:
        mask = np.zeros_like(anomaly_map).astype(np.uint8)
        mask[anomaly_map > threshold] = 1

        kernel = morphology.disk(self.kernel_size)
        mask = morphology.opening(mask, kernel)

        mask *= 255

        return mask


class PADIMModel(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.layers = hparams.model.layers
        self._model = Padim(hparams.model.backbone, hparams.model.layers).eval()

        self.anomaly_map_generator = AnomalyMapGenerator()
        self.callbacks = Callbacks(hparams)()
        self.stats: List[Tensor, Tensor] = []

        self.filenames: Optional[List[Union[str, Path]]] = None
        self.images: Optional[List[Union[np.ndarray, Tensor]]] = None

        self.true_masks: Optional[List[Union[np.ndarray, Tensor]]] = None
        self.anomaly_maps: Optional[List[Union[np.ndarray, Tensor]]] = None

        self.true_labels: Optional[List[Union[np.ndarray, Tensor]]] = None
        self.pred_labels: Optional[List[Union[np.ndarray, Tensor]]] = None

        self.image_roc_auc: Optional[float] = None
        self.pixel_roc_auc: Optional[float] = None

    def configure_optimizers(self):
        return None

    def training_step(self, batch, batch_idx):
        self._model.eval()
        features = self._model(batch["image"])
        return {"features": features}

    def validation_step(self, batch, batch_idx):
        filenames, images, labels, masks = batch["image_path"], batch["image"], batch["label"], batch["mask"]
        features = self._model(images)
        return {
            "filenames": filenames,
            "images": images,
            "features": features,
            "true_labels": labels.cpu().numpy(),
            "true_masks": masks.squeeze().cpu().numpy(),
        }

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def training_epoch_end(self, outputs):

        # TODO: Try to merge append and concat into one method.
        features = self._model.append_features(outputs)
        features = self._model.concat_features(features)
        embedding = self._model.generate_embedding(features)
        self.stats = self._model.gaussian.fit(embedding)

    def validation_epoch_end(self, outputs):
        self.filenames = [Path(f) for x in outputs for f in x["filenames"]]
        self.images = [x["images"] for x in outputs]

        self.true_masks = np.stack([x["true_masks"] for x in outputs])

        test_features = self._model.append_features(outputs)
        test_features = self._model.concat_features(test_features)

        embedding = self._model.generate_embedding(test_features)
        self.anomaly_maps = self.anomaly_map_generator.compute_anomaly_map(
            embedding=embedding, mean=self._model.gaussian.mean, covariance=self._model.gaussian.covariance
        )

        self.true_labels = np.stack([x["true_labels"] for x in outputs])
        self.pred_labels = self.anomaly_maps.reshape(self.anomaly_maps.shape[0], -1).max(axis=1)

        image_roc_auc = roc_auc_score(self.true_labels, self.pred_labels)
        pixel_roc_auc = roc_auc_score(self.true_masks.flatten(), self.anomaly_maps.flatten())

        self.log(name="Image-Level AUC", value=image_roc_auc, on_epoch=True, prog_bar=True)
        self.log(name="Pixel-Level AUC", value=pixel_roc_auc, on_epoch=True, prog_bar=True)

    def test_epoch_end(self, outputs):
        self.validation_epoch_end(outputs)
        threshold = self.anomaly_map_generator.compute_adaptive_threshold(self.true_masks, self.anomaly_maps)

        for (filename, image, true_mask, anomaly_map) in zip(
            self.filenames, self.images, self.true_masks, self.anomaly_maps
        ):
            image = Denormalize()(image.squeeze())

            heat_map = self.anomaly_map_generator.apply_heatmap_on_image(anomaly_map, image)
            pred_mask = self.anomaly_map_generator.compute_mask(anomaly_map=anomaly_map, threshold=threshold)
            vis_img = mark_boundaries(image, pred_mask, color=(1, 0, 0), mode="thick")

            visualizer = Visualizer(num_rows=1, num_cols=5, figure_size=(12, 3))
            visualizer.add_image(image=image, title="Image")
            visualizer.add_image(image=true_mask, color_map="gray", title="Ground Truth")
            visualizer.add_image(image=heat_map, title="Predicted Heat Map")
            visualizer.add_image(image=pred_mask, color_map="gray", title="Predicted Mask")
            visualizer.add_image(image=vis_img, title="Segmentation Result")
            visualizer.save(Path(self.hparams.project.path) / "images" / filename.parent.name / filename.name)
            visualizer.close()
