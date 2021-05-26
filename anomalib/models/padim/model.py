import argparse
import os
import os.path
import pickle
import random
from random import sample
from typing import Dict
from typing import List
from typing import Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from scipy.ndimage import gaussian_filter
from skimage import morphology
from skimage.segmentation import mark_boundaries
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from torch import Tensor

from anomalib.datasets.utils import Denormalize
from anomalib.models.shared.feature_extractor import FeatureExtractor

__all__ = ["PADIMModel"]


random.seed(1024)
torch.manual_seed(1024)
torch.cuda.manual_seed_all(1024)

# # set transforms
# transform_x = T.Compose(
#     [
#         T.Resize(256, Image.ANTIALIAS),
#         T.CenterCrop(224),
#         T.ToTensor(),
#         T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ]
# )
# transform_mask = T.Compose([T.Resize(256, Image.NEAREST), T.CenterCrop(224), T.ToTensor()])


def parse_args():
    parser = argparse.ArgumentParser("PaDiM")
    parser.add_argument("--data_path", type=str, default="/home/sakcay/Projects/ote/anomalib/datasets/MVTec")
    parser.add_argument("--save_path", type=str, default="./results")
    parser.add_argument("--arch", type=str, choices=["resnet18", "wide_resnet50_2"], default="resnet18")
    return parser.parse_args()


np.cov


class MultiVariateGaussian(torch.nn.Module):
    @staticmethod
    def _cov(x: Tensor, rowvar: bool = False, bias: bool = False, ddof: Optional[int] = None, aweights=None) -> Tensor:
        """
        Estimates covariance matrix like numpy.cov
        :param x: A 1-D or 2-D array containing multiple variables and observations.
                        Each row of `m` represents a variable, and each column a single
                        observation of all those variables. Also see `rowvar` below.
        :param rowvar: If `rowvar` is True (default), then each row represents a
                        variable, with observations in the columns. Otherwise, the relationship
                        is transposed: each column represents a variable, while the rows
                        contain observations.
        :param bias: Default normalization (False) is by ``(N - 1)``, where ``N`` is the
                        number of observations given (unbiased estimate). If `bias` is True,
                        then normalization is by ``N``. These values can be overridden by using
                        the keyword ``ddof`` in numpy versions >= 1.5.
        :param ddof: If not ``None`` the default value implied by `bias` is overridden.
                        Note that ``ddof=1`` will return the unbiased estimate, even if both
                        `fweights` and `aweights` are specified, and ``ddof=0`` will return
                        the simple average. See the notes for the details. The default value
                        is ``None``.
        :param aweights: 1-D array of observation vector weights. These relative weights are
                        typically large for observations considered "important" and smaller for
                        observations considered less "important". If ``ddof=0`` the array of
                        weights can be used to assign probabilities to observation vectors.
        :return: The covariance matrix of the variables.
        """
        # ensure at least 2D
        if x.dim() == 1:
            x = x.view(-1, 1)

        # treat each column as a data point, each row as a variable
        if rowvar and x.shape[0] != 1:
            x = x.t()

        if ddof is None:
            if bias == 0:
                ddof = 1
            else:
                ddof = 0

        w = aweights
        if w is not None:
            if not torch.is_tensor(w):
                w = torch.tensor(w, dtype=torch.float)
            w_sum = torch.sum(w)
            avg = torch.sum(x * (w / w_sum)[:, None], 0)
        else:
            avg = torch.mean(x, 0)

        # Determine the normalization
        if w is None:
            fact = x.shape[0] - ddof
        elif ddof == 0:
            fact = w_sum
        elif aweights is None:
            fact = w_sum - ddof
        else:
            fact = w_sum - ddof * torch.sum(w * w) / w_sum

        xm = x.sub(avg.expand_as(x))

        if w is None:
            X_T = xm.t()
        else:
            X_T = torch.mm(torch.diag(w), xm).t()

        c = torch.mm(X_T, xm)
        c = c / fact

        return c.squeeze()

    def forward(self, embedding: Tensor) -> List[Tensor]:
        """
        Calculate multivariate Gaussian distribution
        :param embedding: CNN features whose dimensionality is reduced via either random sampling or PCA.
        :return: mean and covariance of the multi-variate gaussian distribution that fits the features.
        """
        device = embedding.device

        batch, channel, height, width = embedding.size()
        embedding_vectors = embedding.view(batch, channel, height * width)
        mean = torch.mean(embedding_vectors, dim=0)
        cov = torch.zeros(size=(channel, channel, height * width), device=device)
        identity = torch.eye(channel).to(device)
        for i in range(height * width):
            cov[:, :, i] = self._cov(embedding_vectors[:, :, i], rowvar=False) + 0.01 * identity

        return [mean, cov]

    def fit(self, embedding: Tensor) -> List[Tensor]:
        return self.forward(embedding)


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
    def __init__(self, args: DictConfig):
        self.args = args

    def get_callbacks(self) -> List[Callback]:
        checkpoint = ModelCheckpoint(
            dirpath=os.path.join(self.args.project_path, "weights"),
            filename="model",
        )
        early_stopping = EarlyStopping(monitor=self.args.metric, patience=self.args.patience)
        callbacks = [
            checkpoint,
            # early_stopping
        ]
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
    def compute_distance(embedding: Tensor, outputs: List[Tensor]) -> Tensor:
        def _mahalanobis(u: Tensor, v: Tensor, inverse_covariance: Tensor) -> Tensor:
            """
            Compute the Mahalanobis distance between two 1-D arrays.
            The Mahalanobis distance between 1-D arrays `u` and `v`, is defined as

            .. math::
            \\sqrt{ (u-v) V^{-1} (u-v)^T }

            where ``V`` is the covariance matrix.  Note that the argument `VI`
            is the inverse of ``V``.
            :param u:  Input array
            :param v:  Input array
            :param inverse_covariance: Inverse covariance matrix
            :return: Mahalanobis distance of the inputs.
            """
            delta = u - v
            mahalanobis_distance = torch.dot(torch.matmul(delta, inverse_covariance), delta)
            return torch.sqrt(mahalanobis_distance)

        batch, channel, height, width = embedding.shape
        embedding = embedding.reshape(batch, channel, height * width)

        distance_list = []
        for i in range(height * width):
            mean = outputs[0][:, i]
            inverse_covariance = torch.linalg.inv(outputs[1][:, :, i])
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

    @staticmethod
    def normalize(anomaly_map: np.ndarray) -> np.ndarray:
        max_score = anomaly_map.max()
        min_score = anomaly_map.min()
        scores = (anomaly_map - min_score) / (max_score - min_score)
        return scores

    def compute_anomaly_map(self, embedding: Tensor, stats: List[Tensor]) -> np.ndarray:
        score_map = self.compute_distance(embedding, stats)
        score_map = self.up_sample(score_map)
        score_map = self.smooth_anomaly_map(score_map)
        # score_map = self.normalize(score_map)

        return score_map

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
        # get optimal threshold
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


class Visualizer:
    def __init__(self, num_rows: int, num_cols: int, figure_size: Tuple[int, int]):
        self.figure_index: int = 0

        self.figure, self.axis = plt.subplots(num_rows, num_cols, figsize=figure_size)
        self.figure.subplots_adjust(right=0.9)

        for axis in self.axis:
            axis.axes.xaxis.set_visible(False)
            axis.axes.yaxis.set_visible(False)

    def add_image(self, index: int, image: np.ndarray, title: str, cmap: Optional[str] = None):
        self.axis[index].imshow(image, cmap)
        self.axis[index].title.set_text(title)

    def show(self):
        self.figure.show()

    def save(self, filename: str):
        self.figure.savefig(filename, dpi=100)

    def close(self):
        plt.close(self.figure)


class PADIMModel(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.layers = hparams.layers
        self._model = Padim(hparams.backbone, hparams.layers).eval()

        self.anomaly_map_generator = AnomalyMapGenerator()
        self.callbacks = Callbacks(hparams)()
        self.stats: List[Tensor, Tensor] = []

    def configure_optimizers(self):
        return None

    def training_step(self, batch, batch_idx):
        self._model.eval()
        features = self._model(batch["image"])
        return {"features": features}

    def validation_step(self, batch, batch_idx):
        filename, image, label, mask = batch["image_path"], batch["image"], batch["label"], batch["mask"]
        features = self._model(image)
        return {"filename": filename, "image": image, "features": features, "label": label, "mask": mask}

    def test_step(self, batch, batch_idx):
        pass

    def training_epoch_end(self, outputs):

        # TODO: Try to merge append and concat into one method.
        features = self._model.append_features(outputs)
        features = self._model.concat_features(features)
        embedding = self._model.generate_embedding(features)
        self.stats = self._model.gaussian.fit(embedding)

        train_outputs = self.stats
        with open("./results/pl_results.pkl", "wb") as f:
            pickle.dump(train_outputs, f)

    def validation_epoch_end(self, outputs):
        filenames = [f for x in outputs for f in x["filename"]]
        images = [x["image"] for x in outputs]
        true_labels = torch.stack([x["label"] for x in outputs])
        true_masks = torch.stack([x["mask"].squeeze() for x in outputs])

        test_features = self._model.append_features(outputs)
        test_features = self._model.concat_features(test_features)
        embedding = self._model.generate_embedding(test_features)
        anomaly_maps = self.anomaly_map_generator.compute_anomaly_map(embedding, self.stats)

        # Compute performance.
        pred_labels = anomaly_maps.reshape(anomaly_maps.shape[0], -1).max(axis=1)
        true_labels = np.asarray(true_labels.cpu())
        true_masks = np.asarray(true_masks.cpu())

        image_roc_auc = roc_auc_score(true_labels, pred_labels)
        pixel_roc_auc = roc_auc_score(true_masks.flatten(), anomaly_maps.flatten())

        self.log(name="Image-Level AUC", value=image_roc_auc, on_epoch=True, prog_bar=True)
        self.log(name="Pixel-Level AUC", value=pixel_roc_auc, on_epoch=True, prog_bar=True)

        save_dir = "./results/my_results"
        os.makedirs(save_dir, exist_ok=True)
        threshold = self.anomaly_map_generator.compute_adaptive_threshold(true_masks, anomaly_maps)

        for i, (filename, image, true_mask, anomaly_map) in enumerate(zip(filenames, images, true_masks, anomaly_maps)):
            image = Denormalize()(image.squeeze())

            heat_map = self.anomaly_map_generator.apply_heatmap_on_image(anomaly_map, image)
            pred_mask = self.anomaly_map_generator.compute_mask(anomaly_map=anomaly_map, threshold=threshold)
            vis_img = mark_boundaries(image, pred_mask, color=(1, 0, 0), mode="thick")

            visualizer = Visualizer(num_rows=1, num_cols=5, figure_size=(12, 3))
            visualizer.add_image(index=0, image=image, title="Image")
            visualizer.add_image(index=1, image=true_mask, cmap="gray", title="Ground Truth")
            visualizer.add_image(index=2, image=heat_map, title="Predicted Heat Map")
            visualizer.add_image(index=3, image=pred_mask, cmap="gray", title="Predicted Mask")
            visualizer.add_image(index=4, image=vis_img, title="Segmentation Result")
            visualizer.save(os.path.join(save_dir, os.path.split(filename)[1]))
            visualizer.close()

    def test_epoch_end(self, outputs):
        pass
