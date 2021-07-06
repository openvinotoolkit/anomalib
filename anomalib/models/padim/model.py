"""
PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization
https://arxiv.org/abs/2011.08785
"""
import os
import os.path
from pathlib import Path
from random import sample
from typing import Any, Sequence, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from scipy.ndimage import gaussian_filter
from skimage.segmentation import mark_boundaries
from sklearn.metrics import roc_auc_score
from torch import Tensor

from anomalib.core.callbacks.model_loader import LoadModelCallback
from anomalib.core.model.feature_extractor import FeatureExtractor
from anomalib.core.model.multi_variate_gaussian import MultiVariateGaussian
from anomalib.core.utils.anomaly_map_generator import BaseAnomalyMapGenerator
from anomalib.datasets.utils import Denormalize
from anomalib.models.base.model import BaseAnomalySegmentationLightning
from anomalib.utils.visualizer import Visualizer

__all__ = ["PADIMLightning"]


DIMS = {"resnet18": {"t_d": 448, "d": 100}, "wide_resnet50_2": {"t_d": 1792, "d": 550}}


class PadimModel(torch.nn.Module):
    """
    Padim Module
    """

    def __init__(self, backbone: str, layers: List[str]):
        super().__init__()
        self.backbone = getattr(torchvision.models, backbone)
        self.layers = layers
        self.feature_extractor = FeatureExtractor(backbone=self.backbone(pretrained=True), layers=self.layers)
        self.gaussian = MultiVariateGaussian()
        self.dims = DIMS[backbone]
        self.idx = torch.tensor(sample(range(0, DIMS[backbone]["t_d"]), DIMS[backbone]["d"]))

    def forward(self, input_tensor: Tensor) -> Dict[str, Tensor]:
        """Forward-pass image-batch (N, C, H, W) into model to extract features.

        Args:
                input_tensor: Image-batch (N, C, H, W)
                input_tensor: Tensor:

        Returns:
                Features from single/multiple layers.

                :Example:

        >>> x = torch.randn(32, 3, 224, 224)
        >>> features = self.extract_features(input_tensor)
        >>> features.keys()
        dict_keys(['layer1', 'layer2', 'layer3'])

        >>> [v.shape for v in features.values()]
        [torch.Size([32, 64, 56, 56]),
         torch.Size([32, 128, 28, 28]),
         torch.Size([32, 256, 14, 14])]
        """
        with torch.no_grad():
            features = self.feature_extractor(input_tensor)

        return features

    def append_features(self, outputs: List[Dict[str, Any]]) -> Dict[str, List[Tensor]]:
        """append_features from each batch to concatenate

        Args:
                outputs: description]
                outputs: List[Dict[str:Tensor]]:

        Returns:
                description]

        """
        features: Dict[str, List[Tensor]] = {layer: [] for layer in self.layers}
        for batch in outputs:
            for layer in self.layers:
                features[layer].append(batch["features"][layer].detach())

        return features

    @staticmethod
    def concat_features(features: Dict[str, List[Tensor]]) -> Dict[str, Tensor]:
        """Concatenate batch features to form one big feauture matrix.

        Args:
                        features: Features from batches.
                        features: Dict[str:
                        List[Tensor]]:

        Returns:
                        Concatenated feature map.

        """
        concatenated_features: Dict[str, Tensor] = {}
        for layer, feature_list in features.items():
            concatenated_features[layer] = torch.cat(tensors=feature_list, dim=0)

        return concatenated_features

    def generate_embedding(self, features: Dict[str, Tensor]) -> Tensor:
        """Generate embedding from hierarchical feature map

        Args:
                features: Hierarchical feature map from a CNN (ResNet18 or WideResnet)
                features: Dict[str:
                Tensor]:

        Returns:
                Embedding vector

        """

        def __generate_patch_embedding(embedding: Tensor, layer_features: Tensor) -> Tensor:
            """
            Generate patch embedding via pixel patches. A quote from Section IIIA from the paper:

            "As activation maps have a lower resolution  than  the  input  image,
            many  pixels  have  the  same embeddings  and  then  form  pixel  patches
            with  no  overlap  in the  original  image  resolution.  Hence,  an  input
            image  can  be divided  in  a  grid  of (i,j) ∈ [1,W] × [1,H] positions  where
            WxH is  the  resolution  of  the  largest  activation  map  used  to
            generate embeddings."

            :param embedding: Embedding vector from the earlier layers
            :param layer_features: Feature map from the subsequent layer.
            :return:
            """
            device = embedding.device
            batch_x, channel_x, height_x, width_x = embedding.size()
            _, channel_y, height_y, width_y = layer_features.size()
            stride = height_x // height_y
            embedding = F.unfold(embedding, kernel_size=stride, stride=stride)
            embedding = embedding.view(batch_x, channel_x, -1, height_y, width_y)
            updated_embedding = torch.zeros(
                size=(batch_x, channel_x + channel_y, embedding.size(2), height_y, width_y), device=device
            )

            for i in range(embedding.size(2)):
                updated_embedding[:, :, i, :, :] = torch.cat((embedding[:, :, i, :, :], layer_features), 1)
            updated_embedding = updated_embedding.view(batch_x, -1, height_y * width_y)
            updated_embedding = F.fold(
                updated_embedding, kernel_size=stride, output_size=(height_x, width_x), stride=stride
            )

            return updated_embedding

        def __reduce_embedding_dimension(embedding: Tensor, idx: Tensor) -> Tensor:
            """
            Reduce the dimension of the embedding via Random Sampling.

            :param embedding: Embedding vector extracted from the feature maps.
            :param idx: Randomly generated index values from which to sample.
            :return: Updated embedding vector with fewer dimensionality.
            """
            idx = idx.to(embedding.device)
            embedding = torch.index_select(embedding, 1, idx)
            return embedding

        embedding_vectors = features[self.layers[0]]
        for layer in self.layers[1:]:
            embedding_vectors = __generate_patch_embedding(embedding_vectors, features[layer])

        embedding_vectors = __reduce_embedding_dimension(embedding_vectors, self.idx)

        return embedding_vectors


class Callbacks:
    """PADIM-specific callbacks"""

    def __init__(self, config: DictConfig):
        self.config = config

    def get_callbacks(self) -> Sequence:
        """Get PADIM model callbacks."""
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


class AnomalyMapGenerator(BaseAnomalyMapGenerator):
    """Generate Anomaly Heatmap"""

    def __init__(self, image_size: int = 224, alpha: float = 0.4, gamma: int = 0, sigma: int = 4):
        super().__init__(alpha=alpha, gamma=gamma, sigma=sigma)
        self.image_size = image_size

    @staticmethod
    def compute_distance(embedding: Tensor, stats: List[Tensor]) -> Tensor:
        """Compute anomaly score to the patch in position(i,j)of a test image
        Ref: Equation (2), Section III-C of the paper.

        Args:
                embedding: Embedding Vector
                stats: Mean and Covariance Matrix of the multivariate
        Gaussian distribution
                embedding: Tensor:
                stats: List[Tensor]:

        Returns:
                Anomaly score of a test image via mahalanobis distance.

        """

        def _mahalanobis(tensor_u: Tensor, tensor_v: Tensor, inv_cov: Tensor) -> Tensor:
            """Compute the Mahalanobis distance between two 1-D arrays.
            The Mahalanobis distance between 1-D arrays `u` and `v`, is defined as

            .. math::
            \\sqrt{ (u-v) V^{-1} (u-v)^T }

            where ``V`` is the covariance matrix.  Note that the argument `VI`
            is the inverse of ``V``.

            Args:
                    tensor_u: Input array
                    tensor_v: Input array
                    inv_cov: Inverse covariance matrix
                    tensor_u: Tensor:
                    tensor_v: Tensor:
                    inv_cov: Tensor:

            Returns:
                    Mahalanobis distance of the inputs.

            """
            delta = tensor_u - tensor_v
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
        """Up sample anomaly score to match the input image size.

        Args:
                distance: Anomaly score computed via the mahalanobis distance.
                distance: Tensor:

        Returns:
                Resized distance matrix matching the input image size

        """

        score_map = (
            F.interpolate(distance.unsqueeze(1), size=self.image_size, mode="bilinear", align_corners=False)
            .squeeze()
            .cpu()
            .numpy()
        )
        return score_map

    def smooth_anomaly_map(self, anomaly_map: np.ndarray) -> np.ndarray:
        """Apply gaussian smoothing to the anomaly map

        Args:
                anomaly_map: Anomaly score for the test image(s)
                anomaly_map: np.ndarray:

        Returns:
                Filtered anomaly scores

        """

        for i in range(anomaly_map.shape[0]):
            anomaly_map[i] = gaussian_filter(anomaly_map[i], sigma=self.sigma)

        return anomaly_map

    def compute_anomaly_map(self, embedding: Tensor, mean: Tensor, covariance: Tensor) -> np.ndarray:
        """Compute anomaly score based on embedding vector, mean and covariance of the multivariate
        gaussian distribution.

        Args:
                        embedding: Embedding vector extracted from the test set.
                        mean: Mean of the multivariate gaussian distribution
                        covariance: Covariance matrix of the multivariate gaussian distribution.
                        embedding: Tensor:
                        mean: Tensor:
                        covariance: Tensor:

        Returns:
                Output anomaly score.

        """

        score_map = self.compute_distance(embedding, stats=[mean, covariance])
        up_sampled_score_map = self.up_sample(score_map)
        smoothed_anomaly_map = self.smooth_anomaly_map(up_sampled_score_map)

        return smoothed_anomaly_map


class PADIMLightning(BaseAnomalySegmentationLightning):
    """
    PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization
    """

    def __init__(self, hparams):
        super().__init__(hparams)
        self.layers = hparams.model.layers
        self._model = PadimModel(hparams.model.backbone, hparams.model.layers).eval()

        self.anomaly_map_generator = AnomalyMapGenerator()
        self.callbacks = Callbacks(hparams)()
        self.stats: List[Tensor, Tensor] = []
        self.automatic_optimization = False

    @staticmethod
    def configure_optimizers():
        """PADIM doesn't require optimization, therefore returns no optimizers."""
        return None

    def training_step(self, batch, _):
        """Training Step of PADIM.
        For each batch, hierarchical features are extracted from the CNN.

        Args:
                batch: Input batch
                _: Index of the batch.

        Returns:
                Hierarchical feature map

        """
        self._model.eval()
        features = self._model(batch["image"])
        return {"features": features}

    def validation_step(self, batch, _):
        """Validation Step of PADIM.
                        Similar to the training step, hierarchical features
                        are extracted from the CNN for each batch.

        Args:
                batch: Input batch
                _: Index of the batch.

        Returns:
                Dictionary containing images, features, true labels and masks.
                These are required in `validation_epoch_end` for feature concatenation.

        """
        filenames, images, labels, masks = batch["image_path"], batch["image"], batch["label"], batch["mask"]
        features = self._model(images)
        return {
            "filenames": filenames,
            "images": images,
            "features": features,
            "true_labels": labels.cpu().numpy(),
            "true_masks": masks.squeeze().cpu().numpy(),
        }

    def test_step(self, batch, _):
        """Test Step of PADIM.
                        Similar to the training and validation steps,
                        hierarchical features are extracted from the
                        CNN for each batch.

        Args:
                batch: Input batch
                _: Index of the batch.

        Returns:
                Dictionary containing images, features, true labels and masks.
                These are required in `validation_epoch_end` for feature concatenation.

        """
        return self.validation_step(batch, _)

    def training_epoch_end(self, outputs):
        """Fit a multivariate gaussian model on an embedding extracted from deep hierarchical CNN features.

        Args:
                outputs: Batch of outputs from the training step

        Returns:

        """

        features = self._model.append_features(outputs)
        features = self._model.concat_features(features)
        embedding = self._model.generate_embedding(features)
        self.stats = self._model.gaussian.fit(embedding)

    def validation_epoch_end(self, outputs):
        """Compute anomaly scores of the validation set, based on the embedding
                        extracted from deep hierarchical CNN features.

        Args:
                outputs: Batch of outputs from the validation step

        Returns:

        """
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

        self.image_roc_auc = roc_auc_score(self.true_labels, self.pred_labels)
        self.pixel_roc_auc = roc_auc_score(self.true_masks.flatten(), self.anomaly_maps.flatten())

        self.log(name="Image-Level AUC", value=self.image_roc_auc, on_epoch=True, prog_bar=True)
        self.log(name="Pixel-Level AUC", value=self.pixel_roc_auc, on_epoch=True, prog_bar=True)

    def test_epoch_end(self, outputs):
        """
        Compute and save anomaly scores of the test set, based on the embedding
            extracted from deep hierarchical CNN features.

        Args:
            outputs: Batch of outputs from the validation step

        """
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
