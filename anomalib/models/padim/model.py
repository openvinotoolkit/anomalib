"""
PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization
https://arxiv.org/abs/2011.08785
"""

from random import sample
from typing import Dict, List, Tuple, Union

import torch
import torch.nn.functional as F
import torchvision
from kornia import gaussian_blur2d
from omegaconf import ListConfig
from omegaconf.dictconfig import DictConfig
from torch import Tensor, nn

from anomalib.core.callbacks import get_callbacks
from anomalib.core.model.feature_extractor import FeatureExtractor
from anomalib.core.model.multi_variate_gaussian import MultiVariateGaussian
from anomalib.datasets.tiler import Tiler
from anomalib.models.base import SegmentationModule

__all__ = ["PadimLightning", "concat_layer_embedding"]


DIMS = {"resnet18": {"t_d": 448, "d": 100}, "wide_resnet50_2": {"t_d": 1792, "d": 550}}


def concat_layer_embedding(embedding: Tensor, layer_embedding: Tensor) -> Tensor:
    """
    Generate patch embedding via pixel patches. A quote from Section IIIA from the paper:

    "As activation maps have a lower resolution  than  the  input  image,
    many  pixels  have  the  same embeddings  and  then  form  pixel  patches
    with  no  overlap  in the  original  image  resolution.  Hence,  an  input
    image  can  be divided  in  a  grid  of (i,j) ∈ [1,W] × [1,H] positions  where
    WxH is  the  resolution  of  the  largest  activation  map  used  to
    generate embeddings."

    Args:
        embedding: Embedding vector from the earlier layers
        layer_embedding: Feature map from the subsequent layer.
    """
    device = embedding.device
    batch_x, channel_x, height_x, width_x = embedding.size()
    _, channel_y, height_y, width_y = layer_embedding.size()
    stride = height_x // height_y

    embedding = F.unfold(embedding, kernel_size=stride, stride=stride)
    embedding = embedding.view(batch_x, channel_x, -1, height_y, width_y)
    updated_embedding = torch.zeros(
        size=(batch_x, channel_x + channel_y, embedding.size(2), height_y, width_y),
        device=device,
    )

    for i in range(embedding.size(2)):
        updated_embedding[:, :, i, :, :] = torch.cat((embedding[:, :, i, :, :], layer_embedding), 1)
    updated_embedding = updated_embedding.view(batch_x, -1, height_y * width_y)
    updated_embedding = F.fold(updated_embedding, kernel_size=stride, output_size=(height_x, width_x), stride=stride)

    return updated_embedding


class PadimModel(nn.Module):
    """
    Padim Module
    """

    def __init__(self, hparams: DictConfig):
        super().__init__()
        self.hparams = hparams
        self.backbone = getattr(torchvision.models, hparams.model.backbone)
        self.layers = hparams.model.layers
        self.feature_extractor = FeatureExtractor(backbone=self.backbone(pretrained=True), layers=self.layers)
        self.gaussian = MultiVariateGaussian()
        self.dims = DIMS[hparams.model.backbone]
        # pylint: disable=not-callable
        # Since idx is randomaly selected, save it with model to get same results
        self.register_buffer(
            "idx",
            torch.tensor(sample(range(0, DIMS[hparams.model.backbone]["t_d"]), DIMS[hparams.model.backbone]["d"])),
        )
        self.idx: Tensor
        self.loss = None
        input_size = (
            hparams.transform.image_size if hparams.transform.crop_size is None else hparams.transform.crop_size
        )
        self.anomaly_map_generator = AnomalyMapGenerator(image_size=input_size)

        if hparams.dataset.tiling.apply:
            self.tiler = Tiler(hparams.dataset.tiling.tile_size, hparams.dataset.tiling.stride)

    def forward(self, input_tensor: Tensor) -> Tensor:
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
        if self.hparams.dataset.tiling.apply:
            input_tensor = self.tiler.tile(input_tensor)
        with torch.no_grad():
            features = self.feature_extractor(input_tensor)
            embeddings = self.generate_embedding(features)
        if self.hparams.dataset.tiling.apply:
            embeddings = self.tiler.untile(embeddings)
        if not self.training:
            return self.anomaly_map_generator(
                embedding=embeddings, mean=self.gaussian.mean, covariance=self.gaussian.covariance
            )

        return embeddings

    def generate_embedding(self, features: Dict[str, Tensor]) -> Tensor:
        """Generate embedding from hierarchical feature map

        Args:
                features: Hierarchical feature map from a CNN (ResNet18 or WideResnet)
                features: Dict[str:
                Tensor]:

        Returns:
                Embedding vector

        """

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
            embedding_vectors = concat_layer_embedding(embedding_vectors, features[layer])

        embedding_vectors = __reduce_embedding_dimension(embedding_vectors, self.idx)

        return embedding_vectors


class AnomalyMapGenerator:
    """Generate Anomaly Heatmap"""

    def __init__(self, image_size: Union[ListConfig, Tuple], sigma: int = 4):
        self.image_size = image_size if isinstance(image_size, tuple) else tuple(image_size)
        self.sigma = sigma

    @staticmethod
    def compute_distance(embedding: Tensor, stats: List[Tensor]) -> Tensor:
        """
        Compute anomaly score to the patch in position(i,j) of a test image
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

        batch, channel, height, width = embedding.shape
        embedding = embedding.reshape(batch, channel, height * width)

        # calculate mahalanobis distances
        mean, covariance = stats
        delta = (embedding - mean).permute(2, 0, 1)
        inverse_covariance = torch.linalg.inv(covariance.permute(2, 0, 1))

        distances = (torch.matmul(delta, inverse_covariance) * delta).sum(2).T
        distances = distances.reshape(batch, height, width)
        distances = torch.sqrt(distances)

        return distances

    def up_sample(self, distance: Tensor) -> Tensor:
        """
        Up sample anomaly score to match the input image size.

        Args:
            distance: Anomaly score computed via the mahalanobis distance.
            distance: Tensor:

        Returns:
            Resized distance matrix matching the input image size

        """

        score_map = F.interpolate(
            distance.unsqueeze(1),
            size=self.image_size,
            mode="bilinear",
            align_corners=False,
        )
        return score_map

    def smooth_anomaly_map(self, anomaly_map: Tensor) -> Tensor:
        """
        Apply gaussian smoothing to the anomaly map

        Args:
            anomaly_map: Anomaly score for the test image(s)
            anomaly_map: Tensor:

        Returns:
            Filtered anomaly scores

        """
        kernel_size = 2 * int(4.0 * self.sigma + 0.5) + 1
        anomaly_map = gaussian_blur2d(anomaly_map, (kernel_size, kernel_size), sigma=(self.sigma, self.sigma))

        return anomaly_map

    def compute_anomaly_map(self, embedding: Tensor, mean: Tensor, covariance: Tensor) -> Tensor:
        """
        Compute anomaly score based on embedding vector, mean and covariance of the multivariate
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

        score_map = self.compute_distance(
            embedding=embedding,
            stats=[mean.to(embedding.device), covariance.to(embedding.device)],
        )
        up_sampled_score_map = self.up_sample(score_map)
        smoothed_anomaly_map = self.smooth_anomaly_map(up_sampled_score_map)

        return smoothed_anomaly_map

    def __call__(self, **kwds):
        """
        Returns anomaly_map.
        Expects `embedding`, `mean` and `covariance` keywords to be passed explicitly

        Example:
        >>> anomaly_map_generator = AnomalyMapGenerator(image_size=input_size)
        >>> output = anomaly_map_generator(embedding=embedding, mean=mean, covariance=covariance)

        Raises:
            ValueError: `embedding`. `mean` or `covariance` keys are not found

        Returns:
            torch.Tensor: anomaly map
        """
        if not ("embedding" in kwds and "mean" in kwds and "covariance" in kwds):
            raise ValueError(f"Expected keys `embedding`, `mean` and `covariance`. Found {kwds.keys()}")

        embedding: Tensor = kwds["embedding"]
        mean: Tensor = kwds["mean"]
        covariance: Tensor = kwds["covariance"]

        return self.compute_anomaly_map(embedding, mean, covariance)


class PadimLightning(SegmentationModule):
    """
    PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization
    """

    def __init__(self, hparams):
        super().__init__(hparams)
        self.layers = hparams.model.layers
        self.model = PadimModel(hparams).eval()

        self.callbacks = get_callbacks(hparams)
        self.stats: List[Tensor, Tensor] = []
        self.automatic_optimization = False

    @staticmethod
    def configure_optimizers():
        """PADIM doesn't require optimization, therefore returns no optimizers."""
        return None

    def training_step(self, batch, _):  # pylint: disable=arguments-differ
        """Training Step of PADIM.
        For each batch, hierarchical features are extracted from the CNN.

        Args:
            batch: Input batch
            _: Index of the batch.

        Returns:
                Hierarchical feature map

        """
        self.model.feature_extractor.eval()
        embeddings = self.model(batch["image"])
        return {"embeddings": embeddings.cpu()}

    def training_epoch_end(self, outputs):
        """Fit a multivariate gaussian model on an embedding extracted from deep hierarchical CNN features.

        Args:
            outputs: Batch of outputs from the training step

        Returns:

        """
        embeddings = torch.vstack([x["embeddings"] for x in outputs])
        self.stats = self.model.gaussian.fit(embeddings)

    def validation_step(self, batch, _):  # pylint: disable=arguments-differ
        """
        Validation Step of PADIM.
        Similar to the training step, hierarchical features
            are extracted from the CNN for each batch.

        Args:
            batch: Input batch
            _: Index of the batch.

        Returns:
            Dictionary containing images, features, true labels and masks.
            These are required in `validation_epoch_end` for feature concatenation.

        """
        batch["anomaly_maps"] = self.model(batch["image"])
        return batch
