"""CFLOW: Real-Time  Unsupervised Anomaly Detection via Conditional Normalizing Flows.

https://arxiv.org/pdf/2107.12571v1.pdf
"""

# Copyright (C) 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

from typing import List, Tuple, Union, cast

import einops
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from omegaconf import DictConfig, ListConfig
from pytorch_lightning.callbacks import EarlyStopping
from torch import Tensor, nn, optim

from anomalib.core.model import AnomalyModule
from anomalib.core.model.feature_extractor import FeatureExtractor
from anomalib.models.cflow.backbone import cflow_head, positional_encoding_2d

__all__ = ["AnomalyMapGenerator", "CflowModel", "CflowLightning"]


def get_logp(dim_feature_vector: int, p_u: torch.Tensor, logdet_j: torch.Tensor) -> torch.Tensor:
    """Returns the log likelihood estimation.

    Args:
        dim_feature_vector (int): Dimensions of the condition vector
        p_u (torch.Tensor): Random variable u
        logdet_j (torch.Tensor): log of determinant of jacobian returned from the invertable decoder

    Returns:
        torch.Tensor: Log probability
    """
    ln_sqrt_2pi = -np.log(np.sqrt(2 * np.pi))  # ln(sqrt(2*pi))
    logp = dim_feature_vector * ln_sqrt_2pi - 0.5 * torch.sum(p_u ** 2, 1) + logdet_j
    return logp


class AnomalyMapGenerator:
    """Generate Anomaly Heatmap."""

    def __init__(
        self,
        image_size: Union[ListConfig, Tuple],
        pool_layers: List[str],
    ):
        self.distance = torch.nn.PairwiseDistance(p=2, keepdim=True)
        self.image_size = image_size if isinstance(image_size, tuple) else tuple(image_size)
        self.pool_layers: List[str] = pool_layers

    def compute_anomaly_map(
        self, distribution: Union[List[Tensor], List[List]], height: List[int], width: List[int]
    ) -> Tensor:
        """Compute the layer map based on likelihood estimation.

        Args:
          distribution: Probability distribution for each decoder block
          height: blocks height
          width: blocks width

        Returns:
          Final Anomaly Map

        """

        test_map: List[Tensor] = []
        for layer_idx in range(len(self.pool_layers)):
            test_norm = torch.tensor(distribution[layer_idx], dtype=torch.double)  # pylint: disable=not-callable
            test_norm -= torch.max(test_norm)  # normalize likelihoods to (-Inf:0] by subtracting a constant
            test_prob = torch.exp(test_norm)  # convert to probs in range [0:1]
            test_mask = test_prob.reshape(-1, height[layer_idx], width[layer_idx])
            # upsample
            test_map.append(
                F.interpolate(
                    test_mask.unsqueeze(1), size=self.image_size, mode="bilinear", align_corners=True
                ).squeeze()
            )
        # score aggregation
        score_map = torch.zeros_like(test_map[0])
        for layer_idx in range(len(self.pool_layers)):
            score_map += test_map[layer_idx]
        score_mask = score_map
        # invert probs to anomaly scores
        anomaly_map = score_mask.max() - score_mask

        return anomaly_map

    def __call__(self, **kwargs: Union[List[Tensor], List[int], List[List]]) -> Tensor:
        """Returns anomaly_map.

        Expects `distribution`, `height` and 'width' keywords to be passed explicitly

        Example
        >>> anomaly_map_generator = AnomalyMapGenerator(image_size=tuple(hparams.model.input_size),
        >>>        pool_layers=pool_layers)
        >>> output = self.anomaly_map_generator(distribution=dist, height=height, width=width)

        Raises:
            ValueError: `distribution`, `height` and 'width' keys are not found

        Returns:
            torch.Tensor: anomaly map
        """
        if not ("distribution" in kwargs and "height" in kwargs and "width" in kwargs):
            raise KeyError(f"Expected keys `distribution`, `height` and `width`. Found {kwargs.keys()}")

        # placate mypy
        distribution: List[Tensor] = cast(List[Tensor], kwargs["distribution"])
        height: List[int] = cast(List[int], kwargs["height"])
        width: List[int] = cast(List[int], kwargs["width"])
        return self.compute_anomaly_map(distribution, height, width)


class CflowModel(nn.Module):
    """CFLOW: Conditional Normalizing Flows."""

    def __init__(self, hparams: Union[DictConfig, ListConfig]):
        super().__init__()

        self.backbone = getattr(torchvision.models, hparams.model.backbone)
        self.fiber_batch_size = hparams.dataset.fiber_batch_size
        self.condition_vector: int = hparams.model.condition_vector
        self.dec_arch = hparams.model.decoder
        self.pool_layers = hparams.model.layers

        self.encoder = FeatureExtractor(backbone=self.backbone(pretrained=True), layers=self.pool_layers)
        self.pool_dims = self.encoder.out_dims
        self.decoders = nn.ModuleList(
            [
                cflow_head(self.condition_vector, hparams.model.coupling_blocks, hparams.model.clamp_alpha, pool_dim)
                for pool_dim in self.pool_dims
            ]
        )

        # encoder model is fixed
        for parameters in self.encoder.parameters():
            parameters.requires_grad = False

        self.anomaly_map_generator = AnomalyMapGenerator(
            image_size=tuple(hparams.model.input_size), pool_layers=self.pool_layers
        )

    def forward(self, images):
        """Forward-pass images into the network to extract encoder features and compute probability.

        Args:
          images: Batch of images.

        Returns:
          Predicted anomaly maps.

        """

        activation = self.encoder(images)

        distribution = [[] for _ in self.pool_layers]

        height: List[int] = []
        width: List[int] = []
        for layer_idx, layer in enumerate(self.pool_layers):
            encoder_activations = activation[layer].detach()  # BxCxHxW

            batch_size, dim_feature_vector, im_height, im_width = encoder_activations.size()
            image_size = im_height * im_width
            embedding_length = batch_size * image_size  # number of rows in the conditional vector

            height.append(im_height)
            width.append(im_width)
            # repeats positional encoding for the entire batch 1 C H W to B C H W
            pos_encoding = einops.repeat(
                positional_encoding_2d(self.condition_vector, im_height, im_width).unsqueeze(0),
                "b c h w-> (tile b) c h w",
                tile=batch_size,
            ).to(images.device)
            c_r = einops.rearrange(pos_encoding, "b c h w -> (b h w) c")  # BHWxP
            e_r = einops.rearrange(encoder_activations, "b c h w -> (b h w) c")  # BHWxC
            decoder = self.decoders[layer_idx].to(images.device)

            # Sometimes during validation, the last batch E / N is not a whole number. Hence we need to add 1.
            # It is assumed that during training that E / N is a whole number as no errors were discovered during
            # testing. In case it is observed in the future, we can use only this line and ensure that FIB is at
            # least 1 or set `drop_last` in the dataloader to drop the last non-full batch.
            fiber_batches = embedding_length // self.fiber_batch_size + int(
                embedding_length % self.fiber_batch_size > 0
            )

            for batch_num in range(fiber_batches):  # per-fiber processing
                if batch_num < (fiber_batches - 1):
                    idx = torch.arange(batch_num * self.fiber_batch_size, (batch_num + 1) * self.fiber_batch_size)
                else:  # When non-full batch is encountered batch_num+1 * N will go out of bounds
                    idx = torch.arange(batch_num * self.fiber_batch_size, embedding_length)
                c_p = c_r[idx]  # NxP
                e_p = e_r[idx]  # NxC
                # decoder returns the transformed variable z and the log Jacobian determinant
                p_u, log_jac_det = decoder(e_p, [c_p])
                #
                decoder_log_prob = get_logp(dim_feature_vector, p_u, log_jac_det)
                log_prob = decoder_log_prob / dim_feature_vector  # likelihood per dim
                distribution[layer_idx] = distribution[layer_idx] + log_prob.detach().tolist()

        output = self.anomaly_map_generator(distribution=distribution, height=height, width=width)

        return output.to(images.device)


class CflowLightning(AnomalyModule):
    """PL Lightning Module for the CFLOW algorithm."""

    def __init__(self, hparams):
        super().__init__(hparams)

        self.model: CflowModel = CflowModel(hparams)
        self.loss_val = 0
        self.automatic_optimization = False

    def configure_callbacks(self):
        """Configure model-specific callbacks."""
        early_stopping = EarlyStopping(
            monitor=self.hparams.model.early_stopping.metric,
            patience=self.hparams.model.early_stopping.patience,
            mode=self.hparams.model.early_stopping.mode,
        )
        return [early_stopping]

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configures optimizers for each decoder.

        Returns:
            Optimizer: Adam optimizer for each decoder
        """
        decoders_parameters = []
        for decoder_idx in range(len(self.model.pool_layers)):
            decoders_parameters.extend(list(self.model.decoders[decoder_idx].parameters()))

        optimizer = optim.Adam(
            params=decoders_parameters,
            lr=self.hparams.model.lr,
        )
        return optimizer

    def training_step(self, batch, _):  # pylint: disable=arguments-differ
        """Training Step of CFLOW.

        For each batch, decoder layers are trained with a dynamic fiber batch size.
        Training step is performed manually as multiple training steps are involved
            per batch of input images

        Args:
          batch: Input batch
          _: Index of the batch.

        Returns:
          Loss value for the batch

        """
        opt = self.optimizers()
        self.model.encoder.eval()

        images = batch["image"]
        activation = self.model.encoder(images)
        avg_loss = torch.zeros([1], dtype=torch.float64).to(images.device)

        height = []
        width = []
        for layer_idx, layer in enumerate(self.model.pool_layers):
            encoder_activations = activation[layer].detach()  # BxCxHxW

            batch_size, dim_feature_vector, im_height, im_width = encoder_activations.size()
            image_size = im_height * im_width
            embedding_length = batch_size * image_size  # number of rows in the conditional vector

            height.append(im_height)
            width.append(im_width)
            # repeats positional encoding for the entire batch 1 C H W to B C H W
            pos_encoding = einops.repeat(
                positional_encoding_2d(self.model.condition_vector, im_height, im_width).unsqueeze(0),
                "b c h w-> (tile b) c h w",
                tile=batch_size,
            ).to(images.device)
            c_r = einops.rearrange(pos_encoding, "b c h w -> (b h w) c")  # BHWxP
            e_r = einops.rearrange(encoder_activations, "b c h w -> (b h w) c")  # BHWxC
            perm = torch.randperm(embedding_length)  # BHW
            decoder = self.model.decoders[layer_idx].to(images.device)

            fiber_batches = embedding_length // self.model.fiber_batch_size  # number of fiber batches
            assert fiber_batches > 0, "Make sure we have enough fibers, otherwise decrease N or batch-size!"

            for batch_num in range(fiber_batches):  # per-fiber processing
                opt.zero_grad()
                if batch_num < (fiber_batches - 1):
                    idx = torch.arange(
                        batch_num * self.model.fiber_batch_size, (batch_num + 1) * self.model.fiber_batch_size
                    )
                else:  # When non-full batch is encountered batch_num * N will go out of bounds
                    idx = torch.arange(batch_num * self.model.fiber_batch_size, embedding_length)
                # get random vectors
                c_p = c_r[perm[idx]]  # NxP
                e_p = e_r[perm[idx]]  # NxC
                # decoder returns the transformed variable z and the log Jacobian determinant
                p_u, log_jac_det = decoder(e_p, [c_p])
                #
                decoder_log_prob = get_logp(dim_feature_vector, p_u, log_jac_det)
                log_prob = decoder_log_prob / dim_feature_vector  # likelihood per dim
                loss = -F.logsigmoid(log_prob)
                self.manual_backward(loss.mean())
                opt.step()
                avg_loss += loss.sum()

        return {"loss": avg_loss}

    def validation_step(self, batch, _):  # pylint: disable=arguments-differ
        """Validation Step of CFLOW.

            Similar to the training step, encoder features
            are extracted from the CNN for each batch, and anomaly
            map is computed.

        Args:
          batch: Input batch
          _: Index of the batch.

        Returns:
          Dictionary containing images, anomaly maps, true labels and masks.
          These are required in `validation_epoch_end` for feature concatenation.

        """
        batch["anomaly_maps"] = self.model(batch["image"])

        return batch
