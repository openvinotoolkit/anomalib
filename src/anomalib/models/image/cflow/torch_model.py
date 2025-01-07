"""PyTorch model for the CFLOW anomaly detection model.

This module provides the PyTorch implementation of the CFLOW model for anomaly
detection. The model uses conditional normalizing flows to model the distribution
of normal data in the feature space.

The model consists of:
    - A CNN backbone encoder to extract features
    - Multiple decoders using normalizing flows to model feature distributions
    - Positional encoding to capture spatial information

Example:
    >>> import torch
    >>> from anomalib.models.image.cflow.torch_model import CflowModel
    >>> # Initialize the model
    >>> model = CflowModel(
    ...     backbone="resnet18",
    ...     layers=["layer1", "layer2", "layer3"],
    ...     fiber_batch_size=64,
    ...     decoder="freia-cflow",
    ...     condition_vector=128,
    ...     coupling_blocks=8,
    ...     clamp_alpha=1.9,
    ...     permute_soft=False
    ... )
    >>> # Forward pass
    >>> x = torch.randn(32, 3, 256, 256)
    >>> predictions = model(x)
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence

import einops
import torch
from torch import nn

from anomalib.data import InferenceBatch
from anomalib.models.components import TimmFeatureExtractor

from .anomaly_map import AnomalyMapGenerator
from .utils import cflow_head, get_logp, positional_encoding_2d


class CflowModel(nn.Module):
    """CFLOW: Conditional Normalizing Flows.

    Args:
        backbone (str): Name of the backbone CNN network to use as feature
            extractor.
        layers (Sequence[str]): Names of layers from which to extract features.
        pre_trained (bool, optional): Whether to use pre-trained weights for the
            backbone. Defaults to ``True``.
        fiber_batch_size (int, optional): Batch size for processing feature
            fibers. Defaults to ``64``.
        decoder (str, optional): Type of decoder architecture to use.
            Defaults to ``"freia-cflow"``.
        condition_vector (int, optional): Size of the condition vector for the
            normalizing flows. Defaults to ``128``.
        coupling_blocks (int, optional): Number of coupling blocks in the
            normalizing flows. Defaults to ``8``.
        clamp_alpha (float, optional): Clamping value for the alpha parameter in
            the flows. Defaults to ``1.9``.
        permute_soft (bool, optional): Whether to use soft permutation in the
            flows. Defaults to ``False``.

    Example:
        >>> model = CflowModel(
        ...     backbone="resnet18",
        ...     layers=["layer1", "layer2", "layer3"]
        ... )
        >>> x = torch.randn(32, 3, 256, 256)
        >>> predictions = model(x)
    """

    def __init__(
        self,
        backbone: str,
        layers: Sequence[str],
        pre_trained: bool = True,
        fiber_batch_size: int = 64,
        decoder: str = "freia-cflow",
        condition_vector: int = 128,
        coupling_blocks: int = 8,
        clamp_alpha: float = 1.9,
        permute_soft: bool = False,
    ) -> None:
        super().__init__()

        self.backbone = backbone
        self.fiber_batch_size = fiber_batch_size
        self.condition_vector: int = condition_vector
        self.dec_arch = decoder
        self.pool_layers = layers

        self.encoder = TimmFeatureExtractor(
            backbone=self.backbone,
            layers=self.pool_layers,
            pre_trained=pre_trained,
        ).eval()
        self.pool_dims = self.encoder.out_dims
        self.decoders = nn.ModuleList(
            [
                cflow_head(
                    condition_vector=self.condition_vector,
                    coupling_blocks=coupling_blocks,
                    clamp_alpha=clamp_alpha,
                    n_features=pool_dim,
                    permute_soft=permute_soft,
                )
                for pool_dim in self.pool_dims
            ],
        )

        # encoder model is fixed
        for parameters in self.encoder.parameters():
            parameters.requires_grad = False

        self.anomaly_map_generator = AnomalyMapGenerator(pool_layers=self.pool_layers)

    def forward(self, images: torch.Tensor) -> InferenceBatch:
        """Forward pass through the model.

        The method extracts features using the encoder, processes them through
        normalizing flows, and generates anomaly predictions.

        Args:
            images (torch.Tensor): Input images of shape
                ``(batch_size, channels, height, width)``.

        Returns:
            InferenceBatch: Batch containing predicted anomaly scores and maps.
                The anomaly maps have shape ``(batch_size, 1, height, width)``.

        Example:
            >>> x = torch.randn(32, 3, 256, 256)
            >>> model = CflowModel(backbone="resnet18", layers=["layer1"])
            >>> predictions = model(x)
            >>> predictions.anomaly_map.shape
            torch.Size([32, 1, 256, 256])
        """
        self.encoder.eval()
        self.decoders.eval()
        with torch.no_grad():
            activation = self.encoder(images)

        distribution = [torch.Tensor(0).to(images.device) for _ in self.pool_layers]

        height: list[int] = []
        width: list[int] = []
        for layer_idx, layer in enumerate(self.pool_layers):
            encoder_activations = activation[layer]  # BxCxHxW

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
                embedding_length % self.fiber_batch_size > 0,
            )

            for batch_num in range(fiber_batches):  # per-fiber processing
                if batch_num < (fiber_batches - 1):
                    idx = torch.arange(batch_num * self.fiber_batch_size, (batch_num + 1) * self.fiber_batch_size)
                else:  # When non-full batch is encountered batch_num+1 * N will go out of bounds
                    idx = torch.arange(batch_num * self.fiber_batch_size, embedding_length)
                c_p = c_r[idx]  # NxP
                e_p = e_r[idx]  # NxC
                # decoder returns the transformed variable z and the log Jacobian determinant
                with torch.no_grad():
                    p_u, log_jac_det = decoder(e_p, [c_p])
                decoder_log_prob = get_logp(dim_feature_vector, p_u, log_jac_det)
                log_prob = decoder_log_prob / dim_feature_vector  # likelihood per dim
                distribution[layer_idx] = torch.cat((distribution[layer_idx], log_prob))

        anomaly_map = self.anomaly_map_generator(
            distribution=distribution,
            height=height,
            width=width,
            image_size=images.shape[-2:],
        )
        self.decoders.train()

        pred_score = torch.amax(anomaly_map, dim=(-2, -1))
        return InferenceBatch(pred_score=pred_score, anomaly_map=anomaly_map)
