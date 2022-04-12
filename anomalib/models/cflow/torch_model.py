"""PyTorch model for CFlow model implementation."""

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

from typing import List, Union

import einops
import torch
import torchvision
from omegaconf import DictConfig, ListConfig
from torch import nn

from anomalib.models.cflow.anomaly_map import AnomalyMapGenerator
from anomalib.models.cflow.utils import cflow_head, get_logp, positional_encoding_2d
from anomalib.models.components import FeatureExtractor


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
                cflow_head(
                    condition_vector=self.condition_vector,
                    coupling_blocks=hparams.model.coupling_blocks,
                    clamp_alpha=hparams.model.clamp_alpha,
                    n_features=pool_dim,
                    permute_soft=hparams.model.soft_permutation,
                )
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

        self.encoder.eval()
        self.decoders.eval()
        with torch.no_grad():
            activation = self.encoder(images)

        distribution = [torch.Tensor(0).to(images.device) for _ in self.pool_layers]

        height: List[int] = []
        width: List[int] = []
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
                with torch.no_grad():
                    p_u, log_jac_det = decoder(e_p, [c_p])
                #
                decoder_log_prob = get_logp(dim_feature_vector, p_u, log_jac_det)
                log_prob = decoder_log_prob / dim_feature_vector  # likelihood per dim
                distribution[layer_idx] = torch.cat((distribution[layer_idx], log_prob))

        output = self.anomaly_map_generator(distribution=distribution, height=height, width=width)
        self.decoders.train()

        return output.to(images.device)
