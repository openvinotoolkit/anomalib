"""Cflow.

Real-Time Unsupervised Anomaly Detection via Conditional Normalizing Flows.

For more details, see the paper: `Real-Time Unsupervised Anomaly Detection via
Conditional Normalizing Flows <https://arxiv.org/pdf/2107.12571v1.pdf>`_.
"""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

__all__ = ["Cflow"]

from collections.abc import Sequence
from typing import Any

import einops
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import optim
from torch.nn import functional as F  # noqa: N812
from torch.optim import Optimizer

from anomalib import LearningType
from anomalib.models.components import AnomalyModule

from .torch_model import CflowModel
from .utils import get_logp, positional_encoding_2d


class Cflow(AnomalyModule):
    """PL Lightning Module for the CFLOW algorithm.

    Args:
        backbone (str, optional): Backbone CNN architecture.
            Defaults to ``"wide_resnet50_2"``.
        layers (Sequence[str], optional): Layers to extract features from.
            Defaults to ``("layer2", "layer3", "layer4")``.
        pre_trained (bool, optional): Whether to use pre-trained weights.
            Defaults to ``True``.
        fiber_batch_size (int, optional): Fiber batch size.
            Defaults to ``64``.
        decoder (str, optional): Decoder architecture.
            Defaults to ``"freia-cflow"``.
        condition_vector (int, optional): Condition vector size.
            Defaults to ``128``.
        coupling_blocks (int, optional): Number of coupling blocks.
            Defaults to ``8``.
        clamp_alpha (float, optional): Clamping value for the alpha parameter.
            Defaults to ``1.9``.
        permute_soft (bool, optional): Whether to use soft permutation.
            Defaults to ``False``.
        lr (float, optional): Learning rate.
            Defaults to ``0.0001``.
    """

    def __init__(
        self,
        backbone: str = "wide_resnet50_2",
        layers: Sequence[str] = ("layer2", "layer3", "layer4"),
        pre_trained: bool = True,
        fiber_batch_size: int = 64,
        decoder: str = "freia-cflow",
        condition_vector: int = 128,
        coupling_blocks: int = 8,
        clamp_alpha: float = 1.9,
        permute_soft: bool = False,
        lr: float = 0.0001,
    ) -> None:
        super().__init__()

        self.model: CflowModel = CflowModel(
            backbone=backbone,
            pre_trained=pre_trained,
            layers=layers,
            fiber_batch_size=fiber_batch_size,
            decoder=decoder,
            condition_vector=condition_vector,
            coupling_blocks=coupling_blocks,
            clamp_alpha=clamp_alpha,
            permute_soft=permute_soft,
        )
        self.automatic_optimization = False
        # TODO(ashwinvaidya17): LR should be part of optimizer in config.yaml since  cflow has custom optimizer.
        # CVS-122670
        self.learning_rate = lr

    def configure_optimizers(self) -> Optimizer:
        """Configure optimizers for each decoder.

        Returns:
            Optimizer: Adam optimizer for each decoder
        """
        decoders_parameters = []
        for decoder_idx in range(len(self.model.pool_layers)):
            decoders_parameters.extend(list(self.model.decoders[decoder_idx].parameters()))

        return optim.Adam(
            params=decoders_parameters,
            lr=self.learning_rate,
        )

    def training_step(self, batch: dict[str, str | torch.Tensor], *args, **kwargs) -> STEP_OUTPUT:
        """Perform the training step of CFLOW.

        For each batch, decoder layers are trained with a dynamic fiber batch size.
        Training step is performed manually as multiple training steps are involved
            per batch of input images

        Args:
            batch (dict[str, str | torch.Tensor]): Input batch
            *args: Arguments.
            **kwargs: Keyword arguments.

        Returns:
          Loss value for the batch

        """
        del args, kwargs  # These variables are not used.

        opt = self.optimizers()

        images: torch.Tensor = batch["image"]
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
            if fiber_batches <= 0:
                msg = "Make sure we have enough fibers, otherwise decrease N or batch-size!"
                raise ValueError(msg)

            for batch_num in range(fiber_batches):  # per-fiber processing
                opt.zero_grad()
                if batch_num < (fiber_batches - 1):
                    idx = torch.arange(
                        batch_num * self.model.fiber_batch_size,
                        (batch_num + 1) * self.model.fiber_batch_size,
                    )
                else:  # When non-full batch is encountered batch_num * N will go out of bounds
                    idx = torch.arange(batch_num * self.model.fiber_batch_size, embedding_length)
                # get random vectors
                c_p = c_r[perm[idx]]  # NxP
                e_p = e_r[perm[idx]]  # NxC
                # decoder returns the transformed variable z and the log Jacobian determinant
                p_u, log_jac_det = decoder(e_p, [c_p])
                decoder_log_prob = get_logp(dim_feature_vector, p_u, log_jac_det)
                log_prob = decoder_log_prob / dim_feature_vector  # likelihood per dim
                loss = -F.logsigmoid(log_prob)
                self.manual_backward(loss.mean())
                opt.step()
                avg_loss += loss.sum()

        self.log("train_loss", avg_loss.item(), on_epoch=True, prog_bar=True, logger=True)
        return {"loss": avg_loss}

    def validation_step(self, batch: dict[str, str | torch.Tensor], *args, **kwargs) -> STEP_OUTPUT:
        """Perform the validation step of CFLOW.

            Similar to the training step, encoder features
            are extracted from the CNN for each batch, and anomaly
            map is computed.

        Args:
            batch (dict[str, str | torch.Tensor]): Input batch
            *args: Arguments.
            **kwargs: Keyword arguments.

        Returns:
            Dictionary containing images, anomaly maps, true labels and masks.
            These are required in `validation_epoch_end` for feature concatenation.

        """
        del args, kwargs  # These variables are not used.

        batch["anomaly_maps"] = self.model(batch["image"])
        return batch

    @property
    def trainer_arguments(self) -> dict[str, Any]:
        """C-FLOW specific trainer arguments."""
        return {"gradient_clip_val": 0, "num_sanity_val_steps": 0}

    @property
    def learning_type(self) -> LearningType:
        """Return the learning type of the model.

        Returns:
            LearningType: Learning type of the model.
        """
        return LearningType.ONE_CLASS
