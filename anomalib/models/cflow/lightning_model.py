"""CFLOW: Real-Time  Unsupervised Anomaly Detection via Conditional Normalizing Flows.

https://arxiv.org/pdf/2107.12571v1.pdf
"""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import List, Tuple, Union

import einops
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, ListConfig
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.utilities.cli import MODEL_REGISTRY
from torch import optim

from anomalib.models.cflow.torch_model import CflowModel
from anomalib.models.cflow.utils import get_logp, positional_encoding_2d
from anomalib.models.components import AnomalyModule

__all__ = ["Cflow", "CflowLightning"]


@MODEL_REGISTRY
class Cflow(AnomalyModule):
    """PL Lightning Module for the CFLOW algorithm."""

    def __init__(
        self,
        input_size: Tuple[int, int],
        backbone: str,
        layers: List[str],
        pre_trained: bool = True,
        fiber_batch_size: int = 64,
        decoder: str = "freia-cflow",
        condition_vector: int = 128,
        coupling_blocks: int = 8,
        clamp_alpha: float = 1.9,
        permute_soft: bool = False,
        lr: float = 0.0001,
    ):
        super().__init__()

        self.model: CflowModel = CflowModel(
            input_size=input_size,
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
        # TODO: LR should be part of optimizer in config.yaml! Since cflow has custom
        #   optimizer this is to be addressed later.
        self.learning_rate = lr

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configures optimizers for each decoder.

        Note:
            This method is used for the existing CLI.
            When PL CLI is introduced, configure optimizers method will be
                deprecated, and optimizers will be configured from either
                config.yaml file or from CLI.

        Returns:
            Optimizer: Adam optimizer for each decoder
        """
        decoders_parameters = []
        for decoder_idx in range(len(self.model.pool_layers)):
            decoders_parameters.extend(list(self.model.decoders[decoder_idx].parameters()))

        optimizer = optim.Adam(
            params=decoders_parameters,
            lr=self.learning_rate,
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


class CflowLightning(Cflow):
    """PL Lightning Module for the CFLOW algorithm.

    Args:
        hparams (Union[DictConfig, ListConfig]): Model params
    """

    def __init__(self, hparams: Union[DictConfig, ListConfig]) -> None:
        super().__init__(
            input_size=hparams.model.input_size,
            backbone=hparams.model.backbone,
            layers=hparams.model.layers,
            pre_trained=hparams.model.pre_trained,
            fiber_batch_size=hparams.dataset.fiber_batch_size,
            decoder=hparams.model.decoder,
            condition_vector=hparams.model.condition_vector,
            coupling_blocks=hparams.model.coupling_blocks,
            clamp_alpha=hparams.model.clamp_alpha,
            permute_soft=hparams.model.soft_permutation,
        )
        self.hparams: Union[DictConfig, ListConfig]  # type: ignore
        self.save_hyperparameters(hparams)

    def configure_callbacks(self):
        """Configure model-specific callbacks.

        Note:
            This method is used for the existing CLI.
            When PL CLI is introduced, configure callback method will be
                deprecated, and callbacks will be configured from either
                config.yaml file or from CLI.
        """
        early_stopping = EarlyStopping(
            monitor=self.hparams.model.early_stopping.metric,
            patience=self.hparams.model.early_stopping.patience,
            mode=self.hparams.model.early_stopping.mode,
        )
        return [early_stopping]
