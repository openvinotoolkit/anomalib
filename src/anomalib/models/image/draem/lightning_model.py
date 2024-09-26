"""DRÆM - A discriminatively trained reconstruction embedding for surface anomaly detection.

Paper https://arxiv.org/abs/2108.07610
"""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from typing import Any

import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import nn
from torchvision.transforms.v2 import Compose, Resize, Transform

from anomalib import LearningType
from anomalib.data.utils import Augmenter
from anomalib.models.components import AnomalyModule

from .loss import DraemLoss
from .torch_model import DraemModel

__all__ = ["Draem"]


class Draem(AnomalyModule):
    """DRÆM: A discriminatively trained reconstruction embedding for surface anomaly detection.

    Args:
        enable_sspcab (bool): Enable SSPCAB training.
            Defaults to ``False``.
        sspcab_lambda (float): SSPCAB loss weight.
            Defaults to ``0.1``.
        anomaly_source_path (str | None): Path to folder that contains the anomaly source images. Random noise will
            be used if left empty.
            Defaults to ``None``.
    """

    def __init__(
        self,
        enable_sspcab: bool = False,
        sspcab_lambda: float = 0.1,
        anomaly_source_path: str | None = None,
        beta: float | tuple[float, float] = (0.1, 1.0),
    ) -> None:
        super().__init__()

        self.augmenter = Augmenter(anomaly_source_path, beta=beta)
        self.model = DraemModel(sspcab=enable_sspcab)
        self.loss = DraemLoss()
        self.sspcab = enable_sspcab

        if self.sspcab:
            self.sspcab_activations: dict = {}
            self.setup_sspcab()
            self.sspcab_loss = nn.MSELoss()
            self.sspcab_lambda = sspcab_lambda

    def setup_sspcab(self) -> None:
        """Prepare the model for the SSPCAB training step by adding forward hooks for the SSPCAB layer activations."""

        def get_activation(name: str) -> Callable:
            """Retrieve the activations.

            Args:
                name (str): Identifier for the retrieved activations.
            """

            def hook(_, __, output: torch.Tensor) -> None:  # noqa: ANN001
                """Create hook for retrieving the activations.

                Args:
                    _: Placeholder for the module input.
                    __: Placeholder for the module output.
                    output (torch.Tensor): The output tensor of the module.
                """
                self.sspcab_activations[name] = output

            return hook

        self.model.reconstructive_subnetwork.encoder.mp4.register_forward_hook(get_activation("input"))
        self.model.reconstructive_subnetwork.encoder.block5.register_forward_hook(get_activation("output"))

    def training_step(self, batch: dict[str, str | torch.Tensor], *args, **kwargs) -> STEP_OUTPUT:
        """Perform the training step of DRAEM.

        Feeds the original image and the simulated anomaly
        image through the network and computes the training loss.

        Args:
            batch (dict[str, str | torch.Tensor]): Batch containing image filename, image, label and mask
            args: Arguments.
            kwargs: Keyword arguments.

        Returns:
            Loss dictionary
        """
        del args, kwargs  # These variables are not used.

        input_image = batch["image"]
        # Apply corruption to input image
        augmented_image, anomaly_mask = self.augmenter.augment_batch(input_image)
        # Generate model prediction
        reconstruction, prediction = self.model(augmented_image)
        # Compute loss
        loss = self.loss(input_image, reconstruction, anomaly_mask, prediction)

        if self.sspcab:
            loss += self.sspcab_lambda * self.sspcab_loss(
                self.sspcab_activations["input"],
                self.sspcab_activations["output"],
            )

        self.log("train_loss", loss.item(), on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss}

    def validation_step(self, batch: dict[str, str | torch.Tensor], *args, **kwargs) -> STEP_OUTPUT:
        """Perform the validation step of DRAEM. The Softmax predictions of the anomalous class are used as anomaly map.

        Args:
            batch (dict[str, str | torch.Tensor]): Batch of input images
            args: Arguments.
            kwargs: Keyword arguments.

        Returns:
            Dictionary to which predicted anomaly maps have been added.
        """
        del args, kwargs  # These variables are not used.

        prediction = self.model(batch["image"])
        batch["anomaly_maps"] = prediction
        return batch

    @property
    def trainer_arguments(self) -> dict[str, Any]:
        """Return DRÆM-specific trainer arguments."""
        return {"gradient_clip_val": 0, "num_sanity_val_steps": 0}

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure the Adam optimizer."""
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=0.0001)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[400, 600], gamma=0.1)
        return [optimizer], [scheduler]

    @property
    def learning_type(self) -> LearningType:
        """Return the learning type of the model.

        Returns:
            LearningType: Learning type of the model.
        """
        return LearningType.ONE_CLASS

    @staticmethod
    def configure_transforms(image_size: tuple[int, int] | None = None) -> Transform:
        """Default transform for DRAEM. Normalization is not needed as the images are scaled to [0, 1] in Dataset."""
        image_size = image_size or (256, 256)
        return Compose(
            [
                Resize(image_size, antialias=True),
            ],
        )
