"""DRÆM – A discriminatively trained reconstruction embedding for surface anomaly detection.

Paper https://arxiv.org/abs/2108.07610
"""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Callable, Dict, Optional

from pytorch_lightning.utilities.cli import MODEL_REGISTRY
from torch import Tensor, nn

from anomalib.models.components import AnomalyModule
from anomalib.models.draem.loss import DraemLoss
from anomalib.models.draem.torch_model import DraemModel
from anomalib.models.draem.utils import Augmenter

__all__ = ["Draem"]


@MODEL_REGISTRY
class Draem(AnomalyModule):
    """DRÆM: A discriminatively trained reconstruction embedding for surface anomaly detection.

    Args:
        anomaly_source_path (Optional[str]): Path to folder that contains the anomaly source images. Random noise will
            be used if left empty.
    """

    def __init__(
        self, enable_sspcab: bool = False, sspcab_lambda: float = 0.1, anomaly_source_path: Optional[str] = None
    ):
        super().__init__()

        self.augmenter = Augmenter(anomaly_source_path)
        self.model = DraemModel(sspcab=enable_sspcab)
        self.loss = DraemLoss()
        self.sspcab = enable_sspcab

        if self.sspcab:
            self.sspcab_activations: Dict = {}
            self.setup_sspcab()
            self.sspcab_loss = nn.MSELoss()
            self.sspcab_lambda = sspcab_lambda

    def setup_sspcab(self):
        """Prepare the model for the SSPCAB training step by adding forward hooks for the SSPCAB layer activations."""

        def get_activation(name: str) -> Callable:
            """Retrieves the activations.

            Args:
                name (str): Identifier for the retrieved activations.
            """

            def hook(_, __, output: Tensor):
                """Hook for retrieving the activations."""
                self.sspcab_activations[name] = output

            return hook

        self.model.reconstructive_subnetwork.encoder.mp4.register_forward_hook(get_activation("input"))
        self.model.reconstructive_subnetwork.encoder.block5.register_forward_hook(get_activation("output"))

    def training_step(self, batch, _):  # pylint: disable=arguments-differ
        """Training Step of DRAEM.

        Feeds the original image and the simulated anomaly
        image through the network and computes the training loss.

        Args:
            batch (Dict[str, Any]): Batch containing image filename, image, label and mask

        Returns:
            Loss dictionary
        """
        input_image = batch["image"]
        # Apply corruption to input image
        augmented_image, anomaly_mask = self.augmenter.augment_batch(input_image)
        # Generate model prediction
        reconstruction, prediction = self.model(augmented_image)
        # Compute loss
        loss = self.loss(input_image, reconstruction, anomaly_mask, prediction)

        if self.sspcab:
            loss += self.sspcab_lambda * self.sspcab_loss(
                self.sspcab_activations["input"], self.sspcab_activations["output"]
            )

        self.log("train_loss", loss.item(), on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss}

    def validation_step(self, batch, _):
        """Validation step of DRAEM. The Softmax predictions of the anomalous class are used as anomaly map.

        Args:
            batch: Batch of input images

        Returns:
            Dictionary to which predicted anomaly maps have been added.
        """
        prediction = self.model(batch["image"])
        batch["anomaly_maps"] = prediction
        return batch
