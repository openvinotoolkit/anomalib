"""DRÆM – A discriminatively trained reconstruction embedding for surface anomaly detection.

Paper https://arxiv.org/abs/2108.07610
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

import logging
from typing import Optional, Union

import torch
from omegaconf import DictConfig, ListConfig
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.utilities.cli import MODEL_REGISTRY

from anomalib.models.components import AnomalyModule
from anomalib.models.draem.augmenter import Augmenter
from anomalib.models.draem.loss import SSIM, FocalLoss
from anomalib.models.draem.torch_model import DraemModel

logger = logging.getLogger(__name__)

__all__ = ["Draem", "DraemLightning"]


@MODEL_REGISTRY
class Draem(AnomalyModule):
    """DRÆM: A discriminatively trained reconstruction embedding for surface anomaly detection.

    Args:
        anomaly_source_path (Optional[str]): Path to folder that contains the anomaly source images. Random noise will
            be used if left empty.
    """

    def __init__(self, anomaly_source_path: Optional[str] = None):
        super().__init__()

        self.augmenter = Augmenter(anomaly_source_path)
        self.model = DraemModel()

        self.l2_loss = torch.nn.modules.loss.MSELoss()
        self.ssim_loss = SSIM()
        self.focal_loss = FocalLoss()

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
        l2_loss = self.l2_loss(reconstruction, input_image)
        ssim_loss = self.ssim_loss(reconstruction, input_image)
        focal_loss = self.focal_loss(prediction, anomaly_mask)
        loss = l2_loss + ssim_loss + focal_loss
        return {"loss": loss}

    def validation_step(self, batch, _):
        """Validation step of DRAEM. The Softmax predictions of the anomalous class are used as anomaly map.

        Args:
            batch: Batch of input images

        Returns:
            Dictionary to which predicted anomaly maps have been added.
        """
        prediction = self.model(batch["image"])
        batch["anomaly_maps"] = prediction[:, 1, :, :]
        return batch


class DraemLightning(Draem):
    """DRÆM: A discriminatively trained reconstruction embedding for surface anomaly detection.

    Args:
        hparams (Union[DictConfig, ListConfig]): Model parameters
    """

    def __init__(self, hparams: Union[DictConfig, ListConfig]):
        super().__init__(anomaly_source_path=hparams.model.anomaly_source_path)
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

    def configure_optimizers(self):  # pylint: disable=arguments-differ
        """Configure the Adam optimizer."""
        return torch.optim.Adam(params=self.model.parameters(), lr=self.hparams.model.lr)
