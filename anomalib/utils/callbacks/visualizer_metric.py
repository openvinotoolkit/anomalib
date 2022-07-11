"""Metric Visualizer Callback."""

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

import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import CALLBACK_REGISTRY

from anomalib.models.components import AnomalyModule

from .visualizer_base import VisualizerCallbackBase


@CALLBACK_REGISTRY
class VisualizerCallbackMetric(VisualizerCallbackBase):
    """Callback that visualizes the metric results of a model by plotting the corresponding curves.

    To save the images to the filesystem, add the 'local' keyword to the `project.log_images_to` parameter in the
    config.yaml file.
    """

    def on_test_end(self, trainer: pl.Trainer, pl_module: AnomalyModule) -> None:
        """Log images of the metric scoires for all appropriate.

        Args:
            trainer (pl.Trainer): pytorch lightning trainer.
            pl_module (AnomalyModule): pytorch lightning module.
        """
        super().on_batch_end(trainer, pl_module)
