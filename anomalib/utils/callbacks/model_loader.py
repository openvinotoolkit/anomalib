"""Callback that loads model weights from the state dict."""

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

import torch
from pytorch_lightning import Callback

from anomalib.models.components import AnomalyModule

logger = logging.getLogger(__name__)


class LoadModelCallback(Callback):
    """Callback that loads the model weights from the state dict."""

    def __init__(self, weights_path):
        self.weights_path = weights_path

    def on_test_start(self, trainer, pl_module: AnomalyModule) -> None:  # pylint: disable=W0613
        """Call when the test begins.

        Loads the model weights from ``weights_path`` into the PyTorch module.
        """
        logger.info("Loading the model from %s", self.weights_path)
        pl_module.load_state_dict(torch.load(self.weights_path)["state_dict"])
