"""Callbacks for NNCF optimization."""

# Copyright (C) 2022 Intel Corporation
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

from typing import Any, Dict, Optional

import pytorch_lightning as pl
from nncf import NNCFConfig
from nncf.torch import register_default_init_args
from pytorch_lightning import Callback

from anomalib.integration.nncf.compression import wrap_nncf_model
from anomalib.integration.nncf.utils import InitLoader


class NNCFCallback(Callback):
    """Callback for NNCF compression.

    Assumes that the pl module contains a 'model' attribute, which is
    the PyTorch module that must be compressed.

    Args:
        config (Dict): NNCF Configuration
    """

    def __init__(self, nncf_config: Dict):
        self.nncf_config = NNCFConfig(nncf_config)
        self.nncf_ctrl = None

    # pylint: disable=unused-argument
    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: Optional[str] = None) -> None:
        """Call when fit or test begins.

        Takes the pytorch model and wraps it using the compression controller
        so that it is ready for nncf fine-tuning.
        """
        if self.nncf_ctrl:
            return
        # pylint: disable=attr-defined
        init_loader = InitLoader(trainer.datamodule.train_dataloader())  # type: ignore
        nncf_config = register_default_init_args(self.nncf_config, init_loader)

        self.nncf_ctrl, pl_module.model = wrap_nncf_model(
            model=pl_module.model, config=nncf_config, dataloader=trainer.datamodule.train_dataloader()
        )

    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        _pl_module: pl.LightningModule,
        _batch: Any,
        _batch_idx: int,
        _unused: Optional[int] = 0,
    ) -> None:
        """Call when the train batch begins.

        Prepare compression method to continue training the model in the next step.
        """
        if self.nncf_ctrl:
            self.nncf_ctrl.scheduler.step()

    def on_train_epoch_start(self, _trainer: pl.Trainer, _pl_module: pl.LightningModule) -> None:
        """Call when the train epoch starts.

        Prepare compression method to continue training the model in the next epoch.
        """
        if self.nncf_ctrl:
            self.nncf_ctrl.scheduler.epoch_step()
