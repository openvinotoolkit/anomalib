"""NNCF Callback."""

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

import os
from typing import Any, Dict, Iterator, Optional, Tuple, Union

import pytorch_lightning as pl
import yaml
from nncf import NNCFConfig
from nncf.api.compression import CompressionAlgorithmController, CompressionScheduler
from nncf.torch import create_compressed_model, register_default_init_args
from nncf.torch.initialization import PTInitializingDataLoader
from omegaconf import DictConfig, ListConfig, OmegaConf
from pytorch_lightning import Callback
from torch.utils.data.dataloader import DataLoader


def criterion_fn(outputs, criterion):
    """Calls the criterion function on outputs."""
    return criterion(outputs)


class InitLoader(PTInitializingDataLoader):
    """Initializing data loader for NNCF to be used with unsupervised training algorithms."""

    def __init__(self, data_loader: DataLoader):
        super().__init__(data_loader)
        self._data_loader_iter: Iterator

    def __iter__(self):
        """Create iterator for dataloader."""
        self._data_loader_iter = iter(self._data_loader)
        return self

    def __next__(self) -> Any:
        """Return next item from dataloader iterator."""
        loaded_item = next(self._data_loader_iter)
        return loaded_item["image"]

    def get_inputs(self, dataloader_output) -> Tuple[Tuple, Dict]:
        """Get input to model.

        Returns:
            (dataloader_output,), {}: Tuple[Tuple, Dict]: The current model call to be made during
            the initialization process
        """
        return (dataloader_output,), {}

    def get_target(self, _):
        """Return structure for ground truth in loss criterion based on dataloader output.

        This implementation does not do anything and is a placeholder.

        Returns:
            None
        """
        return None


class NNCFCallback(Callback):
    """Callback for NNCF compression.

    Assumes that the pl module contains a 'model' attribute, which is
    the PyTorch module that must be compressed.

    Args:
        config (Union[ListConfig, DictConfig]): NNCF Configuration
        dirpath (str): Path where the export `onnx` and the OpenVINO `xml` and `bin` IR are saved.
        filename (str): Name of the generated model files.
    """

    def __init__(self, config: Union[ListConfig, DictConfig], dirpath: str, filename: str):
        config_dict = yaml.safe_load(OmegaConf.to_yaml(config.optimization.nncf))
        self.nncf_config = NNCFConfig.from_dict(config_dict)
        self.dirpath = dirpath
        self.filename = filename

        self.comp_ctrl: Optional[CompressionAlgorithmController] = None
        self.compression_scheduler: CompressionScheduler

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: Optional[str] = None) -> None:
        # pylint: disable=unused-argument
        """Call when fit or test begins.

        Takes the pytorch model and wraps it using the compression controller so that it is ready for nncf fine-tuning.
        """
        if self.comp_ctrl is None:
            # NOTE: trainer.datamodule returns the following error
            #   "Trainer" has no attribute "datamodule"  [attr-defined]
            init_loader = InitLoader(trainer.datamodule.train_dataloader())  # type: ignore
            nncf_config = register_default_init_args(
                self.nncf_config, init_loader, pl_module.model.loss, criterion_fn=criterion_fn
            )
            # if dump_graphs is not set to False, nncf will generate intermediate .dot files in the current dir
            self.comp_ctrl, pl_module.model = create_compressed_model(pl_module.model, nncf_config, dump_graphs=False)
            self.compression_scheduler = self.comp_ctrl.scheduler

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
        self.compression_scheduler.step()
        if self.comp_ctrl is not None:
            trainer.model.loss_val = self.comp_ctrl.loss()

    def on_train_end(self, _trainer: pl.Trainer, _pl_module: pl.LightningModule) -> None:
        """Call when the train ends.

        Exports onnx model and if compression controller is not None, uses the onnx model to generate the OpenVINO IR.
        """
        os.makedirs(self.dirpath, exist_ok=True)
        onnx_path = os.path.join(self.dirpath, self.filename + ".onnx")
        if self.comp_ctrl is not None:
            self.comp_ctrl.export_model(onnx_path)
        optimize_command = "mo --input_model " + onnx_path + " --output_dir " + self.dirpath
        os.system(optimize_command)

    def on_train_epoch_start(self, _trainer: pl.Trainer, _pl_module: pl.LightningModule) -> None:
        """Call when the train epoch starts.

        Prepare compression method to continue training the model in the next epoch.
        """
        self.compression_scheduler.epoch_step()
