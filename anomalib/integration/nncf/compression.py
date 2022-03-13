"""NNCF functions."""

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

from typing import Any, Dict, Iterator, Tuple

import torch.nn as nn
from nncf import NNCFConfig
from nncf.torch import create_compressed_model, load_state, register_default_init_args
from nncf.torch.compression_method_api import PTCompressionAlgorithmController
from nncf.torch.initialization import PTInitializingDataLoader
from nncf.torch.nncf_network import NNCFNetwork
from ote_anomalib.logging import get_logger
from torch.utils.data.dataloader import DataLoader

logger = get_logger(__name__)


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


def wrap_nncf_model(
    model: nn.Module, config: Dict, dataloader: DataLoader = None, init_state_dict: Dict = None
) -> Tuple[NNCFNetwork, PTCompressionAlgorithmController]:
    """
    Wrap model by NNCF.

    :param model: Anomalib model.
    :param config: NNCF config.
    :param dataloader: Dataloader for initialization of NNCF model.
    :param init_state_dict: Opti
    :return: compression controller, compressed model
    """
    nncf_config = NNCFConfig.from_dict(config)

    if not dataloader and not init_state_dict:
        logger.warning(
            "Either dataloader or NNCF pre-trained "
            "model checkpoint should be set. Without this, "
            "quantizers will not be initialized"
        )

    compression_state = None
    resuming_state_dict = None
    if init_state_dict:
        resuming_state_dict = init_state_dict.get("model")
        compression_state = init_state_dict.get("compression_state")

    if dataloader:
        init_loader = InitLoader(dataloader)  # type: ignore
        nncf_config = register_default_init_args(nncf_config, init_loader)

    nncf_ctrl, nncf_model = create_compressed_model(
        model=model, config=nncf_config, dump_graphs=False, compression_state=compression_state
    )

    if resuming_state_dict:
        load_state(nncf_model, resuming_state_dict, is_resume=True)

    return nncf_ctrl, nncf_model


def is_state_nncf(state: Dict) -> bool:
    """
    The function to check if sate was the result of training of NNCF-compressed model.
    """
    return bool(state.get("meta", {}).get("nncf_enable_compression", False))
