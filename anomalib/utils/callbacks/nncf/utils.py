"""Utils for NNCf optimization."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from copy import copy
from typing import Any, Dict, Iterator, List, Tuple

from nncf import NNCFConfig
from nncf.api.compression import CompressionAlgorithmController
from nncf.torch import create_compressed_model, load_state, register_default_init_args
from nncf.torch.initialization import PTInitializingDataLoader
from nncf.torch.nncf_network import NNCFNetwork
from torch import nn
from torch.utils.data.dataloader import DataLoader

logger = logging.getLogger(name="NNCF compression")


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
) -> Tuple[CompressionAlgorithmController, NNCFNetwork]:
    """Wrap model by NNCF.

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
    """The function to check if sate is the result of NNCF-compressed model."""
    return bool(state.get("meta", {}).get("nncf_enable_compression", False))


def compose_nncf_config(nncf_config: Dict, enabled_options: List[str]) -> Dict:
    """Compose NNCf config by selected options.

    :param nncf_config:
    :param enabled_options:
    :return: config
    """
    optimisation_parts = nncf_config
    optimisation_parts_to_choose = []
    if "order_of_parts" in optimisation_parts:
        # The result of applying the changes from optimisation parts
        # may depend on the order of applying the changes
        # (e.g. if for nncf_quantization it is sufficient to have `total_epochs=2`,
        #  but for sparsity it is required `total_epochs=50`)
        # So, user can define `order_of_parts` in the optimisation_config
        # to specify the order of applying the parts.
        order_of_parts = optimisation_parts["order_of_parts"]
        assert isinstance(order_of_parts, list), 'The field "order_of_parts" in optimisation config should be a list'

        for part in enabled_options:
            assert part in order_of_parts, (
                f"The part {part} is selected, " "but it is absent in order_of_parts={order_of_parts}"
            )

        optimisation_parts_to_choose = [part for part in order_of_parts if part in enabled_options]

    assert "base" in optimisation_parts, 'Error: the optimisation config does not contain the "base" part'
    nncf_config_part = optimisation_parts["base"]

    for part in optimisation_parts_to_choose:
        assert part in optimisation_parts, f'Error: the optimisation config does not contain the part "{part}"'
        optimisation_part_dict = optimisation_parts[part]
        try:
            nncf_config_part = merge_dicts_and_lists_b_into_a(nncf_config_part, optimisation_part_dict)
        except AssertionError as cur_error:
            err_descr = (
                f"Error during merging the parts of nncf configs:\n"
                f"the current part={part}, "
                f"the order of merging parts into base is {optimisation_parts_to_choose}.\n"
                f"The error is:\n{cur_error}"
            )
            raise RuntimeError(err_descr) from None

    return nncf_config_part


# pylint: disable=invalid-name
def merge_dicts_and_lists_b_into_a(a, b):
    """The function to merge dict configs."""
    return _merge_dicts_and_lists_b_into_a(a, b, "")


def _merge_dicts_and_lists_b_into_a(a, b, cur_key=None):
    """The function is inspired by mmcf.Config._merge_a_into_b.

    * works with usual dicts and lists and derived types
    * supports merging of lists (by concatenating the lists)
    * makes recursive merging for dict + dict case
    * overwrites when merging scalar into scalar
    Note that we merge b into a (whereas Config makes merge a into b),
    since otherwise the order of list merging is counter-intuitive.
    """

    def _err_str(_a, _b, _key):
        if _key is None:
            _key_str = "of whole structures"
        else:
            _key_str = f"during merging for key=`{_key}`"
        return (
            f"Error in merging parts of config: different types {_key_str},"
            f" type(a) = {type(_a)},"
            f" type(b) = {type(_b)}"
        )

    assert isinstance(a, (dict, list)), f"Can merge only dicts and lists, whereas type(a)={type(a)}"
    assert isinstance(b, (dict, list)), _err_str(a, b, cur_key)
    assert isinstance(a, list) == isinstance(b, list), _err_str(a, b, cur_key)
    if isinstance(a, list):
        # the main diff w.r.t. mmcf.Config -- merging of lists
        return a + b

    a = copy(a)
    for k in b.keys():
        if k not in a:
            a[k] = copy(b[k])
            continue
        new_cur_key = cur_key + "." + k if cur_key else k
        if isinstance(a[k], (dict, list)):
            a[k] = _merge_dicts_and_lists_b_into_a(a[k], b[k], new_cur_key)
            continue

        assert not isinstance(b[k], (dict, list)), _err_str(a[k], b[k], new_cur_key)

        # suppose here that a[k] and b[k] are scalars, just overwrite
        a[k] = b[k]
    return a
