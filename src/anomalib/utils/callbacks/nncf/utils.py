"""Utils for NNCf optimization."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import logging
from copy import copy
from typing import TYPE_CHECKING, Any

from nncf import NNCFConfig
from nncf.api.compression import CompressionAlgorithmController
from nncf.torch import create_compressed_model, load_state, register_default_init_args
from nncf.torch.initialization import PTInitializingDataLoader
from nncf.torch.nncf_network import NNCFNetwork
from torch import Tensor, nn
from torch.utils.data.dataloader import DataLoader

if TYPE_CHECKING:
    from collections.abc import Iterator


logger = logging.getLogger(name="NNCF compression")


class InitLoader(PTInitializingDataLoader):
    """Initializing data loader for NNCF to be used with unsupervised training algorithms."""

    def __init__(self, data_loader: DataLoader) -> None:
        super().__init__(data_loader)
        self._data_loader_iter: Iterator

    def __iter__(self) -> "InitLoader":
        """Create iterator for dataloader."""
        self._data_loader_iter = iter(self._data_loader)
        return self

    def __next__(self) -> Tensor:
        """Return next item from dataloader iterator."""
        loaded_item = next(self._data_loader_iter)
        return loaded_item["image"]

    def get_inputs(self, dataloader_output: dict[str, str | Tensor]) -> tuple[tuple, dict]:
        """Get input to model.

        Returns:
            (dataloader_output,), {}: tuple[tuple, dict]: The current model call to be made during
            the initialization process
        """
        return (dataloader_output,), {}

    def get_target(self, _):  # noqa: ANN001, ANN201
        """Return structure for ground truth in loss criterion based on dataloader output.

        This implementation does not do anything and is a placeholder.

        Returns:
            None
        """
        return


def wrap_nncf_model(
    model: nn.Module,
    config: dict,
    dataloader: DataLoader,
    init_state_dict: dict,
) -> tuple[CompressionAlgorithmController, NNCFNetwork]:
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
            "quantizers will not be initialized",
        )

    compression_state = None
    resuming_state_dict = None
    if init_state_dict:
        resuming_state_dict = init_state_dict.get("model")
        compression_state = init_state_dict.get("compression_state")

    if dataloader:
        init_loader = InitLoader(dataloader)
        nncf_config = register_default_init_args(nncf_config, init_loader)

    nncf_ctrl, nncf_model = create_compressed_model(
        model=model,
        config=nncf_config,
        dump_graphs=False,
        compression_state=compression_state,
    )

    if resuming_state_dict:
        load_state(nncf_model, resuming_state_dict, is_resume=True)

    return nncf_ctrl, nncf_model


def is_state_nncf(state: dict) -> bool:
    """The function to check if sate is the result of NNCF-compressed model."""
    return bool(state.get("meta", {}).get("nncf_enable_compression", False))


def compose_nncf_config(nncf_config: dict, enabled_options: list[str]) -> dict:
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
            assert (
                part in order_of_parts
            ), f"The part {part} is selected, but it is absent in order_of_parts={order_of_parts}"

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


def merge_dicts_and_lists_b_into_a(
    a: dict[Any, Any] | list[Any],
    b: dict[Any, Any] | list[Any],
) -> dict[Any, Any] | list[Any]:
    """The function to merge dict configs."""
    return _merge_dicts_and_lists_b_into_a(a, b, "")


def _merge_dicts_and_lists_b_into_a(
    a: dict[Any, Any] | list[Any],
    b: dict[Any, Any] | list[Any],
    cur_key: int | str | None = None,
) -> dict[Any, Any] | list[Any]:
    """The function is inspired by mmcf.Config._merge_a_into_b.

    * works with usual dicts and lists and derived types
    * supports merging of lists (by concatenating the lists)
    * makes recursive merging for dict + dict case
    * overwrites when merging scalar into scalar
    Note that we merge b into a (whereas Config makes merge a into b),
    since otherwise the order of list merging is counter-intuitive.
    """

    def _err_str(_a: dict | list, _b: dict | list, _key: int | str | None = None) -> str:
        _key_str = "of whole structures" if _key is None else f"during merging for key=`{_key}`"
        return (
            f"Error in merging parts of config: different types {_key_str},"
            f" type(a) = {type(_a)},"
            f" type(b) = {type(_b)}"
        )

    assert isinstance(a, dict | list), f"Can merge only dicts and lists, whereas type(a)={type(a)}"
    assert isinstance(b, dict | list), _err_str(a, b, cur_key)
    assert isinstance(a, list) == isinstance(b, list), _err_str(a, b, cur_key)
    if isinstance(a, list) and isinstance(b, list):
        # the main diff w.r.t. mmcf.Config -- merging of lists
        return a + b

    a = copy(a)
    for k in b:
        if k not in a:
            a[k] = copy(b[k])
            continue
        new_cur_key = str(cur_key) + "." + k if cur_key else k
        if isinstance(a[k], dict | list):
            a[k] = _merge_dicts_and_lists_b_into_a(a[k], b[k], new_cur_key)
            continue

        assert not isinstance(b[k], dict | list), _err_str(a[k], b[k], new_cur_key)

        # suppose here that a[k] and b[k] are scalars, just overwrite
        a[k] = b[k]
    return a
