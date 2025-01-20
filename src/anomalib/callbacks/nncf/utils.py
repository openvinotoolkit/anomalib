"""Utilities for Neural Network Compression Framework (NNCF) optimization.

This module provides utility functions and classes for working with Intel's Neural Network
Compression Framework (NNCF). It includes functionality for model initialization, state
management, and configuration handling.

The module contains:

- ``InitLoader``: A data loader class for NNCF initialization
- Functions for wrapping PyTorch models with NNCF compression
- Utilities for handling NNCF model states and configurations
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from copy import copy
from typing import TYPE_CHECKING, Any

import torch
from nncf import NNCFConfig
from nncf.api.compression import CompressionAlgorithmController
from nncf.torch import create_compressed_model, load_state, register_default_init_args
from nncf.torch.initialization import PTInitializingDataLoader
from nncf.torch.nncf_network import NNCFNetwork
from torch import nn
from torch.utils.data.dataloader import DataLoader

if TYPE_CHECKING:
    from collections.abc import Iterator


logger = logging.getLogger(name="NNCF compression")


class InitLoader(PTInitializingDataLoader):
    """Initializing data loader for NNCF to be used with unsupervised training algorithms.

    This class extends NNCF's ``PTInitializingDataLoader`` to handle unsupervised training data.
    It provides methods for iterating through the data and extracting inputs for model initialization.

    Args:
        data_loader (DataLoader): PyTorch ``DataLoader`` containing the initialization data.

    Examples:
        Create an initialization loader from a PyTorch dataloader:

        >>> from torch.utils.data import DataLoader, TensorDataset
        >>> import torch
        >>> dataset = TensorDataset(torch.randn(10, 3, 32, 32))
        >>> dataloader = DataLoader(dataset)
        >>> init_loader = InitLoader(dataloader)

        Iterate through the loader:

        >>> for batch in init_loader:
        ...     assert isinstance(batch, torch.Tensor)
        ...     assert batch.shape[1:] == (3, 32, 32)

    Note:
        The loader expects the dataloader to return dictionaries with an ``"image"`` key
        containing the input tensor.
    """

    def __init__(self, data_loader: DataLoader) -> None:
        super().__init__(data_loader)
        self._data_loader_iter: Iterator

    def __iter__(self) -> "InitLoader":
        """Create iterator for dataloader.

        Returns:
            InitLoader: Self reference for iteration.

        Example:
            >>> from torch.utils.data import DataLoader, TensorDataset
            >>> loader = InitLoader(DataLoader(TensorDataset(torch.randn(1,3,32,32))))
            >>> iterator = iter(loader)
            >>> isinstance(iterator, InitLoader)
            True
        """
        self._data_loader_iter = iter(self._data_loader)
        return self

    def __next__(self) -> torch.Tensor:
        """Return next item from dataloader iterator.

        Returns:
            torch.Tensor: Next image tensor from the dataloader.

        Example:
            >>> from torch.utils.data import DataLoader, TensorDataset
            >>> loader = InitLoader(DataLoader(TensorDataset(torch.randn(1,3,32,32))))
            >>> batch = next(iter(loader))
            >>> isinstance(batch, torch.Tensor)
            True
        """
        loaded_item = next(self._data_loader_iter)
        return loaded_item["image"]

    @staticmethod
    def get_inputs(dataloader_output: dict[str, str | torch.Tensor]) -> tuple[tuple, dict]:
        """Get input to model.

        Args:
            dataloader_output (dict[str, str | torch.Tensor]): Output from the dataloader
                containing the input tensor.

        Returns:
            tuple[tuple, dict]: A tuple containing:
                - A tuple with the dataloader output
                - An empty dict for additional arguments

        Example:
            >>> output = {"image": torch.randn(1,3,32,32)}
            >>> args, kwargs = InitLoader.get_inputs(output)
            >>> isinstance(args, tuple) and isinstance(kwargs, dict)
            True
        """
        return (dataloader_output,), {}

    @staticmethod
    def get_target(_) -> None:  # noqa: ANN001
        """Return structure for ground truth in loss criterion based on dataloader output.

        This implementation is a placeholder that returns ``None`` since ground truth
        is not used in unsupervised training.

        Returns:
            None: Always returns ``None`` as targets are not used.

        Example:
            >>> InitLoader.get_target(None) is None
            True
        """
        return


def wrap_nncf_model(
    model: nn.Module,
    config: dict,
    dataloader: DataLoader,
    init_state_dict: dict,
) -> tuple[CompressionAlgorithmController, NNCFNetwork]:
    """Wrap PyTorch model with NNCF compression.

    Args:
        model (nn.Module): Anomalib model to be compressed.
        config (dict): NNCF configuration dictionary.
        dataloader (DataLoader): DataLoader for NNCF model initialization.
        init_state_dict (dict): Initial state dictionary for model initialization.

    Returns:
        tuple[CompressionAlgorithmController, NNCFNetwork]: A tuple containing:
            - The compression controller
            - The compressed model

    Warning:
        Either ``dataloader`` or ``init_state_dict`` must be provided for proper quantizer initialization.

    Example:
        >>> import torch.nn as nn
        >>> from torch.utils.data import DataLoader, TensorDataset
        >>> model = nn.Linear(10, 2)
        >>> config = {"input_info": {"sample_size": [1, 10]}}
        >>> data = torch.randn(100, 10)
        >>> dataloader = DataLoader(TensorDataset(data))
        >>> controller, compressed = wrap_nncf_model(model, config, dataloader, {})
        >>> isinstance(compressed, NNCFNetwork)
        True
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
    """Check if state is the result of NNCF-compressed model.

    Args:
        state (dict): Model state dictionary to check.

    Returns:
        bool: ``True`` if the state is from an NNCF-compressed model, ``False`` otherwise.

    Example:
        >>> state = {"meta": {"nncf_enable_compression": True}}
        >>> is_state_nncf(state)
        True
        >>> state = {"meta": {}}
        >>> is_state_nncf(state)
        False
    """
    return bool(state.get("meta", {}).get("nncf_enable_compression", False))


def compose_nncf_config(nncf_config: dict, enabled_options: list[str]) -> dict:
    """Compose NNCF config by selected options.

    This function merges different parts of the NNCF configuration based on enabled options.
    It supports ordered application of configuration parts through the ``order_of_parts`` field.

    Args:
        nncf_config (dict): Base NNCF configuration dictionary.
        enabled_options (list[str]): List of enabled optimization options.

    Returns:
        dict: Composed NNCF configuration.

    Raises:
        TypeError: If ``order_of_parts`` is not a list.
        ValueError: If an enabled option is not in ``order_of_parts``.
        KeyError: If ``base`` part or any enabled option is missing from config.
        RuntimeError: If there's an error during config merging.

    Example:
        >>> config = {
        ...     "base": {"epochs": 1},
        ...     "quantization": {"epochs": 2},
        ...     "order_of_parts": ["quantization"]
        ... }
        >>> result = compose_nncf_config(config, ["quantization"])
        >>> result["epochs"]
        2
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
        if not isinstance(order_of_parts, list):
            msg = 'The field "order_of_parts" in optimization config should be a list'
            raise TypeError(msg)

        for part in enabled_options:
            if part not in order_of_parts:
                msg = f"The part {part} is selected, but it is absent in order_of_parts={order_of_parts}"
                raise ValueError(msg)

        optimisation_parts_to_choose = [part for part in order_of_parts if part in enabled_options]

    if "base" not in optimisation_parts:
        msg = 'Error: the optimisation config does not contain the "base" part'
        raise KeyError(msg)
    nncf_config_part = optimisation_parts["base"]

    for part in optimisation_parts_to_choose:
        if part not in optimisation_parts:
            msg = f'Error: the optimisation config does not contain the part "{part}"'
            raise KeyError(msg)
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
    """Merge two configuration dictionaries or lists.

    This function provides the public interface for merging configurations.
    It delegates to the internal ``_merge_dicts_and_lists_b_into_a`` function.

    Args:
        a (dict[Any, Any] | list[Any]): First dictionary or list to merge.
        b (dict[Any, Any] | list[Any]): Second dictionary or list to merge into first.

    Returns:
        dict[Any, Any] | list[Any]: Merged configuration.

    Example:
        >>> a = {"x": 1, "y": [1, 2]}
        >>> b = {"y": [3], "z": 2}
        >>> result = merge_dicts_and_lists_b_into_a(a, b)
        >>> result["y"]
        [1, 2, 3]
        >>> result["z"]
        2
    """
    return _merge_dicts_and_lists_b_into_a(a, b, "")


def _merge_dicts_and_lists_b_into_a(
    a: dict[Any, Any] | list[Any],
    b: dict[Any, Any] | list[Any],
    cur_key: int | str | None = None,
) -> dict[Any, Any] | list[Any]:
    """Recursively merge two configuration dictionaries or lists.

    This function implements the following merge behavior:
    - Works with standard dicts, lists and their derived types
    - Merges lists by concatenation
    - Performs recursive merging for nested dictionaries
    - Overwrites scalar values when merging

    Args:
        a (dict[Any, Any] | list[Any]): First dictionary or list to merge.
        b (dict[Any, Any] | list[Any]): Second dictionary or list to merge into first.
        cur_key (int | str | None, optional): Current key in recursive merge. Defaults to None.

    Returns:
        dict[Any, Any] | list[Any]: Merged configuration.

    Raises:
        TypeError: If inputs are not dictionaries or lists, or if types are incompatible.

    Example:
        >>> a = {"x": {"y": [1]}}
        >>> b = {"x": {"y": [2]}}
        >>> result = _merge_dicts_and_lists_b_into_a(a, b)
        >>> result["x"]["y"]
        [1, 2]
    """

    def _err_str(_a: dict | list, _b: dict | list, _key: int | str | None = None) -> str:
        _key_str = "of whole structures" if _key is None else f"during merging for key=`{_key}`"
        return (
            f"Error in merging parts of config: different types {_key_str}, type(a) = {type(_a)}, type(b) = {type(_b)}"
        )

    if not (isinstance(a, dict | list)):
        msg = f"Can merge only dicts and lists, whereas type(a)={type(a)}"
        raise TypeError(msg)

    if not (isinstance(b, dict | list)):
        raise TypeError(_err_str(a, b, cur_key))

    if (isinstance(a, list) and not isinstance(b, list)) or (isinstance(b, list) and not isinstance(a, list)):
        raise TypeError(_err_str(a, b, cur_key))

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

        if any(isinstance(b[k], t) for t in [dict, list]):
            raise TypeError(_err_str(a[k], b[k], new_cur_key))

        # suppose here that a[k] and b[k] are scalars, just overwrite
        a[k] = b[k]
    return a
