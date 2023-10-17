"""Get callbacks related to sweep."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from lightning.pytorch import Callback
from omegaconf import DictConfig, ListConfig

from anomalib.utils.callbacks.timer import TimerCallback


def get_sweep_callbacks(config: DictConfig | ListConfig) -> list[Callback]:
    """Get callbacks relevant to sweep.

    Args:
    ----
        config (DictConfig | ListConfig): Model config loaded from anomalib

    Returns:
    -------
        list[Callback]: List of callbacks
    """
    del config  # Unused argument.

    callbacks: list[Callback] = [TimerCallback()]

    return callbacks
