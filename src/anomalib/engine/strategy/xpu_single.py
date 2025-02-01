"""Lightning strategy for single XPU device."""

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import lightning.pytorch as pl
import torch
from lightning.pytorch.strategies import SingleDeviceStrategy, StrategyRegistry
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from lightning_fabric.plugins import CheckpointIO
from lightning_fabric.plugins.precision import Precision
from lightning_fabric.utilities.types import _DEVICE


class SingleXPUStrategy(SingleDeviceStrategy):
    """Strategy for training on single XPU device."""

    strategy_name = "xpu_single"

    def __init__(
        self,
        device: _DEVICE = "xpu:0",
        accelerator: pl.accelerators.Accelerator | None = None,
        checkpoint_io: CheckpointIO | None = None,
        precision_plugin: Precision | None = None,
    ) -> None:
        if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
            msg = "`SingleXPUStrategy` requires XPU devices to run"
            raise MisconfigurationException(msg)

        super().__init__(
            accelerator=accelerator,
            device=device,
            checkpoint_io=checkpoint_io,
            precision_plugin=precision_plugin,
        )


StrategyRegistry.register(
    SingleXPUStrategy.strategy_name,
    SingleXPUStrategy,
    description="Strategy that enables training on single XPU",
)
