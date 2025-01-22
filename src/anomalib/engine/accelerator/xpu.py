"""XPU Accelerator."""

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import torch
from lightning.pytorch.accelerators import Accelerator, AcceleratorRegistry


class XPUAccelerator(Accelerator):
    """Support for a XPU, optimized for large-scale machine learning."""

    accelerator_name = "xpu"

    @staticmethod
    def setup_device(device: torch.device) -> None:
        """Sets up the specified device."""
        if device.type != "xpu":
            msg = f"Device should be xpu, got {device} instead"
            raise RuntimeError(msg)

        torch.xpu.set_device(device)

    @staticmethod
    def parse_devices(devices: str | list | torch.device) -> list:
        """Parses devices for multi-GPU training."""
        if isinstance(devices, list):
            return devices
        return [devices]

    @staticmethod
    def get_parallel_devices(devices: list) -> list[torch.device]:
        """Generates a list of parrallel devices."""
        return [torch.device("xpu", idx) for idx in devices]

    @staticmethod
    def auto_device_count() -> int:
        """Returns number of XPU devices available."""
        return torch.xpu.device_count()

    @staticmethod
    def is_available() -> bool:
        """Checks if XPU available."""
        return hasattr(torch, "xpu") and torch.xpu.is_available()

    @staticmethod
    def get_device_stats(device: str | torch.device) -> dict[str, Any]:
        """Returns XPU devices stats."""
        del device  # Unused
        return {}

    def teardown(self) -> None:
        """Teardown the XPU accelerator.

        This method is empty as it needs to be overridden otherwise the base class will throw an error.
        """


AcceleratorRegistry.register(
    XPUAccelerator.accelerator_name,
    XPUAccelerator,
    description="Accelerator supports XPU devices",
)
