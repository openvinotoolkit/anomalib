# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Helpers for testing that code can be captured by ``torch.export``."""

from collections.abc import Callable

import torch


class _FunctionModule(torch.nn.Module):
    """Expose a plain function as ``forward`` so ``torch.export`` can capture it."""

    def __init__(self, function: Callable[..., object]) -> None:
        super().__init__()
        self.function = function

    def forward(self, *inputs: torch.Tensor) -> object:
        """Call the wrapped function."""
        return self.function(*inputs)


def assert_exportable(function: Callable[..., object], *inputs: torch.Tensor) -> None:
    """Assert ``function`` captures with ``torch.export`` and the captured graph matches eager.

    Fails on data-dependent control flow (e.g. branching on tensor values), which the
    dynamo ONNX exporter cannot capture.

    Args:
        function (Callable[..., object]): Function of tensors returning a tensor or tuple of tensors.
        *inputs (torch.Tensor): Example inputs.
    """
    module = _FunctionModule(function)
    exported = torch.export.export(module, inputs)
    torch.testing.assert_close(exported.module()(*inputs), module(*inputs))
