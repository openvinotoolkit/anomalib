# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Utility helpers for model export."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from anomalib.deploy.export import ExportType

DEFAULT_EXPORT_SPATIAL_SIZE = (32, 32)


def get_onnx_dynamo_flag(kwargs: dict[str, Any]) -> bool:
    """Return ONNX exporter dynamo flag.

    The dynamo-based exporter is required as of anomalib 2.7.0. Passing
    ``dynamo=False`` raises because the legacy exporter path was removed.

    Args:
        kwargs (dict[str, Any]): Keyword arguments passed to ``torch.onnx.export``.

    Returns:
        bool: Always ``True`` after validating the requested flag.

    Raises:
        TypeError: If ``dynamo`` is not a ``bool`` or ``None``.
        ValueError: If ``dynamo=False`` is requested.
    """
    dynamo = kwargs.pop("dynamo", True)
    if dynamo is None:
        return True
    if not isinstance(dynamo, bool):
        msg = f"`dynamo` must be a bool or None, got {type(dynamo).__name__}: {dynamo!r}"
        raise TypeError(msg)
    if not dynamo:
        msg = (
            "The legacy ONNX exporter path (`dynamo=False`) was removed in anomalib 2.7.0. "
            "Install `anomalib[openvino]` (provides `onnxscript`) and use `dynamo=True` (the default)."
        )
        raise ValueError(msg)
    return True


def get_default_dynamic_axes(
    input_size: tuple[int, int] | None,
    input_names: list[str],
    output_names: list[str],
) -> dict[str, dict[int, str]]:
    """Build default dynamic axes for ONNX export.

    Args:
        input_size (tuple[int, int] | None): Input image dimensions ``(H, W)``.
            When ``None``, height and width axes are marked dynamic as well.
        input_names (list[str]): Resolved ONNX input names.
        output_names (list[str]): Resolved ONNX output names.

    Returns:
        dict[str, dict[int, str]]: Mapping of tensor name to axis-index/axis-name.
    """
    input_name = input_names[0] if input_names else "input"
    input_axes = {0: "batch_size"} if input_size else {0: "batch_size", 2: "height", 3: "width"}
    axes: dict[str, dict[int, str]] = {input_name: input_axes}
    for name in output_names:
        axes[name] = {0: "batch_size"}
    return axes


def get_dynamic_shapes_from_axes(
    dynamic_axes: dict[str, dict[int, str]] | None,
    input_names: list[str],
    output_names: list[str],
) -> tuple[dict[int, Any],] | None:
    """Translate single-input ``dynamic_axes`` to dynamo ``dynamic_shapes``.

    Dynamo expects ``torch.export.Dim`` objects; axes sharing a name share a ``Dim``.
    """
    if not dynamic_axes:
        return None

    input_name = input_names[0] if input_names else "input"
    input_axes = dynamic_axes.get(input_name)
    if input_axes is None:
        input_axes = next((axes for name, axes in dynamic_axes.items() if name not in output_names), None)
    if not input_axes:
        return None

    dimensions = {name: torch.export.Dim(name) for name in input_axes.values()}
    return ({axis: dimensions[name] for axis, name in input_axes.items()},)


def get_example_input(
    input_size: tuple[int, int] | None,
    dynamic_shapes: object,
) -> torch.Tensor:
    """Build example image input matching static dimensions in ``dynamic_shapes``.

    Dynamo specializes example dimensions of size 0 or 1. Use size 2 for symbolic
    dimensions while honoring explicit static sizes, ``None``, and ``Dim.STATIC``.
    Supports positional specs (tuple/list), named argument mappings, and a direct
    axis-to-dimension mapping for this single-image input.

    Args:
        input_size (tuple[int, int] | None): Fixed ``(H, W)``, or ``None``.
        dynamic_shapes (object): Dynamo shape specification passed through to ``torch.onnx.export``.
            Only the specification for the single image input is used to select example sizes.

    Returns:
        torch.Tensor: Zero tensor of shape ``(B, 3, H, W)``.
    """
    height, width = input_size or DEFAULT_EXPORT_SPATIAL_SIZE
    shape = [1, 3, height, width]

    specification = dynamic_shapes
    while True:
        if isinstance(specification, Mapping):
            if not specification or all(isinstance(axis, int) for axis in specification):
                axes = {axis: dimension for axis, dimension in specification.items() if isinstance(axis, int)}
                break
            # This API exports one positional image tensor; named mappings wrap its spec.
            if len(specification) == 1:
                specification = next(iter(specification.values()))
                continue
            # Exporter reports unsupported multi-input specifications for this single-input API.
            axes = {}
            break
        if isinstance(specification, Sequence) and not isinstance(specification, (str, bytes)):
            if len(specification) == 1 and (
                specification[0] is None or isinstance(specification[0], (Mapping, Sequence))
            ):
                specification = specification[0]
                continue
            # A sequence at the tensor-spec level describes dimensions by position.
            axes = dict(enumerate(specification))
            break
        axes = {}
        break

    for axis, dimension in axes.items():
        if dimension is None or dimension is torch.export.Dim.STATIC:
            continue
        if isinstance(dimension, int) and not isinstance(dimension, bool):
            shape[axis] = dimension
        elif dimension is torch.export.Dim.AUTO or dimension is torch.export.Dim.DYNAMIC:
            shape[axis] = max(shape[axis], 2)
        else:
            try:
                minimum, maximum = dimension.min, dimension.max
            except AttributeError:
                minimum, maximum = 0, None
            size = max(2, minimum) if isinstance(minimum, int) else 2
            shape[axis] = min(size, maximum) if isinstance(maximum, int) else size
    return torch.zeros(shape)


def validate_input_names(input_names: object) -> list[str]:
    """Validate ONNX input names.

    Accepts any ``Sequence[str]`` (e.g. list or tuple) and returns a ``list[str]``
    for downstream use.

    Args:
        input_names (object): Candidate input names value.

    Returns:
        list[str]: Validated input names.

    Raises:
        TypeError: If input names are not a sequence of strings.
    """
    if (
        isinstance(input_names, Sequence)
        and not isinstance(input_names, (str, bytes))
        and all(isinstance(name, str) for name in input_names)
    ):
        return [str(name) for name in input_names]
    msg = f"input_names must be a sequence of strings, got {type(input_names).__name__}: {input_names!r}"
    raise TypeError(msg)


def raise_missing_onnxscript_error(cause: BaseException | None = None) -> None:
    """Raise actionable error for missing ``onnxscript`` dependency.

    Args:
        cause (BaseException | None): Original exception to chain via ``raise ... from``.

    Raises:
        ModuleNotFoundError: If ``onnxscript`` is not installed for dynamo export.
    """
    msg = "ONNX export requires the optional `onnxscript` dependency. Install `anomalib[openvino]` or `onnxscript`."
    raise ModuleNotFoundError(msg, name="onnxscript") from cause


def create_export_root(export_root: str | Path, export_type: ExportType) -> Path:
    """Create directory structure for model export.

    Args:
        export_root (str | Path): Root directory for exports.
        export_type (ExportType): Type of export (torch/onnx/openvino).

    Returns:
        Path: Created directory path.
    """
    export_root = Path(export_root) / "weights" / export_type.value
    export_root.mkdir(parents=True, exist_ok=True)
    return export_root
