# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Serialize pre-processing transforms as safe, plain-data specifications.

Transforms are converted to and from a ``{"class_path": ..., "init_args": ...}``
mapping built entirely from strings, numbers, booleans, lists, and dicts. This
lets checkpoints persist preprocessing configuration without pickling live
``nn.Module`` transform objects, so they can be restored under
``torch.load(..., weights_only=True)``.

Only transforms registered in :mod:`anomalib.pre_processing.utils._transform_registry`
can be produced by :func:`spec_to_transform`; see that module for the allowlist.
"""

from __future__ import annotations

import inspect
import typing
from enum import Enum
from typing import TYPE_CHECKING, TypeAlias, cast, get_type_hints

from torchvision.transforms.v2 import Transform
from typing_extensions import TypedDict

from ._transform_registry import TRANSFORM_REGISTRY

if TYPE_CHECKING:
    from collections.abc import Mapping

SpecValue: TypeAlias = str | int | float | bool | list["SpecValue"] | dict[str, "SpecValue"] | None
ConstructorValue: TypeAlias = SpecValue | Transform | Enum | list["ConstructorValue"] | dict[str, "ConstructorValue"]


class TransformSpec(TypedDict):
    """Plain-data representation of a transform."""

    class_path: str
    init_args: dict[str, SpecValue]


def transform_to_spec(transform: Transform | None) -> TransformSpec | None:
    """Convert a registered transform to a plain-data spec.

    Args:
        transform: Transform to serialize, or ``None``.

    Returns:
        Nested transform specification, or ``None``.

    Raises:
        ValueError: If the transform is not registered, or its constructor
            arguments cannot be represented as plain data.
    """
    if transform is None:
        return None

    class_path = TRANSFORM_REGISTRY.path_for(type(transform))
    init_args: dict[str, SpecValue] = {}
    for name in _constructor_params(type(transform)):
        if not hasattr(transform, name):
            msg = f"Cannot serialize {class_path}: missing constructor attribute {name!r}"
            raise ValueError(msg)
        init_args[name] = _encode_value(getattr(transform, name), class_path)

    return {"class_path": class_path, "init_args": init_args}


def spec_to_transform(spec: TransformSpec | None) -> Transform | None:
    """Construct a transform from a plain-data specification.

    Args:
        spec: Nested transform specification, or ``None``.

    Returns:
        Reconstructed transform, or ``None``.

    Raises:
        TypeError: If ``spec`` is not a mapping with ``class_path`` /
            ``init_args``, or those fields have the wrong types.
        ValueError: If ``class_path`` is unregistered or constructor arguments
            cannot be decoded for the target transform.
    """
    if spec is None:
        return None
    if not isinstance(spec, dict) or set(spec) != {"class_path", "init_args"}:
        msg = "Transform spec must contain only class_path and init_args"
        raise TypeError(msg)

    class_path = spec["class_path"]
    if not isinstance(class_path, str):
        msg = f"Unsupported transform class path: {class_path!r}"
        raise TypeError(msg)
    init_args = spec["init_args"]
    if not isinstance(init_args, dict):
        msg = f"Transform init_args must be a dictionary: {class_path}"
        raise TypeError(msg)

    transform_cls = TRANSFORM_REGISTRY.class_for(class_path)
    parameters = _constructor_params(transform_cls)
    hints = get_type_hints(transform_cls.__init__)
    args: dict[str, ConstructorValue] = {}
    for name, value in init_args.items():
        if name not in parameters:
            msg = f"Unknown constructor argument {name!r} for {class_path}"
            raise ValueError(msg)
        args[name] = _decode_value(value, _resolve_enum_annotation(hints.get(name)))
    try:
        return transform_cls(**args)
    except (TypeError, ValueError) as exc:
        msg = f"Invalid transform spec for {class_path}: {exc}"
        raise ValueError(msg) from exc


def _constructor_params(transform_cls: type[Transform]) -> Mapping[str, inspect.Parameter]:
    """Get the constructor parameters that participate in serialization.

    Excludes ``self`` and ``*args``/``**kwargs``, which cannot be represented
    as named ``init_args``.
    """
    parameters = inspect.signature(transform_cls.__init__).parameters
    return {
        name: parameter
        for name, parameter in parameters.items()
        if name != "self" and parameter.kind not in {parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD}
    }


def _resolve_enum_annotation(annotation: object) -> type[Enum] | None:
    """Find an ``Enum`` subtype in a constructor parameter's type annotation.

    Torchvision annotates some parameters as a union that includes an enum,
    for example ``Resize.interpolation: InterpolationMode | int | str``.
    ``annotation`` itself is then a union object, not the enum class, so a
    plain ``issubclass`` check on the whole annotation misses it and a
    serialized enum value round-trips back as its raw string instead of the
    original enum member.

    Args:
        annotation: A constructor parameter's type annotation, or ``None``.

    Returns:
        type[Enum] | None: The first ``Enum`` subtype found in the
            annotation (including inside a union), or ``None`` if none is
            present.
    """
    if annotation is None:
        return None
    if inspect.isclass(annotation) and issubclass(annotation, Enum):
        return annotation
    for arg in typing.get_args(annotation):
        resolved = _resolve_enum_annotation(arg)
        if resolved is not None:
            return resolved
    return None


# NOTE: Deliberately a closed `isinstance` chain rather than `functools.singledispatch`.
# `singledispatch` is an open extension point that any module could register
# against; this codec is a security boundary that must only ever accept the
# fixed set of plain-data shapes below.
def _encode_value(value: object, class_path: str) -> SpecValue:
    """Encode a constructor attribute value as plain data."""
    if isinstance(value, Transform):
        return transform_to_spec(value)  # type: ignore[return-value]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_encode_value(item, class_path) for item in value]
    msg = f"Unsupported argument value: {type(value).__name__}"
    raise ValueError(msg)


def _decode_value(value: SpecValue, annotation: type[Enum] | None = None) -> ConstructorValue:
    """Decode plain data back into a constructor argument value."""
    if isinstance(value, dict) and "class_path" in value:
        return spec_to_transform(cast("TransformSpec", value))
    if isinstance(value, dict):
        return {key: _decode_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode_value(item) for item in value]
    if annotation is not None and inspect.isclass(annotation) and issubclass(annotation, Enum):
        return annotation(value)
    return value
