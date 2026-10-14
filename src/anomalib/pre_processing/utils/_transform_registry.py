# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Private allowlist of transforms that are safe to serialize as plain data.

This module is intentionally not re-exported from any package ``__init__.py``.
It defines the closed set of transform classes that :mod:`anomalib.pre_processing.
utils.spec` is allowed to construct from untrusted checkpoint data. Adding a
transform here is a security-relevant change and should be reviewed as such.
"""

from __future__ import annotations

from torchvision.transforms.v2 import CenterCrop, Compose, Grayscale, Normalize, Resize, Transform

from anomalib.data.transforms import ExportableCenterCrop, SquarePad


class TransformRegistry:
    """Bidirectional allowlist mapping transform classes to stable class paths.

    Each class has one *canonical* path, used when serializing a transform to a
    spec, and one or more *accepted* paths, used when resolving a spec back to a
    class. Registration always accepts the class's concrete module path in
    addition to any explicit canonical path or aliases, so a canonical path can
    be changed later (e.g. when a class moves module) without breaking specs
    that were already written to a checkpoint.
    """

    def __init__(self) -> None:
        self._canonical_path: dict[type[Transform], str] = {}
        self._class_by_path: dict[str, type[Transform]] = {}

    def register(
        self,
        transform_cls: type[Transform],
        canonical_path: str | None = None,
        *,
        aliases: tuple[str, ...] = (),
    ) -> None:
        """Register a transform class as safe to serialize.

        Args:
            transform_cls: Transform class to allowlist.
            canonical_path: Class path written to new specs. Defaults to the
                class's concrete module path (``f"{cls.__module__}.{cls.__qualname__}"``).
            aliases: Additional class paths accepted when reading a spec, for
                example a previous canonical path.
        """
        concrete_path = f"{transform_cls.__module__}.{transform_cls.__qualname__}"
        canonical_path = canonical_path or concrete_path

        self._canonical_path[transform_cls] = canonical_path
        for accepted_path in (canonical_path, concrete_path, *aliases):
            self._class_by_path[accepted_path] = transform_cls

    def path_for(self, transform_cls: type[Transform]) -> str:
        """Get the canonical class path for a registered transform class.

        Args:
            transform_cls: Transform class to look up.

        Returns:
            str: Canonical class path used when writing a spec.

        Raises:
            ValueError: If the class is not registered.
        """
        path = self._canonical_path.get(transform_cls)
        if path is None:
            msg = f"Unsupported transform class: {transform_cls.__module__}.{transform_cls.__qualname__}"
            raise ValueError(msg)
        return path

    def class_for(self, class_path: str) -> type[Transform]:
        """Resolve a class path from a spec to a registered transform class.

        Args:
            class_path: Class path to resolve.

        Returns:
            type[Transform]: Registered transform class.

        Raises:
            ValueError: If the class path is not registered.
        """
        transform_cls = self._class_by_path.get(class_path)
        if transform_cls is None:
            msg = f"Unsupported transform class: {class_path}"
            raise ValueError(msg)
        return transform_cls


TRANSFORM_REGISTRY = TransformRegistry()

# torchvision's concrete modules (``_geometry``, ``_misc``, ...) are private and
# may be reorganised between releases, so the public ``v2`` alias is canonical.
for _cls in (CenterCrop, Compose, Grayscale, Normalize, Resize):
    TRANSFORM_REGISTRY.register(_cls, f"torchvision.transforms.v2.{_cls.__name__}")

# First-party transforms canonicalise to their package export. The concrete
# module path (e.g. ``anomalib.data.transforms.square_pad.SquarePad``) remains
# an accepted alias so specs written before this canonicalisation still load.
for _cls in (ExportableCenterCrop, SquarePad):
    TRANSFORM_REGISTRY.register(_cls, f"anomalib.data.transforms.{_cls.__name__}")

del _cls
