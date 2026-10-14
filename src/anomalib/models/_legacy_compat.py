# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Import shims for checkpoints trained before the timm-encoder migration.

Several DINOv2-based models used to build their encoder from a custom Vision Transformer that
lived in modules which have since been removed (the ``DinoV2Loader`` and its ``vision_transformer``
factories). Checkpoints trained against those versions pickled references to those module paths,
so ``torch.load`` raises ``ModuleNotFoundError`` before the weights are even read.

This module re-registers the removed import paths in :data:`sys.modules` with lightweight
placeholder modules so pickled *references* can resolve during ``torch.load``. Loading
checkpoints trained before the timm-encoder migration is no longer supported as of
anomalib 2.7.0 (legacy encoder weight restoration was removed); retrain or export under
a current release instead.

It is imported for its side effect from :mod:`anomalib.models`, so the aliases are installed before
any checkpoint is loaded.
"""

import sys
import types

from torch import nn

# Module paths that existed in older releases and may be referenced by old checkpoints.
_LEGACY_MODULE_PATHS = (
    "anomalib.models.components.dinov2.dinov2_loader",
    "anomalib.models.components.dinov2.vision_transformer",
    "anomalib.models.image.dinomaly.components.vision_transformer",
)


class _LegacyDinoV2Placeholder(nn.Module):
    """Permissive stand-in for removed DINOv2 ViT classes/factories.

    Resolves any pickled reference to a removed symbol. It is never used for inference - the real
    encoder is a frozen ``TimmFeatureExtractor`` - so it just needs to unpickle without error.
    """

    def __init__(self, *_args, **_kwargs) -> None:
        super().__init__()


def _legacy_getattr(name: str) -> type[_LegacyDinoV2Placeholder]:
    """Return the placeholder for any symbol requested on a legacy module.

    Dunder names (``__file__``, ``__spec__``, ...) must raise ``AttributeError`` so normal module
    machinery and introspection (e.g. ``inspect.getmodule`` walking ``sys.modules``) keep working;
    only real pickled symbols are resolved to the placeholder.
    """
    if name.startswith("__") and name.endswith("__"):
        raise AttributeError(name)
    return _LegacyDinoV2Placeholder


def install_legacy_module_aliases() -> None:
    """Register placeholder modules for removed import paths (idempotent)."""
    for path in _LEGACY_MODULE_PATHS:
        if path in sys.modules:
            # A real module still exists at this path - never shadow it.
            continue
        module = types.ModuleType(path)
        module.__getattr__ = _legacy_getattr  # type: ignore[method-assign]
        sys.modules[path] = module


install_legacy_module_aliases()
