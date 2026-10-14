# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Components for the FoundAD model.

This module provides the manifold projector, encoder loader, CutPaste
augmentation, and few-shot training augmentation components used in the
FoundAD architecture.
"""

from .augmentation import FewShotAugmentation
from .cutpaste import CutPasteUnion
from .encoder_loader import load_encoder
from .manifold_projector import ManifoldProjector

__all__ = ["CutPasteUnion", "FewShotAugmentation", "ManifoldProjector", "load_encoder"]
