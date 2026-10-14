# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint IO plugin that allowlists anomalib types for safe loading."""

from collections.abc import Sequence
from typing import Any

from lightning.fabric.plugins.io.torch_io import TorchCheckpointIO
from lightning.fabric.utilities.types import _MAP_LOCATION_TYPE, _PATH
from typing_extensions import override

from anomalib.utils.serialization import anomalib_safe_globals


class AnomalibCheckpointIO(TorchCheckpointIO):
    """Torch checkpoint IO that allowlists anomalib enums under ``weights_only=True``.

    Used by the Lightning Trainer for ``ckpt_path`` restores on fit, validate, test,
    and predict. Direct ``AnomalibModule.load_from_checkpoint`` uses the same allowlist
    via a separate override.

    Args:
        extra_safe_globals: Model-specific types to allowlist in addition to the
            shared ``ANOMALIB_SAFE_GLOBALS``. Typically set from
            ``AnomalibModule.checkpoint_safe_globals()``.
    """

    def __init__(self, extra_safe_globals: Sequence[Any] | None = None) -> None:
        super().__init__()
        self.extra_safe_globals = list(extra_safe_globals or ())

    @override
    def load_checkpoint(
        self,
        path: _PATH,
        map_location: _MAP_LOCATION_TYPE = None,
        weights_only: bool = True,
    ) -> dict[str, Any]:
        """Load a checkpoint with anomalib types allowlisted for ``weights_only`` loads.

        Args:
            path: Path to the checkpoint file.
            map_location: Device mapping passed to ``torch.load`` (device, string,
                mapping, or callable). Defaults to ``None``.
            weights_only: Whether to restrict unpickling to tensors and allowlisted
                types. Defaults to ``True``.

        Returns:
            dict[str, Any]: Loaded checkpoint dictionary.
        """
        with anomalib_safe_globals(extra=self.extra_safe_globals):
            return super().load_checkpoint(path, map_location=map_location, weights_only=weights_only)
