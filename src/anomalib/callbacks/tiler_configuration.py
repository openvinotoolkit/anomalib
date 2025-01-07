"""Tiler configuration callback.

This module provides the :class:`TilerConfigurationCallback` for configuring image tiling operations
in Anomalib models. Tiling allows processing large images by splitting them into smaller tiles,
which is useful when dealing with high-resolution images that don't fit in GPU memory.

The callback configures tiling parameters such as tile size, stride, and upscaling mode for
models that support tiling operations.

Example:
    Configure tiling with custom parameters:

    >>> from anomalib.callbacks import TilerConfigurationCallback
    >>> from anomalib.data.utils.tiler import ImageUpscaleMode
    >>> callback = TilerConfigurationCallback(
    ...     enable=True,
    ...     tile_size=512,
    ...     stride=256,
    ...     mode=ImageUpscaleMode.PADDING
    ... )
    >>> from lightning.pytorch import Trainer
    >>> trainer = Trainer(callbacks=[callback])

Note:
    The model must support tiling operations for this callback to work.
    It will raise a :exc:`ValueError` if used with a model that doesn't support tiling.
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence

import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback

from anomalib.data.utils.tiler import ImageUpscaleMode, Tiler
from anomalib.models.components import AnomalibModule

__all__ = ["TilerConfigurationCallback"]


class TilerConfigurationCallback(Callback):
    """Callback for configuring image tiling operations.

    This callback configures the tiling operation for models that support it. Tiling is useful
    when working with high-resolution images that need to be processed in smaller chunks.

    Args:
        enable (bool): Whether to enable tiling operation. Defaults to ``False``.
        tile_size (int | Sequence): Size of each tile. Can be a single integer for square tiles
            or a sequence of two integers for rectangular tiles. Defaults to ``256``.
        stride (int | Sequence | None): Stride between tiles. Can be a single integer or a sequence
            of two integers. If ``None``, uses ``tile_size``. Defaults to ``None``.
        remove_border_count (int): Number of pixels to remove from the image border before
            tiling. Useful for removing artifacts at image boundaries. Defaults to ``0``.
        mode (ImageUpscaleMode): Method to use when combining overlapping tiles.
            Options are defined in :class:`~anomalib.data.utils.tiler.ImageUpscaleMode`.
            Defaults to ``ImageUpscaleMode.PADDING``.

    Examples:
        Create a basic tiling configuration:

        >>> callback = TilerConfigurationCallback(enable=True)

        Configure tiling with custom tile size and stride:

        >>> callback = TilerConfigurationCallback(
        ...     enable=True,
        ...     tile_size=512,
        ...     stride=256
        ... )

        Use rectangular tiles with custom upscale mode:

        >>> from anomalib.data.utils.tiler import ImageUpscaleMode
        >>> callback = TilerConfigurationCallback(
        ...     enable=True,
        ...     tile_size=(512, 256),
        ...     mode=ImageUpscaleMode.AVERAGE
        ... )

    Raises:
        ValueError: If used with a model that doesn't support tiling operations.

    Note:
        - The model must have a ``tiler`` attribute to support tiling operations
        - Smaller stride values result in more overlap between tiles but increase computation
        - The upscale mode affects how overlapping regions are combined
    """

    def __init__(
        self,
        enable: bool = False,
        tile_size: int | Sequence = 256,
        stride: int | Sequence | None = None,
        remove_border_count: int = 0,
        mode: ImageUpscaleMode = ImageUpscaleMode.PADDING,
    ) -> None:
        """Initialize tiling configuration."""
        self.enable = enable
        self.tile_size = tile_size
        self.stride = stride
        self.remove_border_count = remove_border_count
        self.mode = mode

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str | None = None) -> None:
        """Set Tiler object within Anomalib Model.

        This method is called by PyTorch Lightning during setup. It configures the tiling
        parameters if tiling is enabled and the model supports it.

        Args:
            trainer (pl.Trainer): PyTorch Lightning Trainer instance.
            pl_module (pl.LightningModule): The Anomalib model being trained/tested.
            stage (str | None, optional): Current stage - ``"fit"``, ``"validate"``,
                ``"test"`` or ``"predict"``. Defaults to ``None``.

        Raises:
            ValueError: If tiling is enabled but the model doesn't support tiling operations
                (i.e., doesn't have a ``tiler`` attribute).
        """
        del trainer, stage  # These variables are not used.

        if self.enable:
            if isinstance(pl_module, AnomalibModule) and hasattr(pl_module.model, "tiler"):
                pl_module.model.tiler = Tiler(
                    tile_size=self.tile_size,
                    stride=self.stride,
                    remove_border_count=self.remove_border_count,
                    mode=self.mode,
                )
            else:
                msg = "Model does not support tiling."
                raise ValueError(msg)
