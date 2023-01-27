"""Tiler Callback."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

from typing import Sequence

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

from anomalib.models.components import AnomalyModule
from anomalib.pre_processing.tiler import Tiler

__all__ = ["TilerConfigurationCallback"]


class TilerConfigurationCallback(Callback):
    """Tiler Configuration Callback."""

    def __init__(
        self,
        enable: bool = False,
        tile_size: int | Sequence = 256,
        stride: int | Sequence | None = None,
        remove_border_count: int = 0,
        mode: str = "padding",
        tile_count: int = 4,
    ) -> None:
        """Sets tiling configuration from the command line.

        Args:
            enable (bool): Boolean to enable tiling operation.
                Defaults to False.
            tile_size ([int | Sequence]): Tile size.
                Defaults to 256.
            stride ([int | Sequence]): Stride to move tiles on the image.
            remove_border_count (int, optional): Number of pixels to remove from the image before
                tiling. Defaults to 0.
            mode (str, optional): Up-scaling mode when untiling overlapping tiles.
                Defaults to "padding".
            tile_count (SupportsIndex, optional): Number of random tiles to sample from the image.
                Defaults to 4.
        """
        self.enable = enable
        self.tile_size = tile_size
        self.stride = stride
        self.remove_border_count = remove_border_count
        self.mode = mode
        self.tile_count = tile_count

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str | None = None) -> None:
        """Setup Tiler object within Anomalib Model.

        Args:
            trainer (pl.Trainer): PyTorch Lightning Trainer
            pl_module (pl.LightningModule): Anomalib Model that inherits pl LightningModule.
            stage (str | None, optional): fit, validate, test or predict. Defaults to None.

        Raises:
            ValueError: When Anomalib Model doesn't contain ``Tiler`` object, it means the model
                doesn not support tiling operation.
        """
        del trainer, stage  # These variables are not used.

        if self.enable:
            if isinstance(pl_module, AnomalyModule) and hasattr(pl_module.model, "tiler"):
                pl_module.model.tiler = Tiler(
                    tile_size=self.tile_size,
                    stride=self.stride,
                    remove_border_count=self.remove_border_count,
                    mode=self.mode,
                    tile_count=self.tile_count,
                )
            else:
                raise ValueError("Model does not support tiling.")
