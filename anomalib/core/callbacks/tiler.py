"""Tiler Callback."""

# Copyright (C) 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.


from typing import Optional, Sequence, SupportsIndex, Union

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


class TilerConfigurationCallback(Callback):
    """Tiler Configuration Callback."""

    def __init__(
        self,
        enable: bool = False,
        tile_size: Optional[Union[int, Sequence]] = None,
        stride: Optional[Union[int, Sequence]] = None,
        remove_border_count: int = 0,
        mode: str = "padding",
        tile_count: SupportsIndex = 4,
    ):
        """Sets tiling configuration from the command line.

        Args:
            enable (bool, optional): Boolean to enable tiling operation. Defaults to False.
            tile_size (Optional[Union[int, Sequence]], optional): Tile size. Defaults to None.
            stride (Optional[Union[int, Sequence]], optional): Stride to move tiles on the image.
                Defaults to None.
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

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        pass

    def on_validation_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        pass

    def on_test_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        pass

    def on_predict_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        pass
