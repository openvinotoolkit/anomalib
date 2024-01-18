"""Classes used to store ensemble predictions."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
from abc import ABC
from enum import Enum
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor


class PredictionStorageType(str, Enum):
    """
    Enum signaling how tiled ensemble predictions are stored.

    memory - predictions are stored in memory,
    file_system - predictions are stored in the file system,
    memory_downscaled - predictions are downscaled and stored in memory.
    """

    MEMORY = "memory"
    FILE_SYSTEM = "file_system"
    MEMORY_DOWNSCALED = "memory_downscaled"


class EnsemblePredictions(ABC):
    """Abstract class used as template for different ways of storing ensemble predictions."""

    def __init__(self) -> None:
        self.num_batches = 0

    def add_tile_prediction(self, tile_index: tuple[int, int], tile_prediction: list[dict[str, Tensor | list]]) -> None:
        """
        Add tile prediction data at specified tile index.

        Args:
            tile_index (tuple[int, int]): Index of tile that we are adding in form (row, column).
            tile_prediction (tile_prediction: list[dict[str, Tensor | list]]):
                List of batches containing all predicted data for current tile position.
        """
        raise NotImplementedError

    def get_batch_tiles(self, batch_index: int) -> dict[tuple[int, int], dict]:
        """
        Get all tiles of current batch.

        Args:
            batch_index (int): Index of current batch of tiles to be returned.

        Returns:
            dict[tuple[int, int], dict]: Dictionary mapping tile index to predicted data, for provided batch index.
        """
        raise NotImplementedError


class MemoryEnsemblePredictions(EnsemblePredictions):
    """Basic implementation of EnsemblePredictionData that keeps all predictions in memory as they are.

    Examples:
        >>> from pytorch_lightning import Trainer
        >>> data = MemoryEnsemblePredictions()
        >>> trainer = Trainer(...)
        >>>
        >>> # predictions for all batches for tile location can be added
        >>> curr_predictions = trainer.predict()
        >>> # curr_predictions is list of batches predicted for index
        >>> data.add_tile_prediction((0, 0), curr_predictions)

        >>> # all tile location predictions can then be obtained for each batch
        >>> batch_data = data.get_batch_tiles(0)
        >>> # batch_data is dictionary where keys are all stored tile indices, and values are batch
        >>> # of predictions at tile location for current index
    """

    def __init__(self) -> None:
        super().__init__()
        self.all_data: dict[tuple[int, int], list] = {}

    def add_tile_prediction(self, tile_index: tuple[int, int], tile_prediction: list[dict[str, Tensor | list]]) -> None:
        """
        Add tile prediction data at provided index to class dictionary in main memory.

        Args:
            tile_index (tuple[int, int]): Index of tile that we are adding in form (row, column).
            tile_prediction (list[dict[str, Tensor | list]]):
                List of batches containing all predicted data for current tile position.

        """
        self.num_batches = len(tile_prediction)

        self.all_data[tile_index] = tile_prediction

    def get_batch_tiles(self, batch_index: int) -> dict[tuple[int, int], dict]:
        """
        Get all tiles of current batch from class dictionary.

        Args:
            batch_index (int): Index of current batch of tiles to be returned.

        Returns:
            dict[tuple[int, int], dict]: Dictionary mapping tile index to predicted data, for provided batch index.
        """
        batch_data = {}

        for index, batches in self.all_data.items():
            batch_data[index] = batches[batch_index]
            if "anomaly_maps" in batch_data[index]:
                # copy anomaly maps, since in case of test data == val data, post-processing might change them
                batch_data[index]["anomaly_maps"] = batch_data[index]["anomaly_maps"].clone()

        return batch_data


class FileSystemEnsemblePredictions(EnsemblePredictions):
    """
    Implementation of EnsemblePredictionData that stores all predictions to file system.

    Args:
        storage_path (str): Path to directory where predictions will be saved.

    Examples:
        >>> from pytorch_lightning import Trainer
        >>> data = FileSystemEnsemblePredictions("path_to_project_or_other")
        >>> trainer = Trainer(...)
        >>>
        >>> # predictions for all batches for tile location can be added
        >>> curr_predictions = trainer.predict()
        >>> # curr_predictions is list of batches predicted for index
        >>> data.add_tile_prediction((0, 0), curr_predictions)
        >>> # behind the scenes the predictions are saved to FS and are then loaded when needed

        >>> # all tile location predictions can then be obtained for each batch
        >>> batch_data = data.get_batch_tiles(0)
        >>> # batch_data is dictionary where keys are all stored tile indices, and values are batch
        >>> # of predictions at tile location for current index
    """

    def __init__(self, storage_path: str) -> None:
        super().__init__()
        self.tile_indices: list[tuple[int, int]] = []

        project_path = Path(storage_path)
        self.tiles_path = project_path / "tile_predictions"

        self.tiles_path.mkdir()

    def add_tile_prediction(self, tile_index: tuple[int, int], tile_prediction: list[dict[str, Tensor | list]]) -> None:
        """
        Save predictions from current position to file system in following hierarchy.

        tile_predictions
        ...0_0
        ......batch_0
                .
                .
        ...0_1
        ......batch_0
        .
        .

        Args:
            tile_index (tuple[int, int]): Index of tile that we are adding in form (row, column).
            tile_prediction (list[dict[str, Tensor | list]]):
                List of batches containing all predicted data for current tile position.

        """
        self.num_batches = len(tile_prediction)

        # prepare (if it doesn't exist) directory for current tile position
        current_pred_path = self.tiles_path / f"{tile_index[0]}_{tile_index[1]}"
        current_pred_path.mkdir(exist_ok=True)

        # save possible tile index
        self.tile_indices.append(tile_index)

        for i, batch in enumerate(tile_prediction):
            # save as row_col/batch_i
            saved_batch_name = current_pred_path / f"batch_{i}"
            torch.save(batch, saved_batch_name)

            # clear from dict (GC will remove from memory)
            batch.clear()

    def get_batch_tiles(self, batch_index: int) -> dict[tuple[int, int], dict]:
        """
        Load batches from file system and assemble into dict.

        Args:
            batch_index (int): Index of current batch of tiles to be returned.

        Returns:
            dict[tuple[int, int], dict]: Dictionary mapping tile index to predicted data, for provided batch index.
        """
        batch_data = {}

        for tile_i, tile_j in self.tile_indices:
            tile_batch_path = self.tiles_path / f"{tile_i}_{tile_j}" / f"batch_{batch_index}"
            tile_batch_data = torch.load(tile_batch_path)

            batch_data[(tile_i, tile_j)] = tile_batch_data

        return batch_data


class DownscaledEnsemblePredictions(EnsemblePredictions):
    """
    Implementation of EnsemblePredictionData that keeps all predictions in memory but scaled down.

    Args:
        downscale_factor (float): Factor by which the tile based predictions (image, maps..) will be downscaled.

    Examples:
        >>> from pytorch_lightning import Trainer
        >>> data = DownscaledEnsemblePredictions(0.5)
        >>> trainer = Trainer(...)
        >>>
        >>> # predictions for all batches for tile location can be added
        >>> curr_predictions = trainer.predict()
        >>> # curr_predictions is list of batches predicted for index
        >>> data.add_tile_prediction((0, 0), curr_predictions)
        >>> # behind the scenes the predictions are downscaled and then upscaled again when needed

        >>> # all tile location predictions can then be obtained for each batch
        >>> batch_data = data.get_batch_tiles(0)
        >>> # batch_data is dictionary where keys are all stored tile indices, and values are batch
        >>> # of predictions at tile location for current index
    """

    def __init__(self, downscale_factor: float) -> None:
        super().__init__()
        self.all_data: dict[tuple[int, int], list] = {}
        self.downscale_factor = downscale_factor
        self.upscale_factor = 1 / self.downscale_factor

    @staticmethod
    def _rescale(batch: dict, scale_factor: float, mode: str) -> dict:
        """
        Rescale all tile data in batch for specified factor.

        Args:
            batch (dict): Dictionary of all predicted data.
            scale_factor (float): Factor by which scaling will be done.
            mode (str): Rescaling mode used for interpolation.

        Returns:
            dict: Dictionary of all predicted data with all tiles rescaled.
        """
        # copy dictionary, but not underlying data, so other data still stays downscaled in memory
        batch = copy.copy(batch)

        # downscale following but NOT gt mask
        tiled_keys = ["image", "anomaly_maps", "pred_masks"]

        # change bool to float32
        if "pred_masks" in batch:
            batch["pred_masks"] = batch["pred_masks"].type(torch.float32)

        for key in tiled_keys:
            if key in batch:
                batch[key] = F.interpolate(batch[key], scale_factor=scale_factor, mode=mode)

        return batch

    def add_tile_prediction(self, tile_index: tuple[int, int], tile_prediction: list[dict[str, Tensor | list]]) -> None:
        """
        Rescale tile prediction data and add it at provided index to class dictionary in main memory.

        Args:
            tile_index (tuple[int, int]): Index of tile that we are adding in form (row, column).
            tile_prediction (list[dict[str, Tensor | list]]):
                List of batches containing all predicted data for current tile position.

        """
        self.num_batches = len(tile_prediction)

        rescaled = []
        for batch in tile_prediction:
            rescaled.append(self._rescale(batch, self.downscale_factor, "bicubic"))

        self.all_data[tile_index] = rescaled

    def get_batch_tiles(self, batch_index: int) -> dict[tuple[int, int], dict]:
        """
        Get all tiles of current batch from class dictionary, rescaled to original size.

        Args:
            batch_index (int): Index of current batch of tiles to be returned.

        Returns:
            dict[tuple[int, int], dict]: Dictionary mapping tile index to predicted data, for provided batch index.
        """
        batch_data = {}

        for tile_index, batches in self.all_data.items():
            current_batch_data = batches[batch_index]

            current_batch_data = self._rescale(current_batch_data, scale_factor=self.upscale_factor, mode="bicubic")

            batch_data[tile_index] = current_batch_data

        return batch_data
