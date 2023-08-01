"""Classes used to store ensemble predictions."""
import copy
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from abc import ABC
from pathlib import Path

from typing import List

from omegaconf import DictConfig, ListConfig

import torch
from torch import Tensor
import torch.nn.functional as F


class EnsemblePredictions(ABC):
    """
    Abstract class used as template for different ways of storing ensemble predictions.
    """

    def __init__(self) -> None:
        self.num_batches = 0

    def add_tile_prediction(
        self, tile_index: (int, int), tile_prediction: list[dict[str, Tensor | List | str]]
    ) -> None:
        """
        Add tile prediction data at specified tile index.

        Args:
            tile_index: Index of tile that we are adding in form (row, column).
            tile_prediction: List of batches containing all predicted data for current tile.
        """
        raise NotImplementedError

    def get_batch_tiles(self, batch_index: int) -> dict[(int, int), dict]:
        """
        Get all tiles of current batch.

        Args:
            batch_index: Index of current batch of tiles to be returned.

        Returns:
            Dictionary mapping tile index to predicted data, for provided batch index.
        """
        raise NotImplementedError


class BasicEnsemblePredictions(EnsemblePredictions):
    """
    Basic implementation of EnsemblePredictionData that keeps all predictions in memory as they are.
    """

    def __init__(self) -> None:
        super().__init__()
        self.all_data = {}

    def add_tile_prediction(
        self, tile_index: (int, int), tile_prediction: list[dict[str, Tensor | List | str]]
    ) -> None:
        """
        Add tile prediction data at provided index to class dictionary in main memory.

        Args:
            tile_index: Index of tile that we are adding in form (row, column).
            tile_prediction: List of batches containing all predicted data for current tile.

        """
        self.num_batches = len(tile_prediction)

        self.all_data[tile_index] = tile_prediction

    def get_batch_tiles(self, batch_index: int) -> dict[(int, int), dict]:
        """
        Get all tiles of current batch from class dictionary.

        Args:
            batch_index: Index of current batch of tiles to be returned.

        Returns:
            Dictionary mapping tile index to predicted data, for provided batch index.
        """
        batch_data = {}

        for index, batches in self.all_data.items():
            batch_data[index] = batches[batch_index]
            # copy anomaly maps, since in case of test data == val data, post-processing might change them
            batch_data[index]["anomaly_maps"] = batch_data[index]["anomaly_maps"].clone()

        return batch_data


class FileSystemEnsemblePredictions(EnsemblePredictions):
    """
    Implementation of EnsemblePredictionData that stores all predictions to file system.

    Args:
        config: Config file with all parameters.
    """

    def __init__(self, config: DictConfig | ListConfig) -> None:
        super().__init__()
        self.tile_indices = []

        project_path = Path(config.project.path)
        self.tiles_path = project_path / "tile_predictions"

        self.tiles_path.mkdir()

    def add_tile_prediction(
        self, tile_index: (int, int), tile_prediction: list[dict[str, Tensor | List | str]]
    ) -> None:
        """
        Save predictions from current position to file system in following hierarchy:

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
            tile_index: Index of current tile position that we are saving.
            tile_prediction: Predictions at tile position.

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

    def get_batch_tiles(self, batch_index: int) -> dict[(int, int), dict]:
        """
        Load batches from file system and assemble into dict.

        Args:
            batch_index: Index of current batch of tiles to be returned.

        Returns:
            Dictionary mapping tile index to predicted data, for provided batch index.
        """
        batch_data = {}

        for tile_i, tile_j in self.tile_indices:
            tile_batch_path = self.tiles_path / f"{tile_i}_{tile_j}" / f"batch_{batch_index}"
            tile_batch_data = torch.load(tile_batch_path)

            batch_data[(tile_i, tile_j)] = tile_batch_data

        return batch_data


class RescaledEnsemblePredictions(EnsemblePredictions):
    """
    Implementation of EnsemblePredictionData that keeps all predictions in memory but scaled down

    Args:
        config: Config file with all parameters.

    """

    def __init__(self, config: DictConfig | ListConfig) -> None:
        super().__init__()
        self.all_data = {}
        self.downscale_factor = config.ensemble.predictions.rescale_factor
        self.upscale_factor = 1 / self.downscale_factor

    @staticmethod
    def _rescale(batch: dict, scale_factor: float, mode: str) -> dict:
        """
        Rescale all tile data in batch for specified factor.

        Args:
            batch: Dictionary of all predicted data.
            scale_factor: Factor by which scaling will be done.
            mode:

        Returns:
            Dictionary of all predicted data with all tiles rescaled.
        """
        # copy data
        batch = copy.copy(batch)

        # downscale following but NOT gt mask
        tiled_keys = ["image", "anomaly_maps", "pred_masks"]

        # change bool to float32
        if "pred_masks" in batch.keys():
            batch["pred_masks"] = batch["pred_masks"].type(torch.float32)

        for key in tiled_keys:
            if key in batch.keys():
                batch[key] = F.interpolate(batch[key], scale_factor=scale_factor, mode=mode)

        return batch

    def add_tile_prediction(
        self, tile_index: (int, int), tile_prediction: list[dict[str, Tensor | List | str]]
    ) -> None:
        """
        Rescale tile prediction data and add it at provided index to class dictionary in main memory.

        Args:
            tile_index: Index of tile that we are adding in form (row, column).
            tile_prediction: List of batches containing all predicted data for current tile.

        """
        self.num_batches = len(tile_prediction)

        rescaled = []
        for batch in tile_prediction:
            rescaled.append(self._rescale(batch, self.downscale_factor, "bicubic"))

        self.all_data[tile_index] = rescaled

    def get_batch_tiles(self, batch_index: int) -> dict[(int, int), dict]:
        """
        Get all tiles of current batch from class dictionary, rescaled to original size.

        Args:
            batch_index: Index of current batch of tiles to be returned.

        Returns:
            Dictionary mapping tile index to predicted data, for provided batch index.
        """
        batch_data = {}

        for tile_index, batches in self.all_data.items():
            current_batch_data = batches[batch_index]

            current_batch_data = self._rescale(current_batch_data, scale_factor=self.upscale_factor, mode="bicubic")

            batch_data[tile_index] = current_batch_data

        return batch_data
