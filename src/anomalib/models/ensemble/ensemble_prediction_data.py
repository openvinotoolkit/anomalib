"""Classes used to store ensemble predictions."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from abc import ABC
from pathlib import Path
from typing import List

import torch
from omegaconf import DictConfig, ListConfig
from torch import Tensor


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
