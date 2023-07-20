"""Classes used to store ensemble predictions."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from abc import ABC
from typing import List

from torch import Tensor


class EnsemblePredictionData(ABC):
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


class BasicEnsemblePredictionData(EnsemblePredictionData):
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
