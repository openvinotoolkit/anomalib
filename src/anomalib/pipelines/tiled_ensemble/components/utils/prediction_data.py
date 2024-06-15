"""Classes used to store ensemble predictions."""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from torch import Tensor


class EnsemblePredictions:
    """Basic implementation of EnsemblePredictionData that keeps all predictions in memory as they are.

    Examples:
        >>> from pytorch_lightning import Trainer
        >>> data = EnsemblePredictions()
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
        """Add tile prediction data at provided index to class dictionary in main memory.

        Args:
            tile_index (tuple[int, int]): Index of tile that we are adding in form (row, column).
            tile_prediction (list[dict[str, Tensor | list]]):
                List of batches containing all predicted data for current tile position.

        """
        self.num_batches = len(tile_prediction)

        self.all_data[tile_index] = tile_prediction

    def get_batch_tiles(self, batch_index: int) -> dict[tuple[int, int], dict]:
        """Get all tiles of current batch from class dictionary.

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
