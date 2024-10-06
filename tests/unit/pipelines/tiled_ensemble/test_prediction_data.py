"""Test tiled prediction storage class."""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import copy
from collections.abc import Callable

import torch
from torch import Tensor

from anomalib.data import AnomalibDataModule
from anomalib.pipelines.tiled_ensemble.components.utils.prediction_data import EnsemblePredictions


class TestPredictionData:
    """Test EnsemblePredictions class, used for tiled prediction storage."""

    @staticmethod
    def store_all(data: EnsemblePredictions, datamodule: AnomalibDataModule) -> dict:
        """Store the tiled predictions in the EnsemblePredictions object."""
        tile_dict = {}
        for tile_index in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            datamodule.collate_fn.tile_index = tile_index

            tile_prediction = []
            for batch in iter(datamodule.train_dataloader()):
                # set mock maps to just one channel of image
                batch["anomaly_maps"] = batch["image"].clone()[:, 0, :, :].unsqueeze(1)
                # set mock pred mask to mask but add channel
                batch["pred_masks"] = batch["mask"].clone().unsqueeze(1)
                tile_prediction.append(batch)
            # save original
            tile_dict[tile_index] = copy.deepcopy(tile_prediction)
            # store to prediction storage object
            data.add_tile_prediction(tile_index, tile_prediction)

        return tile_dict

    @staticmethod
    def verify_equal(name: str, tile_dict: dict, storage: EnsemblePredictions, eq_funct: Callable) -> bool:
        """Verify that all data at same tile index and same batch index matches."""
        batch_num = len(tile_dict[0, 0])

        for batch_i in range(batch_num):
            # batch is dict where key: tile index and val is batched data of that tile
            curr_batch = storage.get_batch_tiles(batch_i)

            # go over all indices of current batch of stored data
            for tile_index, stored_data_batch in curr_batch.items():
                stored_data = stored_data_batch[name]
                # get original data dict at current tile index and batch index
                original_data = tile_dict[tile_index][batch_i][name]
                if isinstance(original_data, Tensor):
                    if not eq_funct(original_data, stored_data):
                        return False
                elif original_data != stored_data:
                    return False

        return True

    def test_prediction_object(self, get_datamodule: AnomalibDataModule) -> None:
        """Test prediction storage class."""
        datamodule = get_datamodule
        storage = EnsemblePredictions()
        original = self.store_all(storage, datamodule)

        for name in original[0, 0][0]:
            assert self.verify_equal(name, original, storage, torch.equal), f"{name} doesn't match"
