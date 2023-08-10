"""Test all prediction storage classes"""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import copy
from tempfile import TemporaryDirectory
from typing import Callable

import torch
from torch import Tensor
from torchmetrics import StructuralSimilarityIndexMeasure

from anomalib.data import AnomalibDataModule
from tools.tiled_ensemble.predictions import (
    BasicEnsemblePredictions,
    EnsemblePredictions,
    FileSystemEnsemblePredictions,
    RescaledEnsemblePredictions,
)


class TestPredictionData:
    @staticmethod
    def store_all(data: EnsemblePredictions, datamodule: AnomalibDataModule):
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
    def verify_equal(name: str, tile_dict: dict, storage: EnsemblePredictions, eq_funct: Callable):
        """Verify that all data at same tile index and same batch index matches"""
        batch_num = len(tile_dict[(0, 0)])

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
                else:
                    if original_data != stored_data:
                        return False

        return True

    def test_basic(self, get_ensemble_config, get_datamodule):
        config = get_ensemble_config
        datamodule = get_datamodule(config, "segmentation")
        storage = BasicEnsemblePredictions()
        original = self.store_all(storage, datamodule)

        for name in original[(0, 0)][0].keys():
            assert self.verify_equal(name, original, storage, torch.equal), f"{name} doesn't match"

    def test_fs(self, get_ensemble_config, get_datamodule):
        config = get_ensemble_config
        with TemporaryDirectory() as project_dir:
            config.project.path = project_dir
            datamodule = get_datamodule(config, "segmentation")
            storage = FileSystemEnsemblePredictions(storage_path=config.project.path)
            original = self.store_all(storage, datamodule)

            for name in original[(0, 0)][0].keys():
                assert self.verify_equal(name, original, storage, torch.equal), f"{name} doesn't match"

    def test_rescaled(self, get_ensemble_config, get_datamodule):
        config = get_ensemble_config
        datamodule = get_datamodule(config, "segmentation")
        storage = RescaledEnsemblePredictions(config.ensemble.predictions.rescale_factor)
        original = self.store_all(storage, datamodule)

        # we want rescaled to be close to original, but we lose some information in scaling
        ssim = StructuralSimilarityIndexMeasure()

        def rescale_eq(input_tensor, other_tensor):
            # check if all zero
            if not (torch.any(input_tensor) and torch.any(other_tensor)):
                return True

            return ssim(input_tensor, other_tensor) > 0.8

        for name in original[(0, 0)][0].keys():
            if name in ["image", "anomaly_maps", "pred_masks"]:
                assert self.verify_equal(name, original, storage, rescale_eq), f"{name} doesn't match"
            else:
                # gt mask and other must not be changed
                assert self.verify_equal(name, original, storage, torch.equal), f"{name} doesn't match"
