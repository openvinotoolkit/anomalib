"""Test ensemble helper functions"""
from typing import List

import pytest
from omegaconf import OmegaConf
from torch import Tensor

from anomalib.data import MVTec
from anomalib.models.ensemble import EnsembleTiler
from anomalib.models.ensemble.ensemble_functions import TileCollater
from tests.helpers.dataset import get_dataset_path


tiler_config = OmegaConf.create(
    {
        "ensemble": {
            "tiling": {
                "tile_size": 256,
                "stride": 256,
            }
        },
        "dataset": {"image_size": 512},
    }
)


def get_datamodule(task):
    datamodule = MVTec(
        root=get_dataset_path(dataset="MVTec"),
        category="leather",
        image_size=tiler_config.dataset.image_size,
        train_batch_size=5,
        eval_batch_size=5,
        num_workers=0,
        task=task,
        test_split_mode="from_dir",
        val_split_mode="same_as_test",
    )
    datamodule.prepare_data()
    datamodule.setup()

    tiler = EnsembleTiler(tiler_config)
    datamodule.custom_collate_fn = TileCollater(tiler, (0, 0))

    return datamodule


class TestTileCollater:
    """Test tile collater"""

    def test_collate_tile_shape(self):
        # datamodule with tile collater
        datamodule = get_datamodule("segmentation")

        tile_size = tiler_config.ensemble.tiling.tile_size

        batch = next(iter(datamodule.train_dataloader()))
        assert batch["image"].shape == (5, 3, tile_size, tile_size)
        assert batch["mask"].shape == (5, tile_size, tile_size)

    def test_collate_box_data(self):
        # datamodule with tile collater
        datamodule = get_datamodule("detection")

        batch = next(iter(datamodule.train_dataloader()))

        # assert that base collate function was called
        assert isinstance(batch["boxes"], List)
        assert isinstance(batch["boxes"][0], Tensor)
