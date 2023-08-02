"""Test ensemble helper functions"""

from tempfile import TemporaryDirectory
from typing import List

from omegaconf import OmegaConf
from torch import Tensor

from anomalib.config import get_configurable_parameters
from anomalib.data import MVTec
from anomalib.models.ensemble import EnsembleTiler
from anomalib.models.ensemble.ensemble_functions import (
    TileCollater,
    prepare_ensemble_configurable_parameters,
    get_ensemble_datamodule,
    get_prediction_storage,
    get_ensemble_callbacks,
)
from anomalib.models.ensemble.predictions import (
    BasicEnsemblePredictions,
    FileSystemEnsemblePredictions,
    RescaledEnsemblePredictions,
)
from anomalib.utils.callbacks import MinMaxNormalizationCallback
from tests.helpers.dataset import get_dataset_path


mock_config = OmegaConf.create(
    {
        "ensemble": {
            "tiling": {
                "tile_size": 256,
                "stride": 256,
            },
            "predictions": {"storage": "direct", "rescale_factor": 0.5},
        },
        "dataset": {"image_size": 512, "val_split_mode": "same_as_test"},
        "project": {"path": ""},
    }
)


def get_datamodule(task):
    datamodule = MVTec(
        root=get_dataset_path(dataset="MVTec"),
        category="leather",
        image_size=mock_config.dataset.image_size,
        train_batch_size=5,
        eval_batch_size=5,
        num_workers=0,
        task=task,
        test_split_mode="from_dir",
        val_split_mode="same_as_test",
    )
    datamodule.prepare_data()
    datamodule.setup()

    tiler = EnsembleTiler(mock_config)
    datamodule.custom_collate_fn = TileCollater(tiler, (0, 0))

    return datamodule


class TestTileCollater:
    """Test tile collater"""

    def test_collate_tile_shape(self):
        # datamodule with tile collater
        datamodule = get_datamodule("segmentation")

        tile_size = mock_config.ensemble.tiling.tile_size

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


class TestHelperFunctions:
    """Test other ensemble helper functions"""

    def test_ensemble_config(self):
        config = get_configurable_parameters(config_path="src/anomalib/models/padim/config.yaml")
        prepare_ensemble_configurable_parameters(
            ens_config_path="tests/pre_merge/models/ensemble/dummy_ens_config.yaml", config=config
        )

        assert "ensemble" in config
        assert config.model.input_size == config.ensemble.tiling.tile_size

    def test_ensemble_datamodule(self):
        config = get_configurable_parameters(config_path="src/anomalib/models/padim/config.yaml")

        tiler = EnsembleTiler(mock_config)
        datamodule = get_ensemble_datamodule(config, tiler)

        assert isinstance(datamodule.custom_collate_fn, TileCollater)

    def test_ensemble_prediction_storage_type(self):
        config = mock_config.copy()

        config.ensemble.predictions.storage = "direct"
        pred_direct = get_prediction_storage(config)

        assert isinstance(pred_direct[0], BasicEnsemblePredictions)
        assert isinstance(pred_direct[1], BasicEnsemblePredictions)

        config.ensemble.predictions.storage = "file_system"

        with TemporaryDirectory() as project_dir:
            config.project.path = project_dir

            pred_fs = get_prediction_storage(config)
            assert isinstance(pred_fs[0], FileSystemEnsemblePredictions)
            assert isinstance(pred_fs[1], FileSystemEnsemblePredictions)

        config.ensemble.predictions.storage = "rescaled"
        pred_rescale = get_prediction_storage(config)

        assert isinstance(pred_rescale[0], RescaledEnsemblePredictions)
        assert isinstance(pred_rescale[1], RescaledEnsemblePredictions)

    def test_ensemble_prediction_storage_reference(self):
        config = mock_config.copy()
        config.dataset.val_split_mode = "same_as_test"
        pred_same = get_prediction_storage(config)

        # if test data is same as validation, storage should be same
        assert pred_same[0] is pred_same[1]

        config.dataset.val_split_mode = "from_test"
        pred_from = get_prediction_storage(config)

        # if test data is NOT same as validation, storage should be separate object
        assert pred_from[0] is not pred_from[1]

    def test_ensemble_callbacks(self):
        config = get_configurable_parameters(config_path="src/anomalib/models/padim/config.yaml")
        prepare_ensemble_configurable_parameters(
            ens_config_path="tests/pre_merge/models/ensemble/dummy_ens_config.yaml", config=config
        )
        config.ensemble.post_processing.normalization = "final"

        callbacks = get_ensemble_callbacks(config, (0, 0))

        for callback in callbacks:
            assert not isinstance(callback, MinMaxNormalizationCallback)
