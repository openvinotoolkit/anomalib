"""
Test Models - STFPM
"""

import random
import tempfile

import pytest
from pytorch_lightning import Trainer

from anomalib.config.config import get_configurable_parameters, update_config_for_nncf
from anomalib.core.callbacks import get_callbacks
from anomalib.core.callbacks.visualizer_callback import VisualizerCallback
from anomalib.datasets import get_datamodule
from anomalib.models import get_model
from tests.helpers.dataset import TestDataset, get_dataset_path


@pytest.fixture(autouse=True)
def category() -> str:
    """
    PyTest fixture to randomly return an MVTec category.

    Returns:
        str: Random MVTec category to train/test.
    """
    categories = [
        "bottle",
        "cable",
        "capsule",
        "carpet",
        "grid",
        "hazelnut",
        "leather",
        "metal_nut",
        "pill",
        "screw",
        "tile",
        "toothbrush",
        "transistor",
        "wood",
        "zipper",
    ]

    category = random.choice(categories)
    return category


@pytest.mark.parametrize(
    "model_name,nncf",
    [
        ("padim", False),
        ("dfkde", False),
        ("dfm", False),
        ("stfpm", False),
        ("stfpm", True),
        ("patchcore", False),
    ],
)
@pytest.mark.flaky(max_runs=3)
@TestDataset(num_train=200, num_test=10, path=get_dataset_path(), use_mvtec=True)
def test_model(category, model_name, nncf, path="./datasets/MVTec"):
    """
    Test Model Training and Test Pipeline.

    Args:
        category (str): Category to test on
        model_name (str): Name of the model
    """
    config = get_configurable_parameters(model_name=model_name)
    config.project.seed = 1234
    config.dataset.category = category
    config.dataset.path = path
    with tempfile.TemporaryDirectory() as temporary_directory:
        config.project.path = temporary_directory
        config.model.weight_file = "weights/model.ckpt"  # add model weights to the config

        if nncf:
            config.optimization.nncf.apply = True
            config = update_config_for_nncf(config)
            config.init_weights = None
            # reassign project path as config is updated in `update_config_for_nncf`
            config.project.path = temporary_directory

        datamodule = get_datamodule(config)
        model = get_model(config)

        callbacks = get_callbacks(config)
        if hasattr(model, "callbacks"):
            callbacks.extend(model.callbacks)

        for index, callback in enumerate(callbacks):
            if isinstance(callback, VisualizerCallback):
                callbacks.pop(index)
                break

        # Train the model.
        trainer = Trainer(callbacks=callbacks, **config.trainer)
        trainer.fit(model=model, datamodule=datamodule)

        # Test the model.
        trainer.test(model=model, datamodule=datamodule)

        assert model.results.performance["image_roc_auc"] >= 0.6

        if config.dataset.task == "segmentation":
            assert model.results.performance["pixel_roc_auc"] >= 0.6

        # Test loading the model
        # TODO add support for dfm once pca is available
        if model_name != "dfm":
            loaded_model = get_model(config)  # get new model

            callbacks = get_callbacks(config)
            if hasattr(model, "callbacks"):
                callbacks.extend(model.callbacks)

            for index, callback in enumerate(callbacks):
                # Remove visualizer callback as saving results takes time
                if isinstance(callback, VisualizerCallback):
                    callbacks.pop(index)
                    break

            # create new trainer object with LoadModel callback (assumes it is present)
            trainer = Trainer(callbacks=callbacks, **config.trainer)
            # Assumes the new model has LoadModel callback and the old one had ModelCheckpoint callback
            trainer.test(model=loaded_model, datamodule=datamodule)
            if config.dataset.task == "segmentation":
                assert (
                    model.results.performance["pixel_roc_auc"] == loaded_model.results.performance["pixel_roc_auc"]
                ), "Loaded model does not give same performance"
