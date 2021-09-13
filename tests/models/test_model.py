"""
Test Models - STFPM
"""

import random
import tempfile

import pytest
from pytorch_lightning import Trainer

from anomalib.config.config import get_configurable_parameters, update_config_for_nncf
from anomalib.datasets import get_datamodule
from anomalib.models import get_model


@pytest.fixture(autouse=True)
def mvtec_dataset_category() -> str:
    """
    mvtec_dataset_category
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
        ("padim", True),
        ("dfkde", False),
        ("stfpm", False),
        ("stfpm", True),
        ("patchcore", False),
    ],
)
@pytest.mark.flaky(reruns=2)
def test_model(mvtec_dataset_category, model_name, nncf):
    """
    Test Model Training and Test Pipeline.

    Args:
        mvtec_dataset_category ([type]): [description]
        model_name ([type]): [description]
    """
    config = get_configurable_parameters(model_name=model_name)
    config.project.seed = 1234
    config.dataset.category = mvtec_dataset_category

    if nncf:
        config.optimization.nncf.apply = True
        config = update_config_for_nncf(config)
        config.init_weights = None

    datamodule = get_datamodule(config)
    model = get_model(config)

    # Train the model.
    trainer = Trainer(callbacks=model.callbacks, **config.trainer)
    trainer.fit(model=model, datamodule=datamodule)

    # Test the model.
    with tempfile.TemporaryDirectory() as temporary_directory:
        config.project.path = temporary_directory
        trainer.test(model=model, datamodule=datamodule)

    assert model.image_roc_auc >= 0.6

    if model_name != "dfkde":
        assert model.pixel_roc_auc >= 0.6
