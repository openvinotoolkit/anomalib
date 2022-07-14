import glob
import os
import tempfile
from unittest import mock

import pytest
import pytorch_lightning as pl
from omegaconf.omegaconf import OmegaConf

from anomalib.utils.loggers import AnomalibTensorBoardLogger

from .dummy_lightning_model import DummyDataModule, DummyModule


def get_dummy_module(config):
    return DummyModule(config)


def get_dummy_logger(config, tempdir):
    logger = AnomalibTensorBoardLogger(name=f"tensorboard_logs", save_dir=tempdir)
    return logger


@pytest.mark.parametrize("dataset", ["segmentation"])
def test_add_images(dataset):
    """Tests if tensorboard logs are generated."""
    with tempfile.TemporaryDirectory() as dir_loc:
        config = OmegaConf.create(
            {
                "dataset": {"task": dataset},
                "model": {"threshold": {"image_default": 0.5, "pixel_default": 0.5, "adaptive": True}},
                "project": {"path": dir_loc},
                "logging": {"logger": ["tensorboard"]},
                "visualization": {"log_images": True, "save_images": True},
                "metrics": {},
            }
        )
        logger = get_dummy_logger(config, dir_loc)
        model = get_dummy_module(config)
        trainer = pl.Trainer(
            callbacks=model.callbacks, logger=logger, checkpoint_callback=False, default_root_dir=config.project.path
        )
        trainer.test(model=model, datamodule=DummyDataModule())
        # test if images are logged
        if len(glob.glob(os.path.join(dir_loc, "images", "*.jpg"))) != 1:
            raise Exception("Failed to save to local path")

        # test if tensorboard logs are created
        if len(glob.glob(os.path.join(dir_loc, "tensorboard_logs", "version_*"))) == 0:
            raise Exception("Failed to save to tensorboard")
