import glob
import os
import tempfile
from pathlib import Path

import lightning.pytorch as pl
import pytest
from omegaconf.omegaconf import OmegaConf

from anomalib.engine import Engine
from anomalib.utils.loggers import AnomalibTensorBoardLogger
from tests.helpers.dummy import DummyDataModule

from .dummy_lightning_model import DummyModule


def get_dummy_module(config):
    return DummyModule(config)


def get_dummy_logger(config, tempdir):
    logger = AnomalibTensorBoardLogger(name="tensorboard_logs", save_dir=tempdir)
    return logger


@pytest.mark.parametrize("task", ["classification", "detection", "segmentation"])
def test_add_images(task):
    """Tests if tensorboard logs are generated."""
    with tempfile.TemporaryDirectory() as dir_loc:
        config = OmegaConf.create(
            {
                "dataset": {"task": task},
                "model": {"threshold": {"image_default": 0.5, "pixel_default": 0.5, "adaptive": True}},
                "project": {"path": dir_loc},
                "logging": {"logger": ["tensorboard"]},
                "visualization": {"log_images": True, "save_images": True},
                "metrics": {},
            }
        )
        logger = get_dummy_logger(config, dir_loc)
        model = get_dummy_module(config)
        engine = Engine(
            callbacks=model.callbacks, logger=logger, enable_checkpointing=False, default_root_dir=config.project.path
        )
        engine.test(model=model, datamodule=DummyDataModule())
        # test if images are logged
        if len(list(Path(dir_loc).glob("**/*.png"))) != 1:
            raise Exception("Failed to save to local path")

        # test if tensorboard logs are created
        if len(glob.glob(os.path.join(dir_loc, "tensorboard_logs", "version_*"))) == 0:
            raise Exception("Failed to save to tensorboard")
