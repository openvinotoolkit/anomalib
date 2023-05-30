import glob
import os
import tempfile
from pathlib import Path

import pytest
from omegaconf.omegaconf import OmegaConf

from anomalib.trainer import AnomalibTrainer
from anomalib.utils.loggers import get_experiment_logger

from .dummy_lightning_model import DummyModule


def get_dummy_module(config):
    return DummyModule(config)


@pytest.mark.parametrize("task", ["classification", "detection", "segmentation"])
def test_add_images(task):
    """Tests if tensorboard logs are generated."""
    with tempfile.TemporaryDirectory() as dir_loc:
        config = OmegaConf.create(
            {
                "model": {"threshold": {"image_default": 0.5, "pixel_default": 0.5, "adaptive": True}},
                "logging": {
                    "loggers": [
                        {
                            "class_path": "anomalib.utils.loggers.FileSystemLogger",
                            "init_args": {"save_dir": dir_loc, "name": "logs"},
                        },
                        {
                            "class_path": "anomalib.utils.loggers.AnomalibTensorBoardLogger",
                            "init_args": {"save_dir": dir_loc, "name": "tensorboard_logs"},
                        },
                    ]
                },
                "visualization": {"log_images": True},
                "metrics": {},
            }
        )
        loggers = get_experiment_logger(config)
        model = get_dummy_module(config)
        trainer = AnomalibTrainer(
            logger=loggers, enable_checkpointing=False, default_root_dir=dir_loc, task_type=task, **config.visualization
        )
        trainer.test(model=model)
        # test if images are logged
        if len(list((Path(dir_loc) / "logs").glob("**/*.png"))) != 1:
            raise Exception("Failed to save to local path")

        # test if tensorboard logs are created
        if len(glob.glob(os.path.join(dir_loc, "tensorboard_logs", "version_*"))) == 0:
            raise Exception("Failed to save to tensorboard")
