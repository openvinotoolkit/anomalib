"""Tests to ascertain requested logger."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

from anomalib.utils.loggers import (
    AnomalibCometLogger,
    AnomalibTensorBoardLogger,
    AnomalibWandbLogger,
    FileSystemLogger,
    UnknownLogger,
    get_experiment_logger,
)


def test_get_experiment_logger(tmpdir):
    """Test whether the right logger is returned."""
    tmpdir = str(tmpdir)
    config = OmegaConf.create(
        {
            "logging": {"loggers": None},
            "dataset": {"name": "dummy", "category": "cat1"},
            "model": {"name": "DummyModel"},
        }
    )

    with patch("anomalib.utils.loggers.wandb.AnomalibWandbLogger.experiment"), patch(
        "pytorch_lightning.loggers.wandb.wandb"
    ), patch("pytorch_lightning.loggers.comet.comet_ml"), patch(
        "anomalib.utils.loggers.comet.AnomalibCometLogger.experiment"
    ):
        # get no logger
        logger = get_experiment_logger(config=config)
        assert isinstance(logger, bool)
        config.logging.loggers = None
        logger = get_experiment_logger(config=config)
        assert isinstance(logger, bool)

        # get tensorboard
        config.logging.loggers = {
            "class_path": "anomalib.utils.loggers.AnomalibTensorBoardLogger",
            "init_args": {"save_dir": tmpdir},
        }
        logger = get_experiment_logger(config=config)
        assert isinstance(logger[0], AnomalibTensorBoardLogger)

        # get wandb logger
        config.logging.loggers = {
            "class_path": "anomalib.utils.loggers.AnomalibWandbLogger",
            "init_args": {"save_dir": tmpdir},
        }
        logger = get_experiment_logger(config=config)
        assert isinstance(logger[0], AnomalibWandbLogger)

        # get comet logger
        config.logging.loggers = {
            "class_path": "anomalib.utils.loggers.AnomalibCometLogger",
            "init_args": {"save_dir": tmpdir},
        }
        logger = get_experiment_logger(config=config)
        assert isinstance(logger[0], AnomalibCometLogger)

        # get csv logger.
        config.logging.loggers = {
            "class_path": "anomalib.utils.loggers.FileSystemLogger",
            "init_args": {"save_dir": tmpdir},
        }
        logger = get_experiment_logger(config=config)
        assert isinstance(logger[0], FileSystemLogger)

        # get multiple loggers
        config.logging.loggers = [
            {"class_path": "anomalib.utils.loggers.AnomalibTensorBoardLogger", "init_args": {"save_dir": tmpdir}},
            {"class_path": "anomalib.utils.loggers.AnomalibWandbLogger", "init_args": {"save_dir": tmpdir}},
            {"class_path": "anomalib.utils.loggers.FileSystemLogger", "init_args": {"save_dir": tmpdir}},
            {"class_path": "anomalib.utils.loggers.AnomalibCometLogger", "init_args": {"save_dir": tmpdir}},
        ]
        logger = get_experiment_logger(config=config)
        assert isinstance(logger[0], AnomalibTensorBoardLogger)
        assert isinstance(logger[1], AnomalibWandbLogger)
        assert isinstance(logger[2], FileSystemLogger)
        assert isinstance(logger[3], AnomalibCometLogger)

        # raise unknown
        with pytest.raises(UnknownLogger):
            config.logging.loggers = {"class_path": "anomalib.utils.loggers.RandomLogger"}
            logger = get_experiment_logger(config=config)
