"""Tests to ascertain requested logger."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

patch("pytorch_lightning.utilities.imports._package_available", False)
patch("pytorch_lightning.loggers.wandb.WandbLogger")

import pytest
from omegaconf import OmegaConf
from pytorch_lightning.loggers import CSVLogger

from anomalib.utils.loggers import (
    AnomalibCometLogger,
    AnomalibTensorBoardLogger,
    AnomalibWandbLogger,
    UnknownLogger,
    get_experiment_logger,
)


@pytest.fixture(scope="session")
def temp_dir(tmp_path_factory):
    """Create a temporary directory for the test."""
    return tmp_path_factory.mktemp("test_logs")


def test_get_experiment_logger(temp_dir):
    """Test whether the right logger is returned."""

    config = OmegaConf.create(
        {
            "logging": {"logger": None, "log_graph": False},
            "dataset": {"name": "dummy", "category": "cat1"},
            "model": {"name": "DummyModel"},
            "trainer": {"default_root_dir": temp_dir},
        }
    )

    with patch("pytorch_lightning.loggers.wandb.wandb"):

        # get no logger
        logger = get_experiment_logger(config=config)
        assert isinstance(logger, bool)
        config.logging.logger = False
        logger = get_experiment_logger(config=config)
        assert isinstance(logger, bool)

        # get tensorboard
        config.logging.logger = "tensorboard"
        logger = get_experiment_logger(config=config)
        assert isinstance(logger[0], AnomalibTensorBoardLogger)

        # get wandb logger
        config.logging.logger = "wandb"
        logger = get_experiment_logger(config=config)
        assert isinstance(logger[0], AnomalibWandbLogger)

        # get comet logger
        config.logging.logger = "comet"
        logger = get_experiment_logger(config=config)
        assert isinstance(logger[0], AnomalibCometLogger)

        # get csv logger.
        config.logging.logger = "csv"
        logger = get_experiment_logger(config=config)
        assert isinstance(logger[0], CSVLogger)

        # get multiple loggers
        config.logging.logger = ["tensorboard", "wandb", "csv", "comet"]
        logger = get_experiment_logger(config=config)
        assert isinstance(logger[0], AnomalibTensorBoardLogger)
        assert isinstance(logger[1], AnomalibWandbLogger)
        assert isinstance(logger[2], CSVLogger)
        assert isinstance(logger[3], AnomalibCometLogger)

        # raise unknown
        with pytest.raises(UnknownLogger):
            config.logging.logger = "randomlogger"
            logger = get_experiment_logger(config=config)
