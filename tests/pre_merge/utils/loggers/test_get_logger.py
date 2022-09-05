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


def test_get_experiment_logger():
    """Test whether the right logger is returned."""

    config = OmegaConf.create(
        {
            "project": {"logger": None, "path": "/tmp"},
            "dataset": {"name": "dummy", "category": "cat1"},
            "model": {"name": "DummyModel"},
        }
    )

    with patch("pytorch_lightning.loggers.wandb.wandb"):

        # get no logger
        logger = get_experiment_logger(config=config)
        assert isinstance(logger, bool)
        config.project.logger = False
        logger = get_experiment_logger(config=config)
        assert isinstance(logger, bool)

        # get tensorboard
        config.project.logger = "tensorboard"
        logger = get_experiment_logger(config=config)
        assert isinstance(logger[0], AnomalibTensorBoardLogger)

        # get wandb logger
        config.project.logger = "wandb"
        logger = get_experiment_logger(config=config)
        assert isinstance(logger[0], AnomalibWandbLogger)

        # get comet logger
        config.project.logger = "comet"
        logger = get_experiment_logger(config=config)
        assert isinstance(logger[0], AnomalibCometLogger)

        # get csv logger.
        config.project.logger = "csv"
        logger = get_experiment_logger(config=config)
        assert isinstance(logger[0], CSVLogger)

        # get multiple loggers
        config.project.logger = ["tensorboard", "wandb", "csv", "comet"]
        logger = get_experiment_logger(config=config)
        assert isinstance(logger[0], AnomalibTensorBoardLogger)
        assert isinstance(logger[1], AnomalibWandbLogger)
        assert isinstance(logger[2], CSVLogger)
        assert isinstance(logger[3], AnomalibCometLogger)

        # raise unknown
        with pytest.raises(UnknownLogger):
            config.project.logger = "randomlogger"
            logger = get_experiment_logger(config=config)
