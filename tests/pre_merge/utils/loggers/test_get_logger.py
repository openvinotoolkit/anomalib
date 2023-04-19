"""Tests to ascertain requested logger."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

try:
    from wandb import init  # noqa: F401

    wandb_installed = True
except ImportError:
    wandb_installed = False

if wandb_installed:
    with patch("wandb.init"):
        from pytorch_lightning.loggers import CSVLogger

        from anomalib.utils.loggers import (
            AnomalibCometLogger,
            AnomalibTensorBoardLogger,
            AnomalibWandbLogger,
            UnknownLogger,
            get_experiment_logger,
        )
else:
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

    with patch("anomalib.utils.loggers.wandb.AnomalibWandbLogger.experiment"), patch(
        "pytorch_lightning.loggers.wandb.wandb"
    ), patch("pytorch_lightning.loggers.comet.comet_ml"):
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
