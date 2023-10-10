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
        from lightning.pytorch.loggers import CSVLogger

        from anomalib.utils.loggers import (
            AnomalibCometLogger,
            AnomalibTensorBoardLogger,
            AnomalibWandbLogger,
            UnknownLogger,
            get_experiment_logger,
        )
else:
    from lightning.pytorch.loggers import CSVLogger

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
            "project": {"path": "/tmp"},
            "data": {"class_path": "dummy", "init_args": {"category": "cat1"}},
            "model": {"class_path": "DummyModel"},
            "trainer": {"logger": None, "default_root_dir": "/tmp"},
        }
    )

    with patch("anomalib.utils.loggers.wandb.AnomalibWandbLogger.experiment"), patch(
        "lightning.pytorch.loggers.wandb.wandb"
    ), patch("lightning.pytorch.loggers.comet.comet_ml"):
        # get no logger
        logger = get_experiment_logger(config=config)
        assert isinstance(logger, bool)
        config.trainer.logger = False
        logger = get_experiment_logger(config=config)
        assert isinstance(logger, bool)

        # get tensorboard
        config.trainer.logger = "tensorboard"
        logger = get_experiment_logger(config=config)
        assert isinstance(logger[0], AnomalibTensorBoardLogger)

        # get wandb logger
        config.trainer.logger = "wandb"
        logger = get_experiment_logger(config=config)
        assert isinstance(logger[0], AnomalibWandbLogger)

        # get comet logger
        config.trainer.logger = "comet"
        logger = get_experiment_logger(config=config)
        assert isinstance(logger[0], AnomalibCometLogger)

        # get csv logger.
        config.trainer.logger = "csv"
        logger = get_experiment_logger(config=config)
        assert isinstance(logger[0], CSVLogger)

        # get multiple loggers
        config.trainer.logger = ["tensorboard", "wandb", "csv", "comet"]
        logger = get_experiment_logger(config=config)
        assert isinstance(logger[0], AnomalibTensorBoardLogger)
        assert isinstance(logger[1], AnomalibWandbLogger)
        assert isinstance(logger[2], CSVLogger)
        assert isinstance(logger[3], AnomalibCometLogger)

        # raise unknown
        with pytest.raises(UnknownLogger):
            config.trainer.logger = "randomlogger"
            logger = get_experiment_logger(config=config)
