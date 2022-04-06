"""Tests to ascertain requested logger."""

# Copyright (C) 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import pytest
from omegaconf import OmegaConf
from pytorch_lightning.loggers import CSVLogger

from anomalib.utils.loggers import (
    AnomalibTensorBoardLogger,
    AnomalibWandbLogger,
    UnknownLogger,
    get_logger,
)


def test_get_logger():
    """Test whether the right logger is returned."""

    config = OmegaConf.create(
        {
            "project": {"logger": None, "path": "/tmp"},
            "dataset": {"name": "dummy", "category": "cat1"},
            "model": {"name": "DummyModel"},
        }
    )

    # get no logger
    logger = get_logger(config=config)
    assert isinstance(logger, bool)
    config.project.logger = False
    logger = get_logger(config=config)
    assert isinstance(logger, bool)

    # get tensorboard
    config.project.logger = "tensorboard"
    logger = get_logger(config=config)
    assert isinstance(logger[0], AnomalibTensorBoardLogger)

    # get wandb logger
    config.project.logger = "wandb"
    logger = get_logger(config=config)
    assert isinstance(logger[0], AnomalibWandbLogger)

    # get csv logger.
    config.project.logger = "csv"
    logger = get_logger(config=config)
    assert isinstance(logger[0], CSVLogger)

    # get multiple loggers
    config.project.logger = ["tensorboard", "wandb", "csv"]
    logger = get_logger(config=config)
    assert isinstance(logger[0], AnomalibTensorBoardLogger)
    assert isinstance(logger[1], AnomalibWandbLogger)
    assert isinstance(logger[2], CSVLogger)

    # raise unknown
    with pytest.raises(UnknownLogger):
        config.project.logger = "randomlogger"
        logger = get_logger(config=config)
