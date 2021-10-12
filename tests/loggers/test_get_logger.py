import pytest
from omegaconf import OmegaConf

from anomalib.loggers import (
    AnomalibTensorBoardLogger,
    SigoptLogger,
    UnknownLogger,
    get_logger,
)


def test_get_logger():
    """Test whether the right logger is returned"""

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

    # get sigopt logger
    config.project.logger = True
    logger = get_logger(config=config)
    assert isinstance(logger, SigoptLogger)
    config.project.logger = "sigopt"
    logger = get_logger(config=config)
    assert isinstance(logger, SigoptLogger)

    # get tensorboard
    config.project.logger = "tensorboard"
    logger = get_logger(config=config)
    assert isinstance(logger, AnomalibTensorBoardLogger)

    # raise unknown
    with pytest.raises(UnknownLogger):
        config.project.logger = "randomlogger"
        logger = get_logger(config=config)
