"""Fixtures that are used in ensemble testing"""

import pytest

from anomalib.config import get_configurable_parameters
from anomalib.models.ensemble import EnsembleTiler, get_ensemble_datamodule, prepare_ensemble_configurable_parameters


@pytest.fixture(scope="module")
def get_config():
    config = get_configurable_parameters(config_path="tests/pre_merge/models/ensemble/dummy_padim_config.yaml")
    prepare_ensemble_configurable_parameters(
        ens_config_path="tests/pre_merge/models/ensemble/dummy_ens_config.yaml", config=config
    )
    return config


@pytest.fixture(scope="module")
def get_datamodule():
    def get_datamodule(config, task):
        tiler = EnsembleTiler(config)
        config.dataset.task = task

        datamodule = get_ensemble_datamodule(config, tiler)

        datamodule.prepare_data()
        datamodule.setup()

        return datamodule

    return get_datamodule
