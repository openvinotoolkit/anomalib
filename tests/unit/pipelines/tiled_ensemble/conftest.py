"""Fixtures that are used in tiled ensemble testing"""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import yaml
import pytest

from anomalib.pipelines.tiled_ensemble.components.utils.helper_functions import get_ensemble_tiler, get_ensemble_model


@pytest.fixture(scope="module")
def get_ensemble_config():
    with Path("tests/unit/pipelines/tiled_ensemble/dummy_config.yaml").open(encoding="utf-8") as file:
        return yaml.safe_load(file)


@pytest.fixture(scope="module")
def get_tiler(get_ensemble_config):
    config = get_ensemble_config

    return get_ensemble_tiler(config["tiling"], config["data"])

@pytest.fixture(scope="module")
def get_model(get_ensemble_config, get_tiler):
    config = get_ensemble_config
    tiler = get_tiler

    return get_ensemble_model(config["TrainModels"]["model"], tiler)