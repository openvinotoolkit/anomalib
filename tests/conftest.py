"""Tests."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch


@pytest.fixture
def device() -> torch.device:
    """Returns the device to run the tests on."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
