"""Utilities for parsing model configuration."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .config import (
    get_configurable_parameters,
    update_input_size_config,
    update_nncf_config,
)

__all__ = ["get_configurable_parameters", "update_nncf_config", "update_input_size_config"]
