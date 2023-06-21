"""Inferencers for Torch and OpenVINO."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .openvino_inferencer import OpenVINOInferencer
from .torch_inferencer import TorchInferencer

__all__ = ["OpenVINOInferencer", "TorchInferencer"]
