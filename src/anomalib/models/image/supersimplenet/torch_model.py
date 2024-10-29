"""PyTorch model for the SuperSimpleNet model implementation."""

from torch import nn


# Original Code
# Copyright (c) 2024 Blaž Rolih
# https://github.com/blaz-r/SuperSimpleNet.
# SPDX-License-Identifier: MIT
#
# Modified
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


class SuperSimpleNet(nn.Module):
    """SuperSimpleNet Pytorch model.

    It consists of feature extractor, feature adaptor, anomaly generation mechanism and segmentation-detection module.

    """

    def __init__(self):
        super().__init__()
