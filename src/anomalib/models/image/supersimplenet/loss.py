"""Loss function for the SuperSimpleNet model implementation."""

from torch import nn

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


class SSNLoss(nn.Module):
    """SuperSimpleNet loss function."""

    def __init__(self):
        super().__init__()
