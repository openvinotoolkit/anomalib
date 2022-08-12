"""Statistical functions."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .kde import GaussianKDE
from .multi_variate_gaussian import MultiVariateGaussian

__all__ = ["GaussianKDE", "MultiVariateGaussian"]
