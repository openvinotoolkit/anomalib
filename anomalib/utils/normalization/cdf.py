"""Tools for CDF normalization."""

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

from typing import Optional, Union

import numpy as np
import torch
from scipy.stats import norm
from torch import Tensor
from torch.distributions import Normal


def standardize(
    targets: Union[np.ndarray, Tensor],
    mean: Union[np.ndarray, Tensor, float],
    std: Union[np.ndarray, Tensor, float],
    center_at: Optional[float] = None,
) -> Union[np.ndarray, Tensor]:
    """Standardize the targets to the z-domain."""
    if isinstance(targets, np.ndarray):
        targets = np.log(targets)
    elif isinstance(targets, Tensor):
        targets = torch.log(targets)
    else:
        raise ValueError(f"Targets must be either Tensor or Numpy array. Received {type(targets)}")
    standardized = (targets - mean) / std
    if center_at:
        standardized -= (center_at - mean) / std
    return standardized


def normalize(
    targets: Union[np.ndarray, Tensor], threshold: Union[np.ndarray, Tensor, float]
) -> Union[np.ndarray, Tensor]:
    """Normalize the targets by using the cumulative density function."""
    if isinstance(targets, Tensor):
        return normalize_torch(targets, threshold)
    if isinstance(targets, np.ndarray):
        return normalize_numpy(targets, threshold)
    raise ValueError(f"Targets must be either Tensor or Numpy array. Received {type(targets)}")


def normalize_torch(targets: Tensor, threshold: Tensor) -> Tensor:
    """Normalize the targets by using the cumulative density function, PyTorch version."""
    device = targets.device
    image_threshold = threshold.cpu()

    dist = Normal(torch.Tensor([0]), torch.Tensor([1]))
    normalized = dist.cdf(targets.cpu() - image_threshold).to(device)
    return normalized


def normalize_numpy(targets: np.ndarray, threshold: Union[np.ndarray, float]) -> np.ndarray:
    """Normalize the targets by using the cumulative density function, Numpy version."""
    return norm.cdf(targets - threshold)
