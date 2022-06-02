"""compute Anomaly map."""

# Copyright (C) 2022 Intel Corporation
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

from typing import List, Tuple, Union

import torch
import torch.nn.functional as F
from kornia.filters import gaussian_blur2d
from omegaconf import ListConfig
from torch import Tensor


class AnomalyMapGenerator:
    """Generate Anomaly Heatmap.

    Args:
        image_size (Union[ListConfig, Tuple]): Size of original image used for upscaling the anomaly map.
        sigma (int): Standard deviation of the gaussian kernel used to smooth anomaly map.
    """

    def __init__(self, image_size: Union[ListConfig, Tuple], sigma: int = 4):
        self.image_size = image_size if isinstance(image_size, tuple) else tuple(image_size)
        self.sigma = sigma
        self.kernel_size = 2 * int(4.0 * sigma + 0.5) + 1

    def __call__(self, fs_list: List[Tensor], ft_list: List[Tensor], amap_mode: str = "mul") -> Tensor:
        """Computes anomaly map given encoder and decoder features.

        Args:
            fs_list (List[Tensor]): List of encoder features
            ft_list (List[Tensor]): List of decoder features
            amap_mode (str, optional): Operation used to generate anomaly map. Options are `add` and `mul`.
                Defaults to "mul".

        Raises:
            ValueError: _description_

        Returns:
            Tensor: _description_
        """
        if amap_mode == "mul":
            anomaly_map = torch.ones([fs_list[0].shape[0], 1, *self.image_size], device=fs_list[0].device)  # b c h w
        elif amap_mode == "add":
            anomaly_map = torch.zeros([fs_list[0].shape[0], 1, *self.image_size], device=fs_list[0].device)
        else:
            raise ValueError(f"Found amap_mode {amap_mode}. Only mul and add are supported.")
        for i in range(len(ft_list)):
            fs = fs_list[i]
            ft = ft_list[i]
            a_map = 1 - F.cosine_similarity(fs, ft)
            a_map = torch.unsqueeze(a_map, dim=1)
            a_map = F.interpolate(a_map, size=self.image_size, mode="bilinear", align_corners=True)
            if amap_mode == "mul":
                anomaly_map *= a_map
            elif amap_mode == "add":
                anomaly_map += a_map
            else:
                raise ValueError(f"Operation {amap_mode} not supported. Only ``add`` and ``mul`` are supported")

        anomaly_map = gaussian_blur2d(
            anomaly_map, kernel_size=(self.kernel_size, self.kernel_size), sigma=(self.sigma, self.sigma)
        )

        return anomaly_map
