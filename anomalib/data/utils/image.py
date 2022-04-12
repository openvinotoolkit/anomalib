"""Image Utils."""

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

import math
from pathlib import Path
from typing import List, Union

import cv2
import numpy as np
import torch.nn.functional as F
from torch import Tensor
from torchvision.datasets.folder import IMG_EXTENSIONS


def get_image_filenames(path: Union[str, Path]) -> List[str]:
    """Get image filenames.

    Args:
        path (Union[str, Path]): Path to image or image-folder.

    Returns:
        List[str]: List of image filenames

    """
    image_filenames: List[str]

    if isinstance(path, str):
        path = Path(path)

    if path.is_file() and path.suffix in IMG_EXTENSIONS:
        image_filenames = [str(path)]

    if path.is_dir():
        image_filenames = [str(p) for p in path.glob("**/*") if p.suffix in IMG_EXTENSIONS]

    if len(image_filenames) == 0:
        raise ValueError(f"Found 0 images in {path}")

    return image_filenames


def read_image(path: Union[str, Path]) -> np.ndarray:
    """Read image from disk in RGB format.

    Args:
        path (str, Path): path to the image file

    Example:
        >>> image = read_image("test_image.jpg")

    Returns:
        image as numpy array
    """
    path = path if isinstance(path, str) else str(path)
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def pad_nextpow2(batch: Tensor) -> Tensor:
    """Compute required padding from input size and return padded images.

    Finds the largest dimension and computes a square image of dimensions that are of the power of 2.
    In case the image dimension is odd, it returns the image with an extra padding on one side.

    Args:
        batch (Tensor): Input images

    Returns:
        batch: Padded batch
    """
    # find the largest dimension
    l_dim = 2 ** math.ceil(math.log(max(*batch.shape[-2:]), 2))
    padding_w = [math.ceil((l_dim - batch.shape[-2]) / 2), math.floor((l_dim - batch.shape[-2]) / 2)]
    padding_h = [math.ceil((l_dim - batch.shape[-1]) / 2), math.floor((l_dim - batch.shape[-1]) / 2)]
    padded_batch = F.pad(batch, pad=[*padding_h, *padding_w])
    return padded_batch
