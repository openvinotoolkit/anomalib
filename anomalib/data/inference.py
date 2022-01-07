"""Test Inference Dataset."""

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

from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import albumentations as A
from torch.utils.data.dataset import Dataset
from torchvision.datasets.folder import IMG_EXTENSIONS

from anomalib.data.transforms import PreProcessor
from anomalib.data.utils import read_image


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

    # If `path` is an image path.
    if path.is_file() and path.suffix in IMG_EXTENSIONS:
        image_filenames = [str(path)]

    # If it is a path to image folder
    if path.is_dir():
        image_filenames = [str(p) for p in path.glob("**/*") if p.suffix in IMG_EXTENSIONS]

    if len(image_filenames) == 0:
        raise ValueError(f"Found 0 images in {path}")

    return image_filenames


class InferenceDataset(Dataset):
    """Inference Dataset to perform prediction."""

    def __init__(
        self,
        path: Union[str, Path],
        image_size: Optional[Union[int, Tuple[int, int]]] = None,
        transform_config: Optional[Union[str, A.Compose]] = None,
    ) -> None:
        super().__init__()

        self.image_filenames = get_image_filenames(path)
        self.pre_process = PreProcessor(config=transform_config, image_size=image_size)

    def __len__(self) -> int:
        """Get the number of images in the given path."""
        return len(self.image_filenames)

    def __getitem__(self, index: int) -> Any:
        """Get the image based on the `index`."""
        image_filename = self.image_filenames[index]
        image = read_image(path=image_filename)
        pre_processed = self.pre_process(image=image)

        return pre_processed
