"""Inference Dataset."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any, Optional, Tuple, Union

import albumentations as A
from torch.utils.data.dataset import Dataset

from anomalib.data.utils import get_image_filenames, get_transforms, read_image


class InferenceDataset(Dataset):
    """Inference Dataset to perform prediction.

    Args:
        path (Union[str, Path]): Path to an image or image-folder.
        transform (Optional[A.Compose], optional): Albumentations Compose object describing the transforms that are
            applied to the inputs.
        image_size (Optional[Union[int, Tuple[int, int]]], optional): Target image size
            to resize the original image. Defaults to None.
    """

    def __init__(
        self,
        path: Union[str, Path],
        transform: Optional[A.Compose] = None,
        image_size: Optional[Union[int, Tuple[int, int]]] = None,
    ) -> None:
        super().__init__()

        self.image_filenames = get_image_filenames(path)

        if transform is None:
            self.transform = get_transforms(image_size=image_size)
        else:
            self.transform = transform

    def __len__(self) -> int:
        """Get the number of images in the given path."""
        return len(self.image_filenames)

    def __getitem__(self, index: int) -> Any:
        """Get the image based on the `index`."""
        image_filename = self.image_filenames[index]
        image = read_image(path=image_filename)
        pre_processed = self.transform(image=image)
        pre_processed["image_path"] = str(image_filename)

        return pre_processed
