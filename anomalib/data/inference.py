"""Inference Dataset."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any, Optional, Tuple, Union

import albumentations as A
from torch.utils.data.dataset import Dataset

from anomalib.data.utils import get_image_filenames, read_image
from anomalib.pre_processing import PreProcessor


class InferenceDataset(Dataset):
    """Inference Dataset to perform prediction."""

    def __init__(
        self,
        path: Union[str, Path],
        pre_process: Optional[PreProcessor] = None,
        image_size: Optional[Union[int, Tuple[int, int]]] = None,
        transform_config: Optional[Union[str, A.Compose]] = None,
    ) -> None:
        """Inference Dataset to perform prediction.

        Args:
            path (Union[str, Path]): Path to an image or image-folder.
            pre_process (Optional[PreProcessor], optional): Pre-Processing transforms to
                pre-process the input dataset. Defaults to None.
            image_size (Optional[Union[int, Tuple[int, int]]], optional): Target image size
                to resize the original image. Defaults to None.
            transform_config (Optional[Union[str, A.Compose]], optional): Configuration file
                parse the albumentation transforms. Defaults to None.
        """
        super().__init__()

        self.image_filenames = get_image_filenames(path)

        if pre_process is None:
            self.pre_process = PreProcessor(transform_config, image_size)
        else:
            self.pre_process = pre_process

    def __len__(self) -> int:
        """Get the number of images in the given path."""
        return len(self.image_filenames)

    def __getitem__(self, index: int) -> Any:
        """Get the image based on the `index`."""
        image_filename = self.image_filenames[index]
        image = read_image(path=image_filename)
        pre_processed = self.pre_process(image=image)
        pre_processed["image_path"] = str(image_filename)

        return pre_processed
