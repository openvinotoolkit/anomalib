"""Inference Dataset."""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any

from torch.utils.data.dataset import Dataset
from torchvision.transforms.v2 import Transform

from anomalib.data.utils import get_image_filenames, read_image


class PredictDataset(Dataset):
    """Inference Dataset to perform prediction.

    Args:
        path (str | Path): Path to an image or image-folder.
        transform (A.Compose | None, optional): Transform object describing the transforms that are
            applied to the inputs.
        image_size (int | tuple[int, int] | None, optional): Target image size
            to resize the original image. Defaults to None.
    """

    def __init__(
        self,
        path: str | Path,
        transform: Transform | None = None,
        image_size: int | tuple[int, int] = (256, 256),
    ) -> None:
        super().__init__()

        self.image_filenames = get_image_filenames(path)
        self.transform = transform
        self.image_size = image_size

    def __len__(self) -> int:
        """Get the number of images in the given path."""
        return len(self.image_filenames)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Get the image based on the `index`."""
        image_filename = self.image_filenames[index]
        image = read_image(image_filename, as_tensor=True)
        if self.transform:
            image = self.transform(image)
        pre_processed = {"image": image}
        pre_processed["image_path"] = str(image_filename)

        return pre_processed
