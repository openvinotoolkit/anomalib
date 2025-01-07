"""Dataset for performing inference on images.

This module provides a dataset class for loading and preprocessing images for
inference in anomaly detection tasks.

Example:
    >>> from pathlib import Path
    >>> from anomalib.data import PredictDataset
    >>> dataset = PredictDataset(path="path/to/images")
    >>> item = dataset[0]
    >>> item.image.shape  # doctest: +SKIP
    torch.Size([3, 256, 256])
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from pathlib import Path

from torch.utils.data.dataset import Dataset
from torchvision.transforms.v2 import Transform

from anomalib.data import ImageBatch, ImageItem
from anomalib.data.utils import get_image_filenames, read_image


class PredictDataset(Dataset):
    """Dataset for performing inference on images.

    Args:
        path (str | Path): Path to an image or directory containing images.
        transform (Transform | None, optional): Transform object describing the
            transforms to be applied to the inputs. Defaults to ``None``.
        image_size (int | tuple[int, int], optional): Target size to which input
            images will be resized. If int, a square image of that size will be
            created. Defaults to ``(256, 256)``.

    Examples:
        >>> from pathlib import Path
        >>> dataset = PredictDataset(
        ...     path=Path("path/to/images"),
        ...     image_size=(224, 224),
        ... )
        >>> len(dataset)  # doctest: +SKIP
        10
        >>> item = dataset[0]  # doctest: +SKIP
        >>> item.image.shape  # doctest: +SKIP
        torch.Size([3, 224, 224])
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
        """Get number of images in dataset.

        Returns:
            int: Number of images in the dataset.
        """
        return len(self.image_filenames)

    def __getitem__(self, index: int) -> ImageItem:
        """Get image item at specified index.

        Args:
            index (int): Index of the image to retrieve.

        Returns:
            ImageItem: Object containing the loaded image and its metadata.
        """
        image_filename = self.image_filenames[index]
        image = read_image(image_filename, as_tensor=True)
        if self.transform:
            image = self.transform(image)

        return ImageItem(
            image=image,
            image_path=str(image_filename),
        )

    @property
    def collate_fn(self) -> Callable:
        """Get collate function for creating batches.

        Returns:
            Callable: Function that collates multiple ``ImageItem`` instances into
                a batch.
        """
        return ImageBatch.collate
