"""Test Helpers - Dataset."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import shutil
from contextlib import ContextDecorator
from pathlib import Path
from tempfile import mkdtemp

import numpy as np
from skimage import img_as_ubyte
from skimage.draw import random_shapes
from skimage.io import imsave

from anomalib.data import DataFormat


class DummyDatasetGenerator(ContextDecorator):
    r"""Context for generating dummy shapes dataset.

    Args:
        root (str, optional): Path to the root directory. Defaults to None.
        num_train (int, optional): Number of training images to generate. Defaults to 1000.
        num_test (int, optional): Number of testing images to generate per category. Defaults to 100.
        img_height (int, optional): Height of the image. Defaults to 128.
        img_width (int, optional): Width of the image. Defaults to 128.
        max_size (Optional[int], optional): Maximum size of the test shapes. Defaults to 10.
        train_shapes (List[str], optional): List of good shapes. Defaults to ["circle", "rectangle"].
        test_shapes (List[str], optional): List of anomalous shapes. Defaults to ["triangle", "ellipse"].
        seed (int, optional): Fixes seed if any number greater than 0 is provided. 0 means no seed. Defaults to 0.

    Examples:
        >>> with DummyDatasetGenerator(num_train=10, num_test=10) as dataset_path:
        >>>     some_function()

        Alternatively, you can use it as a standalone class.
        This will create a temporary directory with the dataset.

        >>> generator = DummyDatasetGenerator(num_train=10, num_test=10)
        >>> generator.generate_dataset()

        If you want to use a specific directory, you can pass it as an argument.

        >>> generator = DummyDatasetGenerator(root="./datasets/dummy")
        >>> generator.generate_dataset()
    """

    def __init__(
        self,
        data_format: str = "mvtec",
        root: str | None = None,
        num_train: int = 5,
        num_test: int = 5,
        image_shape: tuple[int, int] = (256, 256),
        num_channels: int = 3,
        min_size: int = 64,
        train_shape: str = "rectangle",
        test_shape: str = "circle",
        seed: int | None = None,
    ) -> None:
        if data_format not in list(DataFormat):
            message = f"Invalid data format {data_format}. Valid options are {list(DataFormat)}."
            raise ValueError(message)

        self.data_format = data_format
        self.root = Path(mkdtemp() if root is None else root)
        self.num_train = num_train
        self.num_test = num_test
        self.train_shape = train_shape
        self.test_shape = test_shape
        self.image_shape = image_shape
        self.num_channels = num_channels
        self.min_size = min_size
        self.rng = np.random.default_rng(seed) if seed else None

    def _generate_dummy_mvtec_dataset(self) -> None:
        """Generates dummy MVTecAD dataset in a temporary directory using the same convention as MVTec AD."""
        # Create directory names
        train_path = self.root / "shapes" / "train" / "good"
        test_path = self.root / "shapes" / "test"
        normal_test_path = test_path / "good"
        abnormal_test_path = test_path / self.test_shape
        mask_path = self.root / "shapes" / "ground_truth" / self.test_shape

        # Create directories.
        train_path.mkdir(parents=True, exist_ok=True)
        normal_test_path.mkdir(parents=True, exist_ok=True)
        abnormal_test_path.mkdir(parents=True, exist_ok=True)
        mask_path.mkdir(parents=True, exist_ok=True)

        # Create normal training images.
        for i in range(self.num_train):
            image, _ = random_shapes(
                image_shape=self.image_shape,
                max_shapes=1,
                min_size=self.min_size,
                num_channels=self.num_channels,
                shape=self.train_shape,
                rng=self.rng,
            )
            imsave(train_path / f"{i:03}.png", image)

        # Create normal test images.
        for i in range(self.num_test):
            image, _ = random_shapes(
                image_shape=self.image_shape,
                max_shapes=1,
                min_size=self.min_size,
                num_channels=self.num_channels,
                shape=self.train_shape,
                rng=self.rng,
            )
            imsave(normal_test_path / f"{i:03}.png", image)

        # Create abnormal test images.
        for i in range(self.num_test):
            image, _ = random_shapes(
                image_shape=self.image_shape,
                max_shapes=1,
                min_size=self.min_size,
                num_channels=self.num_channels,
                shape=self.test_shape,
                rng=self.rng,
            )

            # Background of ``image`` is white, so mask can be created by thresholding.
            mask = image[..., 0] < 255

            # Save both the image and the mask.
            imsave(abnormal_test_path / f"{i:03}.png", image, check_contrast=False)
            imsave(mask_path / f"{i:03}_mask.png", img_as_ubyte(mask), check_contrast=False)

    def _generate_dummy_btech_dataset(self) -> None:
        """Generates dummy MVTecAD dataset in a temporary directory using the same convention as MVTec AD."""
        # Create directory names
        train_path = self.root / "shapes" / "train" / "ok"
        test_path = self.root / "shapes" / "test"
        normal_test_path = test_path / "ok"
        abnormal_test_path = test_path / "ko"
        mask_path = self.root / "shapes" / "ground_truth" / "ko"

        # Create directories.
        train_path.mkdir(parents=True, exist_ok=True)
        normal_test_path.mkdir(parents=True, exist_ok=True)
        abnormal_test_path.mkdir(parents=True, exist_ok=True)
        mask_path.mkdir(parents=True, exist_ok=True)

        # Create normal training images.
        for i in range(self.num_train):
            image, _ = random_shapes(
                image_shape=self.image_shape,
                max_shapes=1,
                min_size=self.min_size,
                num_channels=self.num_channels,
                shape=self.train_shape,
                rng=self.rng,
            )
            imsave(train_path / f"{i:03}.png", image)

        # Create normal test images.
        for i in range(self.num_test):
            image, _ = random_shapes(
                image_shape=self.image_shape,
                max_shapes=1,
                min_size=self.min_size,
                num_channels=self.num_channels,
                shape=self.train_shape,
                rng=self.rng,
            )
            imsave(normal_test_path / f"{i:03}.png", image)

        # Create abnormal test images.
        for i in range(self.num_test):
            image, _ = random_shapes(
                image_shape=self.image_shape,
                max_shapes=1,
                min_size=self.min_size,
                num_channels=self.num_channels,
                shape=self.test_shape,
                rng=self.rng,
            )

            # Background of ``image`` is white, so mask can be created by thresholding.
            mask = image[..., 0] < 255

            # Save both the image and the mask.
            imsave(abnormal_test_path / f"{i:03}.png", image, check_contrast=False)
            imsave(mask_path / f"{i:03}.png", img_as_ubyte(mask), check_contrast=False)

    def generate_dataset(self) -> None:
        """Generate dataset."""
        # get dataset specific ``generate_dataset`` function based on string.
        method_name = f"_generate_dummy_{self.data_format}_dataset"
        method = getattr(self, method_name)
        method()

    def __enter__(self) -> Path:
        """Creates the dataset in temp folder."""
        self.generate_dataset()
        return self.root

    def __exit__(self, _exc_type, _exc_value, _exc_traceback) -> None:  # noqa: ANN001
        """Cleanup the directory."""
        shutil.rmtree(self.root)
