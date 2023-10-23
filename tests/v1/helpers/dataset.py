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

    def _generate_dummy_mvtec_dataset(
        self,
        normal_dir: str = "good",
        abnormal_dir: str | None = None,
        mask_suffix: str = "_mask",
    ) -> None:
        """Generates dummy MVTecAD dataset in a temporary directory using the same convention as MVTec AD."""
        # Create normal images.
        for split in ("train", "test"):
            path = self.root / "shapes" / split / normal_dir
            path.mkdir(parents=True, exist_ok=True)

            num_images = self.num_train if split == "train" else self.num_test
            for i in range(num_images):
                image, _ = random_shapes(image_shape=self.image_shape, max_shapes=1, shape=self.train_shape)
                imsave(path / f"{i:03}.png", image, check_contrast=False)

        # Create abnormal test images and masks.
        abnormal_dir = abnormal_dir or self.test_shape
        path = self.root / "shapes" / "test" / abnormal_dir
        mask_path = self.root / "shapes" / "ground_truth" / abnormal_dir

        path.mkdir(parents=True, exist_ok=True)
        mask_path.mkdir(parents=True, exist_ok=True)
        for i in range(self.num_test):
            image, _ = random_shapes(image_shape=self.image_shape, max_shapes=1, shape=self.test_shape)

            # Background of ``image`` is white, so mask can be created by thresholding.
            mask = image[..., 0] < 255

            # Save both the image and the mask.
            imsave(path / f"{i:03}.png", image, check_contrast=False)
            imsave(mask_path / f"{i:03}{mask_suffix}.png", img_as_ubyte(mask), check_contrast=False)

    def _generate_dummy_btech_dataset(self) -> None:
        """Generate dummy BeanTech dataset in directory using the same convention as BeanTech AD."""
        # BeanTech AD follows the same convention as MVTec AD.
        self._generate_dummy_mvtec_dataset(normal_dir="ok", abnormal_dir="ko", mask_suffix="")

    def _generate_dummy_mvtec_3d_dataset(self) -> None:
        """Generate dummy MVTec 3D AD dataset in a temporary directory using the same convention as MVTec AD."""
        # Create training and validation images.
        for split in ("train", "validation"):
            split_path = self.root / "shapes" / split / "good"
            for directory in ("rgb", "xyz"):
                (split_path / directory).mkdir(parents=True, exist_ok=True)
                extension = ".tiff" if directory == "xyz" else ".png"
                for i in range(self.num_train):
                    image, _ = random_shapes(image_shape=self.image_shape, max_shapes=1, shape=self.train_shape)
                    imsave(split_path / directory / f"{i:03}{extension}", image, check_contrast=False)

        ## Create test images.
        test_path = self.root / "shapes" / "test"
        for category in ("good", "crack"):
            shape = self.train_shape if category == "good" else self.test_shape
            for i in range(self.num_test):
                image, _ = random_shapes(image_shape=self.image_shape, max_shapes=1, shape=shape)
                for directory in ("rgb", "xyz", "gt"):
                    (test_path / category / directory).mkdir(parents=True, exist_ok=True)

                    # Background of ``image`` is white, so mask can be created by thresholding.
                    # Assign mask to ``image`` if ``directory`` is "gt", and save.
                    image = img_as_ubyte(image[..., 0] < 255) if directory == "gt" else image

                    # Save rgb, xyz, and gt images.
                    extension = ".tiff" if directory == "xyz" else ".png"
                    imsave(test_path / category / directory / f"{i:03}{extension}", image, check_contrast=False)

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
