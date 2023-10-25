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
from anomalib.data.utils import Augmenter, LabelName


class DummyImageGenerator:
    """Dummy image generator.

    Args:
        image_shape (tuple[int, int], optional): Image shape. Defaults to (256, 256).
    """

    def __init__(self, image_shape: tuple[int, int] = (256, 256)) -> None:
        self.image_shape = image_shape
        self.augmenter = Augmenter()

    def generate_normal_image(self) -> tuple[np.ndarray, np.ndarray]:
        """Generate a normal image."""
        image = random_shapes(image_shape=self.image_shape, min_size=256, max_shapes=1, shape="rectangle")[0]
        mask = np.zeros_like(image).astype(np.uint8)

        return image, mask

    def generate_abnormal_image(self, beta: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
        """Generate an abnormal image.

        Args:
            beta (float, optional): beta value for superimposing perturbation on image. Defaults to 0.2.

        Returns:
            tuple[np.ndarray, np.ndarray]: Abnormal image and mask.
        """
        # Generate a random image.
        image, _ = self.generate_normal_image()

        # Generate perturbation.
        perturbation, mask = self.augmenter.generate_perturbation(height=self.image_shape[0], width=self.image_shape[1])

        # Superimpose perturbation on image ``img``.
        abnormal_image = (image * (1 - mask) + (beta) * perturbation + (1 - beta) * image * (mask)).astype(np.uint8)

        return abnormal_image, mask.squeeze()

    def generate_image(
        self,
        label: LabelName = LabelName.NORMAL,
        image_filename: Path | str | None = None,
        mask_filename: Path | str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate a random image.

        Args:
            label (LabelName, optional): Image label (NORMAL - 0 or ABNORMAL - 1). Defaults to 0.
            image_filename (Path | str | None, optional): Image filename to save to filesytem. Defaults to None.
            mask_filename (Path | str | None, optional): Mask filename to save to filesystem. Defaults to None.

        Returns:
            tuple[np.ndarray, np.ndarray]: Image and mask.
        """
        func = self.generate_normal_image if label == LabelName.NORMAL else self.generate_abnormal_image
        image, mask = func()

        if image_filename:
            self.save_image(filename=image_filename, image=image)

        if mask_filename:
            self.save_image(filename=mask_filename, image=img_as_ubyte(mask))

        return image, mask

    def save_image(self, filename: Path | str, image: np.ndarray, check_contrast: bool = False) -> None:
        """Save image to filesystem.

        Args:
            filename (Path | str): Filename to save image to.
            image (np.ndarray): Image to save.
            check_contrast (bool, optional): Check for low contrast and print warning. Defaults to False.
        """
        filename = Path(filename)

        # Check if the directory exists.
        filename.parent.mkdir(parents=True, exist_ok=True)
        # Save image.
        imsave(fname=filename, arr=image, check_contrast=check_contrast)


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
        normal_category: str = "good",
        abnormal_category: str = "bad",
        seed: int | None = None,
    ) -> None:
        if data_format not in list(DataFormat):
            message = f"Invalid data format {data_format}. Valid options are {list(DataFormat)}."
            raise ValueError(message)

        self.data_format = data_format
        self.root = Path(mkdtemp() if root is None else root)
        self.num_train = num_train
        self.num_test = num_test
        self.normal_category = normal_category
        self.abnormal_category = abnormal_category
        self.image_shape = image_shape
        self.num_channels = num_channels
        self.min_size = min_size
        self.rng = np.random.default_rng(seed) if seed else None
        self.image_generator = DummyImageGenerator(image_shape=image_shape)

    def _generate_dummy_mvtec_dataset(
        self,
        normal_dir: str = "good",
        abnormal_dir: str | None = None,
        image_extension: str = ".png",
        mask_suffix: str = "_mask",
        mask_extension: str = ".png",
    ) -> None:
        """Generates dummy MVTecAD dataset in a temporary directory using the same convention as MVTec AD."""
        # Create normal images.
        for split in ("train", "test"):
            path = self.root / "shapes" / split / normal_dir
            num_images = self.num_train if split == "train" else self.num_test
            for i in range(num_images):
                self.image_generator.generate_image(LabelName.NORMAL, path / f"{i:03}{image_extension}")

        # Create abnormal test images and masks.
        abnormal_dir = abnormal_dir or self.abnormal_category
        path = self.root / "shapes" / "test" / abnormal_dir
        mask_path = self.root / "shapes" / "ground_truth" / abnormal_dir

        for i in range(self.num_test):
            label = LabelName.ABNORMAL
            image_filename = path / f"{i:03}{image_extension}"
            mask_filename = mask_path / f"{i:03}{mask_suffix}{mask_extension}"
            self.image_generator.generate_image(label, image_filename, mask_filename)

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
                    image, _ = random_shapes(image_shape=self.image_shape, max_shapes=1, shape=self.normal_category)
                    imsave(split_path / directory / f"{i:03}{extension}", image, check_contrast=False)

        ## Create test images.
        test_path = self.root / "shapes" / "test"
        for category in ("good", "crack"):
            shape = self.normal_category if category == "good" else self.abnormal_category
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

    def _generate_dummy_visa_dataset(self) -> None:
        """Generate dummy Visa dataset in directory using the same convention as Visa AD."""
        # Visa dataset on anomalib follows the same convention as MVTec AD.
        # The only difference is that the root directory has a subdirectory called "visa_pytorch".
        self.root = self.root / "visa_pytorch"
        self._generate_dummy_mvtec_dataset(normal_dir="good", abnormal_dir="bad", image_extension=".JPG")

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
