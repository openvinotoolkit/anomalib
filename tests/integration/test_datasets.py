"""Test Dummy Datasets."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from tests.helpers.data import DummyImageDatasetGenerator, DummyVideoDatasetGenerator

from anomalib.data import ImageDataFormat, VideoDataFormat


@pytest.mark.run(order=1)
def test_if_dummy_datasets_are_generated(dataset_path: str) -> None:
    """Tests dummy datasets are properly generated.

    This test is run first to ensure that the dummy datasets are generated before any other tests are run.
    Overall, the test does the following:

    1. Generate the image dataset.
    2. Generate the video dataset.
    3. Check whether the dataset path exists.
    4. Check whether the dataset path contains any images or videos.
    """
    # 1. Create the image datasets.
    for data_format in list(ImageDataFormat):
        # Do not generate a dummy dataset for folder datasets.
        # We could use one of these datasets to test the folders datasets.
        if not data_format.value.startswith("folder"):
            dataset_generator = DummyImageDatasetGenerator(data_format=data_format, root=dataset_path)
            dataset_generator.generate_dataset()

            assert dataset_generator.dataset_root.is_dir()
            assert dataset_generator.dataset_root.glob("*.[png,jpg,tiff,bmp]")

    # 2. Create the video datasets.
    for data_format in list(VideoDataFormat):
        dataset_generator = DummyVideoDatasetGenerator(data_format=data_format, root=dataset_path)
        dataset_generator.generate_dataset()

        # Check whether the dataset directory is created.
        assert dataset_generator.dataset_root.is_dir()
        assert dataset_generator.dataset_root.glob("*.[png,avi,mat]")
