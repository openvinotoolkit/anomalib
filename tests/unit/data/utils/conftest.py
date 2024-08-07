"""Tests for dataset filter."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# anomalib/tests/data/utils/conftest.py

import numpy as np
import pandas as pd
import pytest

from anomalib.data.utils.label import LabelName


@pytest.fixture()
def sample_segmentation_dataframe() -> pd.DataFrame:
    """Create a sample DataFrame for segmentation tasks.

    This fixture generates a DataFrame with 100 rows, containing image paths,
    label indices (normal or abnormal), and corresponding mask paths.

    Returns:
        pd.DataFrame: A DataFrame with columns 'image_path', 'label_index', and 'mask_path'.
    """
    rng = np.random.default_rng(42)  # Create a Generator instance with seed 42
    return pd.DataFrame(
        {
            "image_path": [f"image_{i}.jpg" for i in range(100)],
            "label_index": rng.choice([LabelName.NORMAL, LabelName.ABNORMAL], size=100),
            "mask_path": [f"mask_{i}.png" for i in range(100)],
        },
    )


@pytest.fixture()
def sample_classification_dataframe() -> pd.DataFrame:
    """Create a sample DataFrame for classification tasks.

    This fixture generates a DataFrame with 100 rows, containing image paths
    and label indices (normal or abnormal), without mask paths.

    Returns:
        pd.DataFrame: A DataFrame with columns 'image_path' and 'label_index'.
    """
    rng = np.random.default_rng(42)  # Create a Generator instance with seed 42
    return pd.DataFrame(
        {
            "image_path": [f"image_{i}.jpg" for i in range(100)],
            "label_index": rng.choice([LabelName.NORMAL, LabelName.ABNORMAL], size=100),
        },
    )
