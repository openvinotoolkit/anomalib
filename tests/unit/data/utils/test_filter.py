"""Tests for dataset filter."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

from anomalib.data.utils.filter import DatasetFilter
from anomalib.data.utils.label import LabelName


def test_filter_by_label_segmentation(sample_segmentation_dataframe: pd.DataFrame) -> None:
    """Test filtering by label for a segmentation dataset."""
    dataset_filter = DatasetFilter(sample_segmentation_dataframe)
    normal_samples = dataset_filter.apply(by=LabelName.NORMAL)
    abnormal_samples = dataset_filter.apply(by=LabelName.ABNORMAL)

    assert len(normal_samples) + len(abnormal_samples) == len(sample_segmentation_dataframe)
    assert all(normal_samples.label_index == LabelName.NORMAL)
    assert all(abnormal_samples.label_index == LabelName.ABNORMAL)
    assert "mask_path" in normal_samples.columns
    assert "mask_path" in abnormal_samples.columns


def test_filter_by_label_classification(sample_classification_dataframe: pd.DataFrame) -> None:
    """Test filtering by label for a classification dataset."""
    dataset_filter = DatasetFilter(sample_classification_dataframe)
    normal_samples = dataset_filter.apply(by=LabelName.NORMAL)
    abnormal_samples = dataset_filter.apply(by=LabelName.ABNORMAL)

    assert len(normal_samples) + len(abnormal_samples) == len(sample_classification_dataframe)
    assert all(normal_samples.label_index == LabelName.NORMAL)
    assert all(abnormal_samples.label_index == LabelName.ABNORMAL)
    assert "mask_path" not in normal_samples.columns
    assert "mask_path" not in abnormal_samples.columns


def test_filter_by_indices(sample_segmentation_dataframe: pd.DataFrame) -> None:
    """Test filtering by specific indices."""
    dataset_filter = DatasetFilter(sample_segmentation_dataframe)
    indices = [0, 10, 20, 30]
    filtered_samples = dataset_filter.apply(by=indices)

    assert len(filtered_samples) == len(indices)
    assert list(filtered_samples.index) == [0, 1, 2, 3]
    assert all(filtered_samples.iloc[i]["image_path"] == f"image_{indices[i]}.jpg" for i in range(len(indices)))


def test_filter_by_ratio(sample_classification_dataframe: pd.DataFrame) -> None:
    """Test filtering by ratio."""
    dataset_filter = DatasetFilter(sample_classification_dataframe)
    ratio = 0.3
    filtered_samples = dataset_filter.apply(by=ratio, seed=42)

    assert len(filtered_samples) == int(len(sample_classification_dataframe) * ratio)


def test_filter_by_count(sample_segmentation_dataframe: pd.DataFrame) -> None:
    """Test filtering by count."""
    dataset_filter = DatasetFilter(sample_segmentation_dataframe)
    count = 50
    filtered_samples = dataset_filter.apply(by=count, seed=42)

    assert len(filtered_samples) == count


def test_filter_by_multiple(sample_classification_dataframe: pd.DataFrame) -> None:
    """Test filtering by multiple criteria."""
    dataset_filter = DatasetFilter(sample_classification_dataframe)
    filters = {"label": LabelName.NORMAL, "ratio": 0.5}
    filtered_samples = dataset_filter.apply(by=filters, seed=42)

    normal_samples = sample_classification_dataframe[sample_classification_dataframe.label_index == LabelName.NORMAL]
    expected_count = int(len(normal_samples) * 0.5)

    assert len(filtered_samples) == expected_count
    assert all(filtered_samples.label_index == LabelName.NORMAL)


def test_invalid_filter(sample_segmentation_dataframe: pd.DataFrame) -> None:
    """Test behavior with invalid filter."""
    dataset_filter = DatasetFilter(sample_segmentation_dataframe)
    with pytest.raises(ValueError, match="Unknown filter key: invalid. Must be 'label', 'ratio', or 'count'."):
        dataset_filter.apply(by={"invalid": "filter"})


def test_inplace_filter(sample_classification_dataframe: pd.DataFrame) -> None:
    """Test inplace filtering."""
    dataset_filter = DatasetFilter(sample_classification_dataframe)
    original_len = len(dataset_filter.samples)
    dataset_filter.apply(by=0.5, inplace=True, seed=42)

    assert len(dataset_filter.samples) == int(original_len * 0.5)
    assert len(dataset_filter.samples) == int(original_len * 0.5)
