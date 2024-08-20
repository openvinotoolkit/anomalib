"""Tests for dataset split."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

from anomalib.data.utils.label import LabelName
from anomalib.data.utils.split import SubsetCreator


def test_create_by_label_segmentation(sample_segmentation_dataframe: pd.DataFrame) -> None:
    """Test creating subsets by label for a segmentation dataset."""
    subset_creator = SubsetCreator(sample_segmentation_dataframe)
    normal, abnormal = subset_creator.create("label")

    assert len(normal) + len(abnormal) == len(sample_segmentation_dataframe)
    assert all(normal.label_index == LabelName.NORMAL)
    assert all(abnormal.label_index == LabelName.ABNORMAL)
    assert "mask_path" in normal.columns
    assert "mask_path" in abnormal.columns


def test_create_by_label_classification(sample_classification_dataframe: pd.DataFrame) -> None:
    """Test creating subsets by label for a classification dataset."""
    subset_creator = SubsetCreator(sample_classification_dataframe)
    normal, abnormal = subset_creator.create("label")

    assert len(normal) + len(abnormal) == len(sample_classification_dataframe)
    assert all(normal.label_index == LabelName.NORMAL)
    assert all(abnormal.label_index == LabelName.ABNORMAL)
    assert "mask_path" not in normal.columns
    assert "mask_path" not in abnormal.columns


def test_create_by_indices(sample_segmentation_dataframe: pd.DataFrame) -> None:
    """Test creating subsets by indices."""
    subset_creator = SubsetCreator(sample_segmentation_dataframe)
    indices = [[0, 1, 2], [3, 4], [5, 6, 7]]
    splits = subset_creator.create(indices)

    assert len(splits) == len(indices)
    assert [len(split) for split in splits] == [len(idx) for idx in indices]
    for split, idx_list in zip(splits, indices, strict=False):
        assert all(split.iloc[i]["image_path"] == f"image_{idx}.jpg" for i, idx in enumerate(idx_list))


def test_create_by_ratio(sample_classification_dataframe: pd.DataFrame) -> None:
    """Test creating subsets by ratio."""
    subset_creator = SubsetCreator(sample_classification_dataframe)
    ratios = [0.7, 0.2, 0.1]
    splits = subset_creator.create(ratios, label_aware=False, seed=42)

    assert len(splits) == len(ratios)
    assert [len(split) for split in splits] == [70, 20, 10]


def test_create_by_count(sample_segmentation_dataframe: pd.DataFrame) -> None:
    """Test creating subsets by count."""
    subset_creator = SubsetCreator(sample_segmentation_dataframe)
    counts = [70, 20, 10]
    splits = subset_creator.create(counts, label_aware=True, seed=42)

    assert len(splits) == len(counts)
    assert [len(split) for split in splits] == counts


def test_create_by_mixed_criteria(sample_classification_dataframe: pd.DataFrame) -> None:
    """Test creating subsets by mixed criteria."""
    subset_creator = SubsetCreator(sample_classification_dataframe)
    criteria = [{"label": LabelName.NORMAL, "ratio": 0.7}, {"label": LabelName.ABNORMAL, "count": 20}]
    splits = subset_creator.create(criteria, seed=42, label_aware=True)

    normal_samples = sample_classification_dataframe[sample_classification_dataframe.label_index == LabelName.NORMAL]
    expected_normal_count = round(len(normal_samples) * 0.7)

    assert len(splits) == len(criteria)
    assert len(splits[0]) == expected_normal_count
    assert len(splits[1]) == 20
    assert all(splits[0].label_index == LabelName.NORMAL)
    assert all(splits[1].label_index == LabelName.ABNORMAL)


def test_invalid_criteria(sample_segmentation_dataframe: pd.DataFrame) -> None:
    """Test behavior with invalid criteria."""
    subset_creator = SubsetCreator(sample_segmentation_dataframe)
    with pytest.raises(ValueError, match="Invalid sequence type for splitting: invalid_criteria"):
        subset_creator.create("invalid_criteria")


def test_overlapping_indices(sample_classification_dataframe: pd.DataFrame) -> None:
    """Test behavior with overlapping indices."""
    subset_creator = SubsetCreator(sample_classification_dataframe)
    with pytest.raises(ValueError, match="Overlapping indices detected in the provided index lists."):
        subset_creator.create([[0, 1, 2], [2, 3, 4]])


def test_ratio_sum_exceeds_one(sample_segmentation_dataframe: pd.DataFrame) -> None:
    """Test behavior when ratio sum exceeds one."""
    subset_creator = SubsetCreator(sample_segmentation_dataframe)
    with pytest.raises(ValueError, match="Sum of ratios must not exceed 1"):
        subset_creator.create([0.7, 0.5])
