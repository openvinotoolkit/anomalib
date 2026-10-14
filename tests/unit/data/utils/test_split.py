# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for dataset split utils."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from anomalib.data.datasets.image.folder import FolderDataset
from anomalib.data.utils.split import random_split


def _make_image(path: Path) -> None:
    """Write a small dummy RGB image to ``path``."""
    rng = np.random.default_rng(seed=len(str(path)))
    image = rng.integers(0, 255, size=(32, 32, 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)


def _make_folder_dataset(root: Path, num_normal: int, num_abnormal: int) -> FolderDataset:
    """Create a ``FolderDataset`` backed by generated dummy images."""
    for i in range(num_normal):
        _make_image(root / "good" / f"{i:03d}.png")
    for i in range(num_abnormal):
        _make_image(root / "bad" / f"{i:03d}.png")
    return FolderDataset(
        name="dummy",
        normal_dir=root / "good",
        abnormal_dir=(root / "bad") if num_abnormal else None,
    )


class TestRandomSplit:
    """Tests for the ``random_split`` function."""

    @staticmethod
    def test_ratio_preserves_total_length(tmp_path: Path) -> None:
        """Split subsets must add up to the source dataset."""
        dataset = _make_folder_dataset(tmp_path, num_normal=8, num_abnormal=4)
        splits = random_split(dataset, [0.6, 0.4], seed=42)
        assert len(splits) == 2
        assert sum(len(split) for split in splits) == len(dataset)

    @staticmethod
    def test_label_aware_single_sample_label_group(tmp_path: Path) -> None:
        """A label group with a single sample must not crash the split.

        Regression test: when ``floor(len * ratio)`` was 0 for every ratio
        (e.g. one sample with a 50/50 split), the remainder-distribution
        loop computed ``i % sum(subset_lengths)`` = ``i % 0`` and raised
        ``ZeroDivisionError``.
        """
        dataset = _make_folder_dataset(tmp_path, num_normal=1, num_abnormal=4)
        splits = random_split(dataset, 0.5, label_aware=True, seed=42)
        assert len(splits) == 2
        assert sum(len(split) for split in splits) == len(dataset)

    @staticmethod
    def test_label_aware_keeps_labels_in_both_subsets(tmp_path: Path) -> None:
        """Label-aware split keeps both labels represented when possible."""
        dataset = _make_folder_dataset(tmp_path, num_normal=6, num_abnormal=6)
        splits = random_split(dataset, 0.5, label_aware=True, seed=42)
        for split in splits:
            assert set(split.samples["label_index"]) == {0, 1}

    @staticmethod
    def test_invalid_ratio_sum_raises(tmp_path: Path) -> None:
        """Ratios that do not sum to 1 must raise a ``ValueError``."""
        dataset = _make_folder_dataset(tmp_path, num_normal=4, num_abnormal=0)
        with pytest.raises(ValueError, match="must sum to 1"):
            random_split(dataset, [0.5, 0.4])

    @staticmethod
    def test_seed_zero_is_reproducible(tmp_path: Path) -> None:
        """``seed=0`` is a valid seed and must give a reproducible split.

        Regression test: the generator was created with ``if seed`` rather
        than ``if seed is not None``, so the falsy ``0`` fell through to an
        unseeded generator.
        """
        dataset = _make_folder_dataset(tmp_path, num_normal=8, num_abnormal=4)
        first = random_split(dataset, [0.5, 0.5], seed=0)
        second = random_split(dataset, [0.5, 0.5], seed=0)
        for split_a, split_b in zip(first, second, strict=True):
            assert split_a.samples["image_path"].tolist() == split_b.samples["image_path"].tolist()
