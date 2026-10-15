# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``normal_split_ratio`` argument of the Folder and Tabular datamodules.

``normal_split_ratio`` historically accepted and stored a value that no code ever
read. These tests pin the restored semantics: when the test split contains no
normal images, an explicit ``normal_split_ratio`` drives how many normal training
images are moved into the test split, overriding ``test_split_ratio`` for that
sampling step.

All tests pass ``val_split_mode="none"`` so the normal images moved into the test
split stay there: the default ``from_test`` validation mode would afterwards move
half of the test normals into the validation set, which is unrelated to the
ratio under test.
"""

from pathlib import Path

import pandas as pd
import pytest

from anomalib.data import Folder, Tabular


def make_folder_layout(root: Path, num_train: int = 10, num_test: int = 4) -> None:
    """Create a folder layout whose test split contains no normal images."""
    normal_dir = root / "normal"
    abnormal_dir = root / "abnormal"
    normal_dir.mkdir(parents=True)
    abnormal_dir.mkdir(parents=True)
    for i in range(num_train):
        (normal_dir / f"{i:03}.png").write_bytes(b"")
    for i in range(num_test):
        (abnormal_dir / f"{i:03}.png").write_bytes(b"")


def normal_test_count(datamodule: Folder | Tabular) -> int:
    """Count normal images in the test split."""
    samples = datamodule.test_data.samples
    return int((samples.label_index == 0).sum())


def make_tabular_layout(root: Path, num_train: int = 10, num_test: int = 4) -> pd.DataFrame:
    """Create real files on disk and return a samples frame with no test normals."""
    (root / "train").mkdir(parents=True)
    (root / "test").mkdir(parents=True)
    rows = []
    for i in range(num_train):
        path = root / "train" / f"normal_{i:03}.png"
        path.write_bytes(b"")
        rows.append({"image_path": str(path), "label_index": 0, "split": "train"})
    for i in range(num_test):
        path = root / "test" / f"abnormal_{i:03}.png"
        path.write_bytes(b"")
        rows.append({"image_path": str(path), "label_index": 1, "split": "test"})
    return pd.DataFrame(rows)


class TestFolderNormalSplitRatio:
    """Folder datamodule: ``normal_split_ratio`` drives the train-to-test split of normals."""

    @staticmethod
    @pytest.mark.parametrize(("ratio", "expected"), [(0.2, 2), (0.5, 5), (0.1, 1)])
    def test_explicit_ratio_moves_normals_to_test(tmp_path: Path, ratio: float, expected: int) -> None:
        """An explicit ratio moves floor(n * ratio) normal images into the test split."""
        make_folder_layout(tmp_path)
        datamodule = Folder(
            name="dummy",
            root=tmp_path,
            normal_dir="normal",
            abnormal_dir="abnormal",
            normal_split_ratio=ratio,
            val_split_mode="none",
            train_batch_size=4,
            eval_batch_size=4,
            num_workers=0,
            test_split_ratio=0.9,  # deliberately different: the explicit value must win
            seed=42,
        )
        datamodule.setup()
        assert normal_test_count(datamodule) == expected

    @staticmethod
    def test_default_none_keeps_test_split_ratio(tmp_path: Path) -> None:
        """Without an explicit value, ``test_split_ratio`` keeps driving the sampling."""
        make_folder_layout(tmp_path)
        datamodule = Folder(
            name="dummy",
            root=tmp_path,
            normal_dir="normal",
            abnormal_dir="abnormal",
            val_split_mode="none",
            train_batch_size=4,
            eval_batch_size=4,
            num_workers=0,
            test_split_ratio=0.3,
            seed=42,
        )
        datamodule.setup()
        assert normal_test_count(datamodule) == 3

    @staticmethod
    def test_explicit_ratio_ignored_when_normal_test_dir_present(tmp_path: Path) -> None:
        """When the test split already contains normals, the ratio must not move train normals."""
        make_folder_layout(tmp_path, num_train=4)
        normal_test_dir = tmp_path / "normal_test"
        normal_test_dir.mkdir()
        for i in range(3):
            (normal_test_dir / f"{i:03}.png").write_bytes(b"")
        datamodule = Folder(
            name="dummy",
            root=tmp_path,
            normal_dir="normal",
            abnormal_dir="abnormal",
            normal_test_dir=normal_test_dir,
            normal_split_ratio=0.9,
            val_split_mode="none",
            train_batch_size=4,
            eval_batch_size=4,
            num_workers=0,
            test_split_ratio=0.3,
            seed=42,
        )
        datamodule.setup()
        # Test normals come from normal_test_dir only; no train image was moved.
        assert normal_test_count(datamodule) == 3
        train_paths = set(datamodule.train_data.samples.image_path)
        test_paths = set(datamodule.test_data.samples.image_path)
        assert train_paths.isdisjoint(test_paths)

    @staticmethod
    def test_ratio_zero_moves_no_normals(tmp_path: Path) -> None:
        """A ratio of 0 is a valid explicit request: keep every normal training image in train."""
        make_folder_layout(tmp_path)
        datamodule = Folder(
            name="dummy",
            root=tmp_path,
            normal_dir="normal",
            abnormal_dir="abnormal",
            normal_split_ratio=0.0,
            val_split_mode="none",
            train_batch_size=4,
            eval_batch_size=4,
            num_workers=0,
            seed=42,
        )
        datamodule.setup()
        assert normal_test_count(datamodule) == 0
        assert len(datamodule.train_data.samples) == 10

    @staticmethod
    def test_negative_ratio_is_rejected_not_silently_zeroed(tmp_path: Path) -> None:
        """A negative ``normal_split_ratio`` must be rejected, not silently treated as 0."""
        make_folder_layout(tmp_path)
        datamodule = Folder(
            name="dummy",
            root=tmp_path,
            normal_dir="normal",
            abnormal_dir="abnormal",
            normal_split_ratio=-0.1,
            val_split_mode="none",
            train_batch_size=4,
            eval_batch_size=4,
            num_workers=0,
            seed=42,
        )
        with pytest.raises(ValueError, match="between 0 and 1"):
            datamodule.setup()

    @staticmethod
    def test_negative_test_split_ratio_is_rejected(tmp_path: Path) -> None:
        """A negative ``test_split_ratio`` stays rejected when it drives the normal-image sampling."""
        make_folder_layout(tmp_path)
        datamodule = Folder(
            name="dummy",
            root=tmp_path,
            normal_dir="normal",
            abnormal_dir="abnormal",
            val_split_mode="none",
            train_batch_size=4,
            eval_batch_size=4,
            num_workers=0,
            test_split_ratio=-0.1,
            seed=42,
        )
        with pytest.raises(ValueError, match="between 0 and 1"):
            datamodule.setup()

    @staticmethod
    def test_normal_split_ratio_attribute_is_preserved(tmp_path: Path) -> None:
        """The stored attribute must reflect the value passed by the user."""
        make_folder_layout(tmp_path)
        datamodule = Folder(
            name="dummy",
            root=tmp_path,
            normal_dir="normal",
            abnormal_dir="abnormal",
            normal_split_ratio=0.25,
            val_split_mode="none",
            train_batch_size=4,
            eval_batch_size=4,
            num_workers=0,
            seed=42,
        )
        assert datamodule.normal_split_ratio == pytest.approx(0.25)


class TestTabularNormalSplitRatio:
    """Tabular datamodule: same semantics, with samples given as a dataframe."""

    @staticmethod
    def test_explicit_ratio_overrides_test_split_ratio(tmp_path: Path) -> None:
        """An explicit ``normal_split_ratio`` wins over ``test_split_ratio``."""
        samples = make_tabular_layout(tmp_path)
        datamodule = Tabular(
            name="dummy",
            samples=samples,
            normal_split_ratio=0.2,
            val_split_mode="none",
            train_batch_size=4,
            eval_batch_size=4,
            num_workers=0,
            test_split_ratio=0.9,
            seed=42,
        )
        datamodule.setup()
        assert normal_test_count(datamodule) == 2

    @staticmethod
    def test_ratio_zero_moves_no_normals(tmp_path: Path) -> None:
        """A ratio of 0 keeps every normal training sample in the train split."""
        samples = make_tabular_layout(tmp_path)
        datamodule = Tabular(
            name="dummy",
            samples=samples,
            normal_split_ratio=0.0,
            val_split_mode="none",
            train_batch_size=4,
            eval_batch_size=4,
            num_workers=0,
            seed=42,
        )
        datamodule.setup()
        assert normal_test_count(datamodule) == 0
        assert len(datamodule.train_data.samples) == 10

    @staticmethod
    def test_negative_ratio_is_rejected_not_silently_zeroed(tmp_path: Path) -> None:
        """A negative ``normal_split_ratio`` must be rejected, not silently treated as 0."""
        samples = make_tabular_layout(tmp_path)
        datamodule = Tabular(
            name="dummy",
            samples=samples,
            normal_split_ratio=-0.1,
            val_split_mode="none",
            train_batch_size=4,
            eval_batch_size=4,
            num_workers=0,
            seed=42,
        )
        with pytest.raises(ValueError, match="between 0 and 1"):
            datamodule.setup()

    @staticmethod
    def test_default_none_keeps_test_split_ratio(tmp_path: Path) -> None:
        """Without an explicit value, behavior is unchanged: ``test_split_ratio`` drives it."""
        samples = make_tabular_layout(tmp_path)
        datamodule = Tabular(
            name="dummy",
            samples=samples,
            val_split_mode="none",
            train_batch_size=4,
            eval_batch_size=4,
            num_workers=0,
            test_split_ratio=0.3,
            seed=42,
        )
        datamodule.setup()
        assert normal_test_count(datamodule) == 3
