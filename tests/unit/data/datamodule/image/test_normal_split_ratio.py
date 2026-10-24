# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests pinning the removal of ``normal_split_ratio`` and the zero-ratio guard.

``normal_split_ratio`` was accepted and stored by the Folder and Tabular
datamodules without any code ever reading it. The argument is now gone: passing
it fails with a ``TypeError`` instead of silently doing nothing, and
``test_split_ratio`` alone controls how many normal training images are sampled
into the test split when the test set contains none. These tests pin the
removal, the zero ``test_split_ratio`` handling (which used to crash the
underlying ``random_split`` call), and the unchanged sampling semantics.

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


class TestFolderNormalSplitRatioRemoved:
    """Folder datamodule: the dead argument is gone for good."""

    @staticmethod
    def test_normal_split_ratio_is_rejected(tmp_path: Path) -> None:
        """Passing the removed argument fails loudly instead of being silently stored."""
        make_folder_layout(tmp_path)
        with pytest.raises(TypeError, match="normal_split_ratio"):
            Folder(
                name="dummy",
                root=tmp_path,
                normal_dir="normal",
                abnormal_dir="abnormal",
                normal_split_ratio=0.2,
                train_batch_size=4,
                eval_batch_size=4,
                num_workers=0,
            )

    @staticmethod
    def test_normal_split_ratio_attribute_is_gone(tmp_path: Path) -> None:
        """No attribute keeps the removed value around for introspection."""
        make_folder_layout(tmp_path)
        datamodule = Folder(
            name="dummy",
            root=tmp_path,
            normal_dir="normal",
            abnormal_dir="abnormal",
            train_batch_size=4,
            eval_batch_size=4,
            num_workers=0,
        )
        assert not hasattr(datamodule, "normal_split_ratio")


class TestFolderTestSplitRatio:
    """Folder datamodule: ``test_split_ratio`` alone drives the train-to-test sampling."""

    @staticmethod
    def test_ratio_zero_keeps_every_normal_in_train(tmp_path: Path) -> None:
        """A ratio of 0 is a valid request: no normal training image is moved."""
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
            test_split_ratio=0,
            seed=42,
        )
        datamodule.setup()
        assert normal_test_count(datamodule) == 0
        assert len(datamodule.train_data.samples) == 10

    @staticmethod
    def test_ratio_moves_normals_to_test(tmp_path: Path) -> None:
        """The ratio moves floor(n * ratio) normal training images into the test split."""
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
    def test_negative_ratio_is_rejected(tmp_path: Path) -> None:
        """A negative ``test_split_ratio`` stays rejected rather than silently treated as 0."""
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
    def test_split_skipped_when_test_normals_present(tmp_path: Path) -> None:
        """With normal test images on disk the ratio must not move train normals."""
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
            val_split_mode="none",
            train_batch_size=4,
            eval_batch_size=4,
            num_workers=0,
            test_split_ratio=0.9,
            seed=42,
        )
        datamodule.setup()
        # Test normals come from normal_test_dir only; no train image was moved.
        assert normal_test_count(datamodule) == 3
        assert len(datamodule.train_data.samples) == 4
        train_paths = set(datamodule.train_data.samples.image_path)
        test_paths = set(datamodule.test_data.samples.image_path)
        assert train_paths.isdisjoint(test_paths)


class TestTabularNormalSplitRatioRemoved:
    """Tabular datamodule: same removal, with samples given as a dataframe."""

    @staticmethod
    def test_normal_split_ratio_is_rejected(tmp_path: Path) -> None:
        """Passing the removed argument fails loudly instead of being silently stored."""
        samples = make_tabular_layout(tmp_path)
        with pytest.raises(TypeError, match="normal_split_ratio"):
            Tabular(
                name="dummy",
                samples=samples,
                normal_split_ratio=0.2,
                train_batch_size=4,
                eval_batch_size=4,
                num_workers=0,
            )


class TestTabularTestSplitRatio:
    """Tabular datamodule: ``test_split_ratio`` alone drives the train-to-test sampling."""

    @staticmethod
    def test_ratio_zero_keeps_every_normal_in_train(tmp_path: Path) -> None:
        """A ratio of 0 keeps every normal training sample in the train split."""
        samples = make_tabular_layout(tmp_path)
        datamodule = Tabular(
            name="dummy",
            samples=samples,
            val_split_mode="none",
            train_batch_size=4,
            eval_batch_size=4,
            num_workers=0,
            test_split_ratio=0,
            seed=42,
        )
        datamodule.setup()
        assert normal_test_count(datamodule) == 0
        assert len(datamodule.train_data.samples) == 10

    @staticmethod
    def test_ratio_moves_normals_to_test(tmp_path: Path) -> None:
        """The ratio moves floor(n * ratio) normal training samples into the test split."""
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

    @staticmethod
    def test_negative_ratio_is_rejected(tmp_path: Path) -> None:
        """A negative ``test_split_ratio`` stays rejected rather than silently treated as 0."""
        samples = make_tabular_layout(tmp_path)
        datamodule = Tabular(
            name="dummy",
            samples=samples,
            val_split_mode="none",
            train_batch_size=4,
            eval_batch_size=4,
            num_workers=0,
            test_split_ratio=-0.1,
            seed=42,
        )
        with pytest.raises(ValueError, match="between 0 and 1"):
            datamodule.setup()
