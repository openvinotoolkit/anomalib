# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for dataset metadata path confinement."""

import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from anomalib.data import Visa
from anomalib.data.datasets.image.datumaro import make_datumaro_dataset
from anomalib.data.datasets.image.realiad import make_realiad_dataset
from anomalib.data.datasets.image.tabular import make_tabular_dataset
from anomalib.data.utils import LabelName, Split
from anomalib.data.utils.image import save_image
from anomalib.data.utils.path import resolve_path_under_root


class TestVisaPathConfinement:
    """VisA split CSV must not read files outside the dataset root."""

    @staticmethod
    def test_apply_cls1_split_rejects_out_of_root_image_path() -> None:
        """Malicious ``image_path`` in ``1cls.csv`` must not copy host files."""
        with TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            root = base / "visa"
            (root / "candle").mkdir(parents=True)
            (root / "split_csv").mkdir()

            outside_file = base / "outside_host_file.txt"
            outside_file.write_text("host contents outside the dataset root", encoding="utf-8")

            with (root / "split_csv" / "1cls.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["category", "split", "label", "image_path", "mask_path"])
                writer.writerow(["candle", "train", "normal", "../outside_host_file.txt", ""])

            datamodule = Visa(root=root, category="candle")
            with pytest.raises(ValueError, match="Access denied"):
                datamodule.prepare_data()

            leaked = root / "visa_pytorch" / "candle" / "train" / "good" / "outside_host_file.txt"
            assert not leaked.is_file()

    @staticmethod
    def test_prepare_data_rejects_symlinked_processed_category() -> None:
        """A symlinked processed-category dir must not be treated as already-split."""
        with TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            root = base / "visa"
            root.mkdir()

            outside_category = base / "outside_candle"
            outside_category.mkdir()

            split_root = root / "visa_pytorch"
            split_root.mkdir()
            (split_root / "candle").symlink_to(outside_category)

            datamodule = Visa(root=root, category="candle")
            with pytest.raises(ValueError, match="outside"):
                datamodule.prepare_data()

    @staticmethod
    def test_apply_cls1_split_rejects_symlinked_split_root() -> None:
        """``apply_cls1_split`` must not create directories through a symlinked split root."""
        with TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            root = base / "visa"
            root.mkdir()

            outside = base / "outside_split_root"
            outside.mkdir()

            # split_root ("visa_pytorch") is a symlink pointing outside root.
            (root / "visa_pytorch").symlink_to(outside)

            datamodule = Visa(root=root, category="candle")
            with pytest.raises(ValueError, match="outside"):
                datamodule.apply_cls1_split()

            assert not any(outside.iterdir())

    @staticmethod
    def test_apply_cls1_split_rejects_symlinked_leaf_category() -> None:
        """A symlinked leaf category dir under an in-root split_root must not be used."""
        with TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            root = base / "visa"
            (root / "split_csv").mkdir(parents=True)
            (root / "split_csv" / "1cls.csv").write_text(
                "category,split,label,image_path,mask_path\n",
                encoding="utf-8",
            )

            outside = base / "outside_category"
            outside.mkdir()

            split_root = root / "visa_pytorch"
            split_root.mkdir()
            # "candle" (one of the unselected categories the loop iterates over) is a
            # symlink pointing outside the dataset root.
            (split_root / "candle").symlink_to(outside)

            datamodule = Visa(root=root, category="capsules")
            with pytest.raises(ValueError, match="Access denied"):
                datamodule.apply_cls1_split()

            assert not any(outside.iterdir())

    @staticmethod
    def test_apply_cls1_split_rejects_symlinked_split_file() -> None:
        """A symlinked ``split_csv/1cls.csv`` must not be read from outside root."""
        with TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            root = base / "visa"
            (root / "split_csv").mkdir(parents=True)

            outside_file = base / "outside_1cls.csv"
            outside_file.write_text(
                "category,split,label,image_path,mask_path\ncandle,train,normal,foo.png,\n",
                encoding="utf-8",
            )
            (root / "split_csv" / "1cls.csv").symlink_to(outside_file)

            datamodule = Visa(root=root, category="candle")
            with pytest.raises(ValueError, match="Access denied"):
                datamodule.apply_cls1_split()


class TestMvtecAdPathConfinement:
    """``make_mvtec_ad_dataset`` must not follow symlinks that escape ``root``."""

    @staticmethod
    def test_rejects_symlinked_subdir_escape() -> None:
        """A symlinked category subdirectory must not leak files outside ``root``."""
        from anomalib.data.datasets.image.mvtecad import make_mvtec_ad_dataset

        with TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            outside = base / "outside"
            (outside / "good").mkdir(parents=True)
            (outside / "good" / "000.png").write_bytes(b"x")

            root = base / "candle"
            (root / "train" / "good").mkdir(parents=True)
            (root / "train" / "good" / "001.png").write_bytes(b"y")

            # "bad" is a symlink pointing outside root.
            (root / "train" / "bad").symlink_to(outside / "good")

            samples = make_mvtec_ad_dataset(root, split=Split.TRAIN)
            assert len(samples) == 1
            for image_path in samples["image_path"]:
                assert not str(Path(image_path).resolve()).startswith(str(outside.resolve()))


class TestDatumaroPathConfinement:
    """Datumaro JSON image paths must stay under the dataset root."""

    @staticmethod
    def test_rejects_path_traversal_in_image_path() -> None:
        """Reject ``items[].image.path`` that escapes the dataset root."""
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "datumaro"
            (root / "annotations").mkdir(parents=True)
            (root / "images" / "default").mkdir(parents=True)
            outside_file = Path(tmp_dir) / "outside_target.txt"
            outside_file.write_text("outside root", encoding="utf-8")

            annotations = {
                "categories": {"label": {"labels": [{"name": "Normal"}, {"name": "Anomalous"}]}},
                "items": [
                    {
                        "image": {"path": "../../../outside_target.txt"},
                        "annotations": [{"label_id": 0}],
                    },
                ],
            }
            (root / "annotations" / "default.json").write_text(json.dumps(annotations), encoding="utf-8")

            with pytest.raises(ValueError, match="Access denied"):
                make_datumaro_dataset(root)

    @staticmethod
    def test_rejects_symlinked_annotation_file() -> None:
        """A symlinked ``annotations/default.json`` must not be read from outside root."""
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "datumaro"
            (root / "annotations").mkdir(parents=True)
            (root / "images" / "default").mkdir(parents=True)

            outside_annotations = {
                "categories": {"label": {"labels": [{"name": "Normal"}]}},
                "items": [],
            }
            outside_file = Path(tmp_dir) / "outside_default.json"
            outside_file.write_text(json.dumps(outside_annotations), encoding="utf-8")
            (root / "annotations" / "default.json").symlink_to(outside_file)

            with pytest.raises(ValueError, match="Access denied"):
                make_datumaro_dataset(root)


class TestTabularPathConfinement:
    """Tabular paths must resolve under ``root`` when ``root`` is set."""

    @staticmethod
    def test_rejects_relative_escape_when_root_set() -> None:
        """Reject ``../`` image paths when a root is provided."""
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "data"
            root.mkdir()
            samples = {
                "image_path": ["../outside.png"],
                "label_index": [LabelName.NORMAL],
                "split": [Split.TRAIN],
            }
            with pytest.raises(ValueError, match="Access denied"):
                make_tabular_dataset(samples=samples, root=root)

    @staticmethod
    def test_rejects_absolute_path_outside_root() -> None:
        """Reject absolute image paths outside the provided root."""
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "data"
            root.mkdir()
            outside = Path(tmp_dir) / "outside.png"
            outside.write_bytes(b"x")
            samples = {
                "image_path": [str(outside)],
                "label_index": [LabelName.NORMAL],
                "split": [Split.TRAIN],
            }
            with pytest.raises(ValueError, match="Access denied"):
                make_tabular_dataset(samples=samples, root=root)

    @staticmethod
    def test_rejects_escape_with_empty_string_root() -> None:
        """An empty-string ``root`` must still confine paths, not be treated as unset."""
        import os

        with TemporaryDirectory() as tmp_dir:
            cwd_dir = Path(tmp_dir) / "data"
            cwd_dir.mkdir()
            outside = Path(tmp_dir) / "outside.png"
            outside.write_bytes(b"x")
            cwd = Path.cwd()
            os.chdir(cwd_dir)
            try:
                samples = {
                    "image_path": [str(outside)],
                    "label_index": [LabelName.NORMAL],
                    "split": [Split.TRAIN],
                }
                with pytest.raises(ValueError, match="Access denied"):
                    make_tabular_dataset(samples=samples, root="")
            finally:
                os.chdir(cwd)


class TestRealIADPathConfinement:
    """RealIAD JSON paths must stay under the category root."""

    @staticmethod
    def test_rejects_path_traversal_in_metadata() -> None:
        """Reject metadata ``image_path`` values that escape the root."""
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "realiad"
            root.mkdir()
            metadata = {
                "train": [
                    {
                        "image_path": "../../outside_target.jpg",
                        "anomaly_class": "OK",
                    },
                ],
                "test": [],
            }
            with pytest.raises(ValueError, match="Access denied"):
                make_realiad_dataset(root=root, split=Split.TRAIN, metadata=metadata)


class TestKaputtStylePathConfinement:
    """Kaputt-style relative joins must use ``resolve_path_under_root``."""

    @staticmethod
    def test_resolve_under_query_subdir_rejects_escape() -> None:
        """Parquet-relative paths must not escape the query image subdirectory."""
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "kaputt"
            image_subdir = root / "query-image"
            image_subdir.mkdir(parents=True)
            (Path(tmp_dir) / "outside_target.jpg").write_bytes(b"x")
            with pytest.raises(ValueError, match="Access denied"):
                resolve_path_under_root(image_subdir, "../outside_target.jpg", should_exist=False)


class TestSaveImagePathConfinement:
    """``save_image`` must not write outside the provided root."""

    @staticmethod
    def test_rejects_parent_directory_escape() -> None:
        """Reject filenames that escape ``root`` via ``..``."""
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "results"
            root.mkdir()
            image = np.zeros((4, 4, 3), dtype=np.uint8)
            with pytest.raises(ValueError, match="Access denied"):
                save_image(filename="../escaped.png", image=image, root=root)
            assert not (Path(tmp_dir) / "escaped.png").exists()
