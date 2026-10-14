# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit Tests - Tabular Datamodule."""

import pickle  # noqa: S403
import tempfile
from pathlib import Path

import pandas as pd
import pytest
from torchvision.transforms.v2 import Resize

from anomalib.data import Folder, Tabular
from tests.unit.data.datamodule.base.image import _TestAnomalibImageDatamodule


class _EvilPicklePayload:
    """Pickle payload that writes a marker file when deserialized."""

    def __init__(self, marker: Path) -> None:
        self._marker = marker

    def __reduce__(self) -> tuple:
        return (Path(self._marker).write_text, ("pwned",))


class TestTabular(_TestAnomalibImageDatamodule):
    """Tabular Datamodule Unit Tests."""

    @staticmethod
    def get_samples_dataframe(mvtecad_path: Path) -> pd.DataFrame:
        """Create samples DataFrame using the Folder datamodule."""
        datamodule_ = Folder(
            name="dummy",
            root=mvtecad_path / "dummy",
            normal_dir="train/good",
            abnormal_dir="test/bad",
            normal_test_dir="test/good",
            mask_dir="ground_truth/bad",
            train_batch_size=4,
            eval_batch_size=4,
            num_workers=0,
        )
        datamodule_.setup()
        return pd.concat([
            datamodule_.train_data.samples,
            datamodule_.test_data.samples,
            datamodule_.val_data.samples,
        ])

    @pytest.fixture(
        params=[
            None,
            ["label"],
            ["label_index"],
            ["split"],
            ["mask_path"],
        ],
    )
    @staticmethod
    def columns_to_drop(request: pytest.FixtureRequest) -> list[str] | None:
        """Return the columns to be dropped from the samples dataframe."""
        return request.param

    @pytest.fixture()
    @staticmethod
    def datamodule(mvtecad_path: Path, columns_to_drop: list | None) -> Tabular:
        """Create and return a Tabular datamodule."""
        samples = TestTabular.get_samples_dataframe(mvtecad_path)
        if columns_to_drop:
            samples = samples.drop(columns_to_drop, axis="columns")
        datamodule_ = Tabular(
            name="dummy",
            samples=samples,
            train_batch_size=4,
            eval_batch_size=4,
            num_workers=0,
            augmentations=Resize((256, 256)),
        )
        datamodule_.setup()
        return datamodule_

    @pytest.fixture()
    @staticmethod
    def fxt_data_config_path() -> str:
        """Return the path to the test data config."""
        return "examples/configs/data/tabular.yaml"


class TestTabularFromFile(TestTabular):
    """Tabular Datamodule Unit Tests for alternative constructor.

    Tests for the Datamodule creation from file.
    """

    @pytest.fixture()
    @staticmethod
    def datamodule(mvtecad_path: Path) -> Tabular:
        """Create and return a Tabular datamodule."""
        samples = TestTabular.get_samples_dataframe(mvtecad_path)
        with tempfile.NamedTemporaryFile(suffix=".csv") as samples_file:
            samples.to_csv(samples_file)
            samples_file.seek(0)

            datamodule_ = Tabular.from_file(
                name="dummy",
                file_path=samples_file.name,
                train_batch_size=4,
                eval_batch_size=4,
                num_workers=0,
                augmentations=Resize((256, 256)),
            )
            datamodule_.setup()

        return datamodule_


class TestTabularFromFileFormatValidation:
    """Tests for the file format allowlist in ``Tabular.from_file``.

    Regression tests for an ``insecure deserialization`` issue where
    ``Tabular.from_file`` resolved the pandas reader via
    ``getattr(pd, f"read_{file_format}")`` with no allowlist, allowing formats
    such as ``pickle`` and ``hdf`` to be selected. Loading a malicious
    ``*.pickle`` "dataset table" via the documented ``Tabular.from_file`` API
    executed arbitrary code embedded in the file during deserialization.
    """

    @staticmethod
    def test_missing_file_raises() -> None:
        """A non-existent file should raise FileNotFoundError before any format check."""
        with pytest.raises(FileNotFoundError):
            Tabular.from_file(name="dummy", file_path="/no/such/file.csv")

    @staticmethod
    def test_rejects_pickle_suffix_without_executing_payload(tmp_path: Path) -> None:
        """A malicious ``.pickle`` file must be rejected without being deserialized."""
        marker = tmp_path / "PWNED.txt"
        evil_path = tmp_path / "samples.pickle"
        with evil_path.open("wb") as f:
            pickle.dump(_EvilPicklePayload(marker), f)  # nosemgrep

        with pytest.raises(ValueError, match="Unsupported file format"):
            Tabular.from_file(name="dummy", file_path=str(evil_path))

        assert not marker.exists(), "Payload was executed: pickle file must not be deserialized"

    @staticmethod
    @pytest.mark.parametrize("file_format", ["pickle", "hdf", "xml", "html", "clipboard", "sql"])
    def test_rejects_dangerous_explicit_format(tmp_path: Path, file_format: str) -> None:
        """Explicitly requesting a disallowed format must also be rejected."""
        csv_path = tmp_path / "samples.csv"
        pd.DataFrame({"image_path": ["a.png"]}).to_csv(csv_path)

        with pytest.raises(ValueError, match="Unsupported file format"):
            Tabular.from_file(name="dummy", file_path=str(csv_path), file_format=file_format)

    @staticmethod
    def test_rejects_unknown_format(tmp_path: Path) -> None:
        """An unrecognized file suffix should raise a clear ValueError."""
        txt_path = tmp_path / "samples.txt"
        txt_path.write_text("not a real dataset table")

        with pytest.raises(ValueError, match="Unsupported file format"):
            Tabular.from_file(name="dummy", file_path=str(txt_path))

    @staticmethod
    def test_rejects_missing_format(tmp_path: Path) -> None:
        """A file with no suffix and no explicit file_format should raise ValueError."""
        no_suffix_path = tmp_path / "samples"
        no_suffix_path.write_text("image_path\n")

        with pytest.raises(ValueError, match="File format not specified"):
            Tabular.from_file(name="dummy", file_path=str(no_suffix_path))

    @staticmethod
    def test_format_is_case_insensitive(tmp_path: Path) -> None:
        """Format inference from the suffix should be case-insensitive."""
        csv_path = tmp_path / "samples.CSV"
        pd.DataFrame({"image_path": ["a.png"]}).to_csv(csv_path)

        datamodule_ = Tabular.from_file(name="dummy", file_path=str(csv_path))
        assert isinstance(datamodule_, Tabular)
