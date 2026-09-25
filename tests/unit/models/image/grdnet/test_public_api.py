# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the public GRD-Net API and configuration paths."""

from pathlib import Path

import pytest

from anomalib.cli import AnomalibCLI
from anomalib.data import GRDNetBatch, GRDNetFolder, GRDNetFolderDataset, GRDNetItem, MVTecAD
from anomalib.data.dataclasses import GRDNetBatch as DataclassesGRDNetBatch
from anomalib.data.dataclasses import GRDNetItem as DataclassesGRDNetItem
from anomalib.data.datamodules import GRDNetFolder as DatamodulesGRDNetFolder
from anomalib.data.datasets import GRDNetFolderDataset as DatasetsGRDNetFolderDataset
from anomalib.models import GRDNet, get_model, list_models
from anomalib.models.image import GRDNet as ImageGRDNet


def test_public_imports() -> None:
    """GRD-Net model and data types are available from their public packages."""
    assert ImageGRDNet is GRDNet
    assert DataclassesGRDNetItem is GRDNetItem
    assert DataclassesGRDNetBatch is GRDNetBatch
    assert DatamodulesGRDNetFolder is GRDNetFolder
    assert DatasetsGRDNetFolderDataset is GRDNetFolderDataset


@pytest.mark.parametrize("name", ["GRDNet", "grd_net", "g_r_d_net"])
def test_get_model_by_name(name: str) -> None:
    """Existing model-name normalization resolves GRD-Net spellings."""
    assert isinstance(get_model(name), GRDNet)


def test_model_listing_uses_existing_acronym_conversion() -> None:
    """Model listings retain the repository's existing acronym conversion."""
    assert "g_r_d_net" in list_models()
    assert "GRDNet" in list_models(case="pascal")


def test_yaml_configuration_with_mvtec(tmp_path: Path) -> None:
    """A YAML configuration constructs GRD-Net with the standard MVTec datamodule."""
    config_path = tmp_path / "grdnet.yaml"
    config_path.write_text(
        f"""
model:
  class_path: anomalib.models.GRDNet
  init_args:
    texture_source: image
data:
  class_path: anomalib.data.MVTecAD
  init_args:
    root: {tmp_path / "mvtec"}
    category: bottle
trainer:
  logger: false
""",
        encoding="utf-8",
    )

    cli = AnomalibCLI(args=["fit", "--config", str(config_path)], run=False)

    assert isinstance(cli.model, GRDNet)
    assert cli.model.anomaly_generator.texture_source == "image"
    assert isinstance(cli.datamodule, MVTecAD)


def test_cli_configuration_with_roi_folder(tmp_path: Path) -> None:
    """CLI arguments construct the ROI-aware folder datamodule."""
    cli = AnomalibCLI(
        args=[
            "fit",
            "--model",
            "anomalib.models.GRDNet",
            "--data",
            "anomalib.data.GRDNetFolder",
            "--data.name",
            "custom",
            "--data.root",
            str(tmp_path),
            "--data.normal_dir",
            "train/good",
            "--data.roi_dir",
            "roi",
        ],
        run=False,
    )

    assert isinstance(cli.model, GRDNet)
    assert isinstance(cli.datamodule, GRDNetFolder)
    assert cli.datamodule.roi_dir == "roi"
