# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the public GRD-Net API and configuration paths."""

from pathlib import Path

import pytest

from anomalib.cli import AnomalibCLI
from anomalib.data import MVTecAD
from anomalib.models import GRDNet, get_model
from anomalib.models.image.grdnet.data import GRDNetFolder

_NONDEFAULT_PARAMETERS = {
    "contextual_l1_weight": 2.0,
    "contextual_ssim_weight": 0.5,
    "generator_scheduler_decay": 0.2,
    "generator_scheduler_patience": 0,
    "rotation_degrees": 30.0,
    "smoothing_kernel_size": 3,
}


def test_get_model_by_name() -> None:
    """Existing model discovery resolves GRD-Net."""
    assert isinstance(get_model("grd_net"), GRDNet)


@pytest.mark.parametrize("parameters", [{}, _NONDEFAULT_PARAMETERS])
def test_yaml_configuration_with_mvtec(tmp_path: Path, parameters: dict[str, float | int]) -> None:
    """A YAML configuration constructs GRD-Net with the standard MVTec datamodule."""
    config_path = tmp_path / "grdnet.yaml"
    parameter_lines = "\n".join(f"    {name}: {value}" for name, value in parameters.items())
    config_path.write_text(
        f"""
model:
  class_path: anomalib.models.GRDNet
  init_args:
    texture_source: image
{parameter_lines}
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
    expected = parameters or {
        "contextual_l1_weight": 1.0,
        "contextual_ssim_weight": 1.0,
        "generator_scheduler_decay": 0.1,
        "generator_scheduler_patience": 3,
        "rotation_degrees": 90.0,
        "smoothing_kernel_size": 21,
    }
    assert all(cli.model.hparams[name] == value for name, value in expected.items())
    assert cli.model.model.smoothing_kernel_size == expected["smoothing_kernel_size"]


def test_cli_configuration_with_roi_folder(tmp_path: Path) -> None:
    """CLI arguments construct the ROI-aware folder datamodule."""
    cli = AnomalibCLI(
        args=[
            "fit",
            "--model",
            "anomalib.models.GRDNet",
            "--data",
            "anomalib.models.image.grdnet.data.GRDNetFolder",
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
