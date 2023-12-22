"""Tests for exporting."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from anomalib.deploy import ExportMode
from anomalib.models import get_available_models
from tests.integration.model.test_models import get_objects


@pytest.mark.parametrize("model_name", get_available_models())
def test_openvino_export(model_name: str, project_path: Path, dataset_path: Path) -> None:
    """Tests export trained models to OpenVINO to verify their compatibility.

    Args:
    model_name (str): Model type to export
    project_path (Path): Path to temporary project folder.
    dataset_path (Path): Path to dummy dataset.
    """
    model, datamodule, engine = get_objects(
        model_name=model_name,
        dataset_path=dataset_path,
        project_path=project_path,
    )

    engine.fit(model, datamodule=datamodule)

    export_path = project_path / model_name.lower() / "dummy"
    real_exported_path = engine.export(
        model=model,
        export_mode=ExportMode.OPENVINO,
        input_size=(256, 256),
        export_path=export_path,
        datamodule=datamodule,
    )

    xml_path = real_exported_path
    bin_path = xml_path.with_suffix(".bin")
    metadata_path = xml_path.parent / "metadata.json"

    # check all
    assert xml_path.exists()
    assert bin_path.exists()
    assert metadata_path.exists()
