"""Test tiled ensemble training and prediction."""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
import yaml

from anomalib.pipelines.tiled_ensemble import EvalTiledEnsemble, TrainTiledEnsemble


@pytest.fixture(scope="session")
def get_mock_environment(dataset_path: Path, project_path: Path) -> Path:
    """Return mock directory for testing with datapath setup to dummy data."""
    ens_temp_dir = project_path / "ens_tmp"
    ens_temp_dir.mkdir(exist_ok=True)

    with Path("tests/integration/pipelines/tiled_ensemble.yaml").open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    # use separate project temp dir to avoid messing with other tests
    config["default_root_dir"] = str(ens_temp_dir)
    config["data"]["init_args"]["root"] = str(dataset_path / "mvtec")

    with (Path(ens_temp_dir) / "tiled_ensemble.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file)

    return Path(ens_temp_dir)


def test_train(get_mock_environment: Path, capsys: pytest.CaptureFixture) -> None:
    """Test training of the tiled ensemble."""
    train_pipeline = TrainTiledEnsemble()
    train_parser = train_pipeline.get_parser()
    args = train_parser.parse_args(["--config", str(get_mock_environment / "tiled_ensemble.yaml")])
    train_pipeline.run(args)
    # check that no errors were printed -> all stages were successful
    out = capsys.readouterr().out
    assert not any(line.startswith("There were some errors") for line in out.split("\n"))


def test_predict(get_mock_environment: Path, capsys: pytest.CaptureFixture) -> None:
    """Test prediction with the tiled ensemble."""
    predict_pipeline = EvalTiledEnsemble(root_dir=get_mock_environment / "Padim" / "MVTec" / "dummy" / "v0")
    predict_parser = predict_pipeline.get_parser()
    args = predict_parser.parse_args(["--config", str(get_mock_environment / "tiled_ensemble.yaml")])
    predict_pipeline.run(args)
    # check that no errors were printed -> all stages were successful
    out = capsys.readouterr().out
    assert not any(line.startswith("There were some errors") for line in out.split("\n"))


def test_visualisation(get_mock_environment: Path) -> None:
    """Test that images were produced."""
    assert (get_mock_environment / "Padim/MVTec/dummy/v0/images/bad/000.png").exists()


def test_metric_results(get_mock_environment: Path) -> None:
    """Test that metrics were saved."""
    assert (get_mock_environment / "Padim/MVTec/dummy/v0/metric_results.csv").exists()
