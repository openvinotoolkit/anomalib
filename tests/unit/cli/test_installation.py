# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for installation utils."""

import os
import tempfile
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from pytest_mock import MockerFixture

from anomalib.cli.utils.installation import (
    get_cuda_suffix,
    get_cuda_version,
    get_hardware_suffix,
    get_requirements,
    get_torch_install_args,
    parse_requirements,
    update_cuda_version_with_available_torch_cuda_build,
)


@pytest.fixture()
def requirements_file() -> Path:
    """Create a temporary requirements file with some example requirements."""
    requirements = ["numpy==1.19.5", "opencv-python-headless>=4.5.1.48"]
    with tempfile.NamedTemporaryFile(mode="w", delete=False, encoding="utf-8") as f:
        f.write("\n".join(requirements))
        return Path(f.name)


def test_get_requirements(mocker: MockerFixture) -> None:
    """Test that get_requirements returns the expected dictionary of requirements."""
    requirements = get_requirements("anomalib")
    assert isinstance(requirements, dict)
    assert len(requirements) > 0
    for reqs in requirements.values():
        assert isinstance(reqs, list)
        for req in reqs:
            assert isinstance(req, Requirement)
    mocker.patch("anomalib.cli.utils.installation.requires", return_value=None)
    assert get_requirements() == {}


def test_parse_requirements() -> None:
    """Test that parse_requirements returns the expected tuple of requirements."""
    requirements = [
        Requirement("torch==2.0.0"),
        Requirement("onnx>=1.8.1"),
    ]
    torch_req, other_reqs = parse_requirements(requirements)
    assert isinstance(torch_req, str)
    assert isinstance(other_reqs, list)
    assert torch_req == "torch==2.0.0"
    assert other_reqs == ["onnx>=1.8.1"]

    requirements = [
        Requirement("torch<=2.0.1, >=1.8.1"),
    ]
    torch_req, other_reqs = parse_requirements(requirements)
    assert torch_req == "torch<=2.0.1,>=1.8.1"
    assert other_reqs == []

    requirements = [
        Requirement("onnx>=1.8.1"),
    ]
    with pytest.raises(ValueError, match=r"Could not find torch requirement."):
        parse_requirements(requirements)


def test_get_cuda_version_with_version_file(mocker: MockerFixture, tmp_path: Path) -> None:
    """Test that get_cuda_version returns the expected CUDA version when version file exists."""
    tmp_path = tmp_path / "cuda"
    tmp_path.mkdir()
    mocker.patch.dict(os.environ, {"CUDA_HOME": str(tmp_path)})
    version_file = tmp_path / "version.json"
    version_file.write_text('{"cuda": {"version": "11.2.0"}}')
    assert get_cuda_version() == "11.2"


def test_get_cuda_version_with_nvcc(mocker: MockerFixture) -> None:
    """Test that get_cuda_version returns the expected CUDA version when nvcc is available."""
    mocker.patch("anomalib.cli.utils.installation.Path.exists", return_value=False)
    mock_run = mocker.patch("anomalib.cli.utils.installation.subprocess.run")
    mock_run.return_value.stdout = "Build cuda_11.2.r11.2/compiler.00000_0"
    assert get_cuda_version() == "11.2"

    mock_run = mocker.patch("anomalib.cli.utils.installation.subprocess.run")
    mock_run.side_effect = FileNotFoundError
    assert get_cuda_version() is None


def test_update_cuda_version_with_available_torch_cuda_build() -> None:
    """Test that update_cuda_version_with_available_torch_cuda_build returns the expected CUDA version."""
    assert update_cuda_version_with_available_torch_cuda_build("11.1", "2.10.0") == "12.6"
    assert update_cuda_version_with_available_torch_cuda_build("11.8", "2.10.0") == "12.6"
    assert update_cuda_version_with_available_torch_cuda_build("12.6", "2.10.0") == "12.6"
    assert update_cuda_version_with_available_torch_cuda_build("13.0", "2.10.0") == "13.0"
    assert update_cuda_version_with_available_torch_cuda_build("13.0", "2.11.0") == "13.0"
    assert update_cuda_version_with_available_torch_cuda_build("13.2", "2.14.0") == "13.2"
    # Gaps in the supported list: 12.8 isn't built for torch 2.12, so use the closest lower build.
    assert update_cuda_version_with_available_torch_cuda_build("12.8", "2.12.0") == "12.6"
    assert update_cuda_version_with_available_torch_cuda_build("13.1", "2.12.0") == "13.0"
    # Newer than every build: use the newest; compare numerically, not as strings.
    assert update_cuda_version_with_available_torch_cuda_build("13.10", "2.12.0") == "13.2"


def test_get_cuda_suffix() -> None:
    """Test the get_cuda_suffix function."""
    assert get_cuda_suffix(cuda_version="11.2") == "cu112"
    assert get_cuda_suffix(cuda_version="11.8") == "cu118"


def test_get_hardware_suffix(mocker: MockerFixture) -> None:
    """Test the behavior of the get_hardware_suffix function."""
    mocker.patch("anomalib.cli.utils.installation.get_cuda_version", return_value="11.2")
    assert get_hardware_suffix() == "cu112"

    mocker.patch("anomalib.cli.utils.installation.get_cuda_version", return_value="12.6")
    assert get_hardware_suffix(with_available_torch_build=True, torch_version="2.10.0") == "cu126"

    with pytest.raises(ValueError, match=r"``torch_version`` must be provided"):
        get_hardware_suffix(with_available_torch_build=True)

    mocker.patch("anomalib.cli.utils.installation.get_cuda_version", return_value=None)
    assert get_hardware_suffix() == "cpu"


def test_get_torch_install_args(mocker: MockerFixture) -> None:
    """Test that get_torch_install_args returns the expected install arguments."""
    requirement = Requirement("torch>=2.10.0")
    mocker.patch("anomalib.cli.utils.installation.platform.system", return_value="Linux")
    mocker.patch("anomalib.cli.utils.installation.get_hardware_suffix", return_value="cpu")
    install_args = get_torch_install_args(requirement)
    expected_args = [
        "--extra-index-url",
        "https://download.pytorch.org/whl/cpu",
        "torch>=2.10.0",
        "torchvision>=0.25.0",
    ]
    for arg in expected_args:
        assert arg in install_args

    requirement = Requirement("torch>=2.11.0")
    mocker.patch("anomalib.cli.utils.installation.get_hardware_suffix", return_value="cu130")
    install_args = get_torch_install_args(requirement)
    expected_args = [
        "--extra-index-url",
        "https://download.pytorch.org/whl/cu130",
    ]
    for arg in expected_args:
        assert arg in install_args

    requirement = Requirement("torch==2.10.0")
    mocker.patch("anomalib.cli.utils.installation.get_hardware_suffix", return_value="cu126")
    expected_args = [
        "--extra-index-url",
        "https://download.pytorch.org/whl/cu126",
        "torch==2.10.0",
        "torchvision==0.25.0",
    ]
    install_args = get_torch_install_args(requirement)
    for arg in expected_args:
        assert arg in install_args

    install_args = get_torch_install_args("torch")
    assert install_args == ["torch"]

    mocker.patch("anomalib.cli.utils.installation.platform.system", return_value="Darwin")
    requirement = Requirement("torch==2.8.0")
    install_args = get_torch_install_args(requirement)
    assert install_args == ["torch==2.8.0"]

    mocker.patch("anomalib.cli.utils.installation.platform.system", return_value="Unknown")
    with pytest.raises(RuntimeError, match=r"Unsupported OS: Unknown"):
        get_torch_install_args(requirement)
