# Copyright (C) 2024-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Anomalib installation utilities.

This module provides utilities for managing Anomalib package installation,
including dependency resolution and hardware-specific package selection.
"""

from __future__ import annotations

import json
import os
import platform
import re
import subprocess  # nosec B404 - hardcoded nvcc invocation only
from importlib.metadata import requires
from pathlib import Path
from warnings import warn

from packaging.requirements import Requirement
from packaging.version import Version

AVAILABLE_TORCH_VERSIONS = {
    # NOTE: Minimum torch>=2.10.0 required as of anomalib 2.7.0 (dynamo ONNX exporter).
    "2.10.0": {"torchvision": "0.25.0", "cuda": ("12.6", "12.8", "13.0")},
    "2.11.0": {"torchvision": "0.26.0", "cuda": ("12.6", "12.8", "13.0")},
    "2.12.0": {"torchvision": "0.27.0", "cuda": ("12.6", "13.0", "13.2")},
    "2.13.0": {"torchvision": "0.28.0", "cuda": ("12.6", "13.0", "13.2")},
    "2.14.0": {"torchvision": "0.29.0", "cuda": ("12.6", "13.0", "13.2")},
}


def get_requirements(module: str = "anomalib") -> dict[str, list[Requirement]]:
    """Get package requirements from importlib.metadata.

    Args:
        module (str): Name of the module to get requirements for. Defaults to "anomalib".

    Returns:
        dict[str, list[Requirement]]: Dictionary mapping requirement groups to their
            package requirements.

    Example:
        ```python
        get_requirements("anomalib")
        # Returns:
        {
            "base": ["jsonargparse==4.27.1", ...],
            "core": ["torch==2.1.1", ...],
            ...
        }
        ```

    Test:
        >>> result = get_requirements("anomalib")
        >>> isinstance(result, dict)
        True
        >>> all(isinstance(v, list) for v in result.values())
        True
    """
    requirement_list: list[str] | None = requires(module)
    extra_requirement: dict[str, list[Requirement]] = {}
    if requirement_list is None:
        return extra_requirement
    for requirement in requirement_list:
        extra = "core"
        requirement_extra: list[str] = requirement.replace(" ", "").split(";")
        if isinstance(requirement_extra, list) and len(requirement_extra) > 1:
            extra = requirement_extra[-1].split("==")[-1].strip("'\"")
        requirement_name_ = requirement_extra[0]
        requirement_ = Requirement(requirement_name_)
        if extra in extra_requirement:
            extra_requirement[extra].append(requirement_)
        else:
            extra_requirement[extra] = [requirement_]
    return extra_requirement


def parse_requirements(
    requirements: list[Requirement],
    skip_torch: bool = False,
) -> tuple[str | None, list[str]]:
    """Parse requirements into torch and other requirements.

    Args:
        requirements (list[Requirement]): List of requirements to parse.
        skip_torch (bool): Whether to skip torch requirement. Defaults to False.

    Returns:
        tuple[str | None, list[str]]: Tuple containing:
            - Torch requirement string or None if skipped
            - List of other requirement strings

    Raises:
        ValueError: If torch requirement is not found and skip_torch is False.

    Example:
        ```python
        requirements = [
            Requirement.parse("torch==1.13.0"),
            Requirement.parse("onnx>=1.8.1"),
        ]
        parse_requirements(requirements)
        # Returns: ('torch==1.13.0', ['onnx>=1.8.1'])
        ```

    Test:
        >>> reqs = [Requirement.parse("torch==1.13.0"), Requirement.parse("onnx>=1.8.1")]
        >>> torch_req, other_reqs = parse_requirements(reqs)
        >>> torch_req == "torch==1.13.0"
        True
        >>> other_reqs == ["onnx>=1.8.1"]
        True
    """
    torch_requirement: str | None = None
    other_requirements: list[str] = []

    for requirement in requirements:
        if requirement.name.lower() == "torch":
            torch_requirement = str(requirement)
            specifiers = list(requirement.specifier)
            if len(specifiers) > 1:
                warn(
                    f"Multiple version specifiers for 'torch' detected in requirements.txt "
                    f"(requirement: '{requirement}'). Please specify only one torch version "
                    "and remove any additional torch specifiers from your requirements.",
                    stacklevel=2,
                )

        # Rest of the requirements are task requirements.
        # Other torch-related requirements such as `torchvision` are to be excluded.
        # This is because torch-related requirements are already handled in torch_requirement.
        else:
            # if not requirement.unsafe_name.startswith("torch"):
            other_requirements.append(str(requirement))

    if not skip_torch and not torch_requirement:
        msg = "Could not find torch requirement. Anoamlib depends on torch. Please add torch to your requirements."
        raise ValueError(msg)

    # Get the unique list of the requirements.
    other_requirements = list(set(other_requirements))

    return torch_requirement, other_requirements


def get_cuda_version() -> str | None:
    """Get CUDA version installed on the system.

    Returns:
        str | None: CUDA version string (e.g., "11.8") or None if not found.

    Example:
        ```python
        # System with CUDA 11.8 installed
        get_cuda_version()
        # Returns: "11.8"

        # System without CUDA
        get_cuda_version()
        # Returns: None
        ```

    Test:
        >>> version = get_cuda_version()
        >>> version is None or isinstance(version, str)
        True
        >>> if version is not None:
        ...     version.count('.') == 1 and all(part.isdigit() for part in version.split('.'))
        ...     True
    """
    # 1. Check CUDA_HOME Environment variable
    cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")

    if Path(cuda_home).exists():
        # Check $CUDA_HOME/version.json file.
        version_file = Path(cuda_home) / "version.json"
        if version_file.is_file():
            with Path(version_file).open(encoding="utf-8") as file:
                data = json.load(file)
                cuda_version = data.get("cuda", {}).get("version", None)
                if cuda_version is not None:
                    cuda_version_parts = cuda_version.split(".")
                    return ".".join(cuda_version_parts[:2])
    # 2. 'nvcc --version' check & without version.json case
    try:
        result = subprocess.run(
            # command is a hardcoded literal with no user-controlled input
            ["nvcc", "--version"],  # noqa: S607  # nosec B603, B607
            capture_output=True,
            text=True,
            check=False,
        )
        output = result.stdout

        cuda_version_pattern = r"cuda_(\d+\.\d+)"
        cuda_version_match = re.search(cuda_version_pattern, output)

        if cuda_version_match is not None:
            return cuda_version_match.group(1)
    except OSError:
        msg = "Could not find cuda-version. Instead, the CPU version of torch will be installed."
        warn(msg, stacklevel=2)
    return None


def update_cuda_version_with_available_torch_cuda_build(cuda_version: str, torch_version: str) -> str:
    """Update CUDA version to match PyTorch's supported versions.

    Args:
        cuda_version (str): The installed CUDA version.
        torch_version (str): The PyTorch version.

    Returns:
        str: The updated CUDA version that's compatible with PyTorch.

    Example:
        ```python
        update_cuda_version_with_available_torch_cuda_build("12.8", "2.14.0")
        # Returns: "12.6"  # PyTorch 2.14.0 has no cu128 wheel; cu126 is nearest below.
        ```
    """
    supported = sorted(AVAILABLE_TORCH_VERSIONS[torch_version]["cuda"], key=Version)
    # Supported builds can have gaps (e.g. 12.6, 13.0), so pick the newest build not above
    # the installed version; fall back to the oldest if the installed version is older.
    compatible = [version for version in supported if Version(version) <= Version(cuda_version)]
    selected_cuda_version = compatible[-1] if compatible else supported[0]

    if cuda_version != selected_cuda_version:
        warn(
            f"Installed CUDA version is v{cuda_version}. \n"
            f"Supported CUDA versions for torch {torch_version}: {', '.join(supported)}.\n"
            f"This script will use CUDA v{selected_cuda_version}.\n"
            f"However, this may not be safe, and you are advised to install the correct version of CUDA.\n"
            f"For more details, refer to https://pytorch.org/get-started/locally/",
            stacklevel=2,
        )
        cuda_version = selected_cuda_version

    return cuda_version


def get_cuda_suffix(cuda_version: str) -> str:
    """Get CUDA suffix for PyTorch versions.

    Args:
        cuda_version (str): CUDA version string (e.g., "11.8").

    Returns:
        str: CUDA suffix for PyTorch (e.g., "cu118").

    Example:
        ```python
        get_cuda_suffix("11.8")
        # Returns: "cu118"
        ```

    Test:
        >>> get_cuda_suffix("11.8")
        'cu118'
        >>> get_cuda_suffix("12.1")
        'cu121'
    """
    return f"cu{cuda_version.replace('.', '')}"


def get_hardware_suffix(with_available_torch_build: bool = False, torch_version: str | None = None) -> str:
    """Get hardware suffix for PyTorch package names.

    Args:
        with_available_torch_build (bool): Whether to use available PyTorch builds
            to determine the suffix. Defaults to False.
        torch_version (str | None): PyTorch version to check against. Required if
            with_available_torch_build is True.

    Returns:
        str: Hardware suffix (e.g., "cu118" or "cpu").

    Raises:
        ValueError: If torch_version is not provided when with_available_torch_build is True.

    Example:
        ```python
        # System with CUDA 11.8
        get_hardware_suffix()
        # Returns: "cu118"

        # System without CUDA
        get_hardware_suffix()
        # Returns: "cpu"
        ```

    Test:
        >>> suffix = get_hardware_suffix()
        >>> isinstance(suffix, str)
        True
        >>> suffix in {'cpu'} or suffix.startswith('cu')
        True
    """
    cuda_version = get_cuda_version()
    if cuda_version:
        if with_available_torch_build:
            if torch_version is None:
                msg = "``torch_version`` must be provided when with_available_torch_build is True."
                raise ValueError(msg)
            cuda_version = update_cuda_version_with_available_torch_cuda_build(cuda_version, torch_version)
        hardware_suffix = get_cuda_suffix(cuda_version)
    else:
        hardware_suffix = "cpu"

    return hardware_suffix


def get_torch_install_args(requirement: str | Requirement) -> list[str]:
    """Get pip install arguments for PyTorch packages.

    Args:
        requirement (str | Requirement): The torch requirement specification.

    Returns:
        list[str]: List of pip install arguments.

    Raises:
        RuntimeError: If the OS is not supported.

    Example:
        ```python
        requirement = "torch>=2.10.0"
        get_torch_install_args(requirement)
        # Returns:
        [
            '--extra-index-url',
            'https://download.pytorch.org/whl/cu130',
            'torch>=2.10.0',
            'torchvision>=0.25.0'
        ]
        ```

    Test:
        >>> args = get_torch_install_args("torch>=2.10.0")
        >>> isinstance(args, list)
        True
        >>> all(isinstance(arg, str) for arg in args)
        True
        >>> any('torch' in arg for arg in args)
        True
    """
    if isinstance(requirement, str):
        requirement = Requirement(requirement)

    # NOTE: This does not take into account if the requirement has multiple versions
    #   such as torch<2.0.1,>=1.13.0
    specifiers = list(requirement.specifier)
    if not specifiers:
        return [str(requirement)]
    preferred_operators = ("==", "<=", "<", ">=", ">")
    selected_spec = None
    for operator in preferred_operators:
        for spec in specifiers:
            if spec.operator == operator:
                selected_spec = spec
                break
        if selected_spec is not None:
            break
    if selected_spec is None:
        selected_spec = specifiers[0]
    operator = selected_spec.operator
    version = selected_spec.version
    if version not in AVAILABLE_TORCH_VERSIONS:
        version = max(AVAILABLE_TORCH_VERSIONS.keys(), key=Version)
        warn(
            f"Torch Version will be selected as {version}.",
            stacklevel=2,
        )
    install_args: list[str] = []

    if platform.system() in {"Linux", "Windows"}:
        # Get the hardware suffix (eg., +cpu, +cu116 and +cu118 etc.)
        hardware_suffix = get_hardware_suffix(with_available_torch_build=True, torch_version=version)

        # Create the PyTorch Index URL to download the correct wheel.
        index_url = f"https://download.pytorch.org/whl/{hardware_suffix}"

        torch_version = f"{requirement.name}{operator}{version}"  # eg: torch==1.13.0

        # Get the torchvision version depending on the torch version.
        torchvision_version = AVAILABLE_TORCH_VERSIONS[version]["torchvision"]
        torchvision_requirement = f"torchvision{operator}{torchvision_version}"

        # Return the install arguments.
        install_args += [
            "--extra-index-url",
            index_url,
            torch_version,
            torchvision_requirement,
        ]
    elif platform.system() in {"macos", "Darwin"}:
        torch_version = str(requirement)
        install_args += [torch_version]
    else:
        msg = f"Unsupported OS: {platform.system()}"
        raise RuntimeError(msg)

    return install_args
