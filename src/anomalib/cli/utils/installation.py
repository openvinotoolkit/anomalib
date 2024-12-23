"""Anomalib installation utilities.

This module provides utilities for managing Anomalib package installation,
including dependency resolution and hardware-specific package selection.
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import platform
import re
from importlib.metadata import requires
from pathlib import Path
from warnings import warn

from pkg_resources import Requirement

AVAILABLE_TORCH_VERSIONS = {
    "2.0.0": {"torchvision": "0.15.1", "cuda": ("11.7", "11.8")},
    "2.0.1": {"torchvision": "0.15.2", "cuda": ("11.7", "11.8")},
    "2.1.1": {"torchvision": "0.16.1", "cuda": ("11.8", "12.1")},
    "2.1.2": {"torchvision": "0.16.2", "cuda": ("11.8", "12.1")},
    "2.2.0": {"torchvision": "0.16.2", "cuda": ("11.8", "12.1")},
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
        _requirement_name = requirement_extra[0]
        _requirement = Requirement.parse(_requirement_name)
        if extra in extra_requirement:
            extra_requirement[extra].append(_requirement)
        else:
            extra_requirement[extra] = [_requirement]
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
        if requirement.unsafe_name == "torch":
            torch_requirement = str(requirement)
            if len(requirement.specs) > 1:
                warn(
                    "requirements.txt contains. Please remove other versions of torch from requirements.",
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
        result = os.popen(cmd="nvcc --version")
        output = result.read()

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
        update_cuda_version_with_available_torch_cuda_build("12.1", "2.0.1")
        # Returns: "11.8"  # PyTorch 2.0.1 only supports up to CUDA 11.8
        ```
    """
    max_supported_cuda = max(AVAILABLE_TORCH_VERSIONS[torch_version]["cuda"])
    min_supported_cuda = min(AVAILABLE_TORCH_VERSIONS[torch_version]["cuda"])
    bounded_cuda_version = max(min(cuda_version, max_supported_cuda), min_supported_cuda)

    if cuda_version != bounded_cuda_version:
        warn(
            f"Installed CUDA version is v{cuda_version}. \n"
            f"v{min_supported_cuda} <= Supported CUDA version <= v{max_supported_cuda}.\n"
            f"This script will use CUDA v{bounded_cuda_version}.\n"
            f"However, this may not be safe, and you are advised to install the correct version of CUDA.\n"
            f"For more details, refer to https://pytorch.org/get-started/locally/",
            stacklevel=2,
        )
        cuda_version = bounded_cuda_version

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
        requirement = "torch>=2.0.0"
        get_torch_install_args(requirement)
        # Returns:
        [
            '--extra-index-url',
            'https://download.pytorch.org/whl/cu118',
            'torch>=2.0.0',
            'torchvision==0.15.1'
        ]
        ```

    Test:
        >>> args = get_torch_install_args("torch>=2.0.0")
        >>> isinstance(args, list)
        True
        >>> all(isinstance(arg, str) for arg in args)
        True
        >>> any('torch' in arg for arg in args)
        True
    """
    if isinstance(requirement, str):
        requirement = Requirement.parse(requirement)

    # NOTE: This does not take into account if the requirement has multiple versions
    #   such as torch<2.0.1,>=1.13.0
    if len(requirement.specs) < 1:
        return [str(requirement)]
    select_spec_idx = 0
    for i, spec in enumerate(requirement.specs):
        if "=" in spec[0]:
            select_spec_idx = i
            break
    operator, version = requirement.specs[select_spec_idx]
    if version not in AVAILABLE_TORCH_VERSIONS:
        version = max(AVAILABLE_TORCH_VERSIONS.keys())
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
