"""Anomalib installation util functions."""

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
    """Get requirements of module from importlib.metadata.

    This function returns list of required packages from importlib_metadata.

    Example:
        >>> get_requirements("anomalib")
        {
            "base": ["jsonargparse==4.27.1", ...],
            "core": ["torch==2.1.1", ...],
            ...
        }

    Returns:
        dict[str, list[Requirement]]: List of required packages for each optional-extras.
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
    """Parse requirements and returns torch and other requirements.

    Args:
        requirements (list[Requirement]): List of requirements.
        skip_torch (bool): Whether to skip torch requirement. Defaults to False.

    Raises:
        ValueError: If torch requirement is not found.

    Examples:
        >>> requirements = [
        ...     Requirement.parse("torch==1.13.0"),
        ...     Requirement.parse("onnx>=1.8.1"),
        ... ]
        >>> parse_requirements(requirements=requirements)
        (Requirement.parse("torch==1.13.0"),
        Requirement.parse("onnx>=1.8.1"))

    Returns:
        tuple[str, list[str], list[str]]: Tuple of torch and other requirements.
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

    Examples:
        >>> # Assume that CUDA version is 11.2
        >>> get_cuda_version()
        "11.2"

        >>> # Assume that CUDA is not installed on the system
        >>> get_cuda_version()
        None

    Returns:
        str | None: CUDA version installed on the system.
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
    """Update the installed CUDA version with the highest supported CUDA version by PyTorch.

    Args:
        cuda_version (str): The installed CUDA version.
        torch_version (str): The PyTorch version.

    Raises:
        Warning: If the installed CUDA version is not supported by PyTorch.

    Examples:
        >>> update_cuda_version_with_available_torch_cuda_builds("11.1", "1.13.0")
        "11.6"

        >>> update_cuda_version_with_available_torch_cuda_builds("11.7", "1.13.0")
        "11.7"

        >>> update_cuda_version_with_available_torch_cuda_builds("11.8", "1.13.0")
        "11.7"

        >>> update_cuda_version_with_available_torch_cuda_builds("12.1", "2.0.1")
        "11.8"

    Returns:
        str: The updated CUDA version.
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
        cuda_version (str): CUDA version installed on the system.

    Note:
        The CUDA version of PyTorch is not always the same as the CUDA version
            that is installed on the system. For example, the latest PyTorch
            version (1.10.0) supports CUDA 11.3, but the latest CUDA version
            that is available for download is 11.2. Therefore, we need to use
            the latest available CUDA version for PyTorch instead of the CUDA
            version that is installed on the system. Therefore, this function
            shoudl be regularly updated to reflect the latest available CUDA.

    Examples:
        >>> get_cuda_suffix(cuda_version="11.2")
        "cu112"

        >>> get_cuda_suffix(cuda_version="11.8")
        "cu118"

    Returns:
        str: CUDA suffix for PyTorch or mmX version.
    """
    return f"cu{cuda_version.replace('.', '')}"


def get_hardware_suffix(with_available_torch_build: bool = False, torch_version: str | None = None) -> str:
    """Get hardware suffix for PyTorch or mmX versions.

    Args:
        with_available_torch_build (bool): Whether to use the latest available
            PyTorch build or not. If True, the latest available PyTorch build
            will be used. If False, the installed PyTorch build will be used.
            Defaults to False.
        torch_version (str | None): PyTorch version. This is only used when the
            ``with_available_torch_build`` is True.

    Examples:
        >>> # Assume that CUDA version is 11.2
        >>> get_hardware_suffix()
        "cu112"

        >>> # Assume that CUDA is not installed on the system
        >>> get_hardware_suffix()
        "cpu"

        Assume that that installed CUDA version is 12.1.
        However, the latest available CUDA version for PyTorch v2.0 is 11.8.
        Therefore, we use 11.8 instead of 12.1. This is because PyTorch does not
        support CUDA 12.1 yet. In this case, we could correct the CUDA version
        by setting `with_available_torch_build` to True.

        >>> cuda_version = get_cuda_version()
        "12.1"
        >>> get_hardware_suffix(with_available_torch_build=True, torch_version="2.0.1")
        "cu118"

    Returns:
        str: Hardware suffix for PyTorch or mmX version.
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
    """Get the install arguments for Torch requirement.

    This function will return the install arguments for the Torch requirement
    and its corresponding torchvision requirement.

    Args:
        requirement (str | Requirement): The torch requirement.

    Raises:
        RuntimeError: If the OS is not supported.

    Example:
        >>> from pkg_resources import Requirement
        >>> requriment = "torch>=1.13.0"
        >>> get_torch_install_args(requirement)
        ['--extra-index-url', 'https://download.pytorch.org/whl/cpu',
        'torch>=1.13.0', 'torchvision==0.14.0']

    Returns:
        list[str]: The install arguments.
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
