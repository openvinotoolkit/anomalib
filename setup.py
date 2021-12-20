"""Setup file for anomalib."""

# Copyright (C) 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.
from pathlib import Path
from typing import List

from setuptools import find_packages, setup


def get_version() -> str:
    """Get version from `anomalib.__init__`.

    Version is stored in the main __init__ module in `anomalib`.
    The varible storing the version is `__version__`. This function
    reads `__init__` file, checks `__version__ variable and return
    the value assigned to it.

    Example:
        >>> # Assume that __version__ = "0.2.1"
        >>> get_version()
        "0.2.1"

    Returns:
        str: Version number of `anomalib` package.
    """

    with open(Path.cwd() / "anomalib" / "__init__.py", "r", encoding="utf8") as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip()
            if line.startswith("__version__"):
                version = line.replace("__version__ = ", "")

    return version


def get_required_packages(requirement_files: List[str]) -> List[str]:
    """Get packages from requirements.txt file.

    This function returns list of required packages from requirement files.

    Args:
        requirement_files (List[str]): txt files that contains list of required
            packages.

    Example:
        >>> get_required_packages(requirement_files=["openvino"])
        ['onnx>=1.8.1', 'networkx~=2.5', 'openvino-dev==2021.4.1', ...]

    Returns:
        List[str]: List of required packages
    """

    required_packages: List[str] = []

    for requirement_file in requirement_files:
        with open(f"requirements/{requirement_file}.txt", "r", encoding="utf8") as file:
            for line in file:
                package = line.strip()
                if package and not package.startswith(("#", "-f")):
                    required_packages.append(package)

    return required_packages


VERSION = get_version()
INSTALL_REQUIRES = get_required_packages(requirement_files=["base"])
EXTRAS_REQUIRE = {
    "dev": get_required_packages(requirement_files=["dev", "docs"]),
    "openvino": get_required_packages(requirement_files=["openvino"]),
    "full": get_required_packages(requirement_files=["dev", "docs", "openvino"]),
}

setup(
    name="anomalib",
    # TODO: https://github.com/openvinotoolkit/anomalib/issues/36
    version="0.2.2",
    author="Intel OpenVINO",
    author_email="help@openvino.intel.com",
    description="anomalib - Anomaly Detection Library",
    url="",
    license="Copyright (c) Intel - All Rights Reserved. "
    'Licensed under the Apache License, Version 2.0 (the "License")'
    "See LICENSE file for more details.",
    python_requires=">=3.8",
    packages=find_packages("."),
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    package_data={"": ["config.yaml"]},
)
