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

from typing import List

from setuptools import find_packages, setup

import anomalib


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


if __name__ == "__main__":
    base_packages = get_required_packages(requirement_files=["base", "openvino"])
    dev_packages = get_required_packages(requirement_files=["dev", "docs"])
    openvino_packages = get_required_packages(requirement_files=["openvino"])

    setup(
        name="anomalib",
        version=anomalib.__version__,
        packages=find_packages(include=["anomalib", "anomalib.*"]),
        url="",
        license="Copyright (c) Intel - All Rights Reserved. "
        'Licensed under the Apache License, Version 2.0 (the "License")'
        "See LICENSE file for more details.",
        install_requires=base_packages,
        extras_require={
            "dev": dev_packages,
            "openvino": openvino_packages,
        },
        author="Intel",
        description="anomalib - Anomaly Detection Library",
        package_data={"": ["config.yaml"]},
    )
