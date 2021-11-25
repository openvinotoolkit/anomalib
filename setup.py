"""
Setup file for anomalib
"""

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

from setuptools import find_packages, setup

import anomalib

install_requires = []
with open("requirements/requirements.txt", "r", encoding="utf8") as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#"):
            install_requires.append(line)

setup(
    name="anomalib",
    version=anomalib.__version__,
    packages=find_packages(include=["anomalib", "anomalib.*"]),
    url="",
    license="Copyright (c) Intel - All Rights Reserved. "
    'Licensed under the Apache License, Version 2.0 (the "License")'
    "See LICENSE file for more details.",
    install_requires=install_requires,
    author="Intel",
    description="anomalib - Anomaly Detection Library",
)
