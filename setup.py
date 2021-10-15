"""
Setup file for anomalib
"""

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
    "Unauthorized copying of any part of the software via any medium is strictly prohibited. "
    "Proprietary and confidential.",
    install_requires=install_requires,
    author="Intel",
    description="anomalib - Anomaly Detection Library",
)
