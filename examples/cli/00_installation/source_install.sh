#!/usr/bin/env bash
# shellcheck shell=bash

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# This script demonstrates how to install anomalib from source
# starting with virtual environment setup.

echo "=== Installing Anomalib from source ==="

echo -e "\n1. Create and activate a virtual environment"
echo "# Create a new virtual environment"
echo "$ python -m venv .venv"
# python -m venv .venv

echo "# Activate the virtual environment (Linux/macOS)"
echo "$ source .venv/bin/activate"
# source .venv/bin/activate

echo "# Activate the virtual environment (Windows)"
echo "$ .venv\\Scripts\\activate"

echo -e "\n2. Clone the repository"
echo "$ git clone https://github.com/openvinotoolkit/anomalib.git"
# git clone https://github.com/openvinotoolkit/anomalib.git

echo "$ cd anomalib"
# cd anomalib

echo -e "\n3. Install in development mode"
echo "# Install the base package in development mode"
echo "$ pip install -e ."
# pip install -e .

echo -e "\n4. Install additional dependencies"
echo "# You can use either pip or anomalib install"

echo -e "\n4a. Using pip:"
echo "# Install full dependencies"
echo "$ pip install -e .[full]"
# pip install -e .[full]

echo "# Install development dependencies"
echo "$ pip install -e .[dev]"
# pip install -e .[dev]

echo "# Install OpenVINO dependencies"
echo "$ pip install -e .[openvino]"
# pip install -e .[openvino]

echo -e "\n4b. Using anomalib install:"
echo "# Install full dependencies"
echo "$ anomalib install --option full"
# anomalib install --option full

echo "# Install development dependencies"
echo "$ anomalib install --option dev"
# anomalib install --option dev

echo "# Install OpenVINO dependencies"
echo "$ anomalib install --option openvino"
# anomalib install --option openvino

echo -e "\n5. Verify the installation"
echo "$ python -c 'import anomalib; print(f\"Anomalib version: {anomalib.__version__}\")'"
# python -c 'import anomalib; print(f"Anomalib version: {anomalib.__version__}")'

echo -e "\n=== Example Installation Output ==="
echo '
❯ python -m venv .venv
❯ source .venv/bin/activate
(.venv) ❯ git clone https://github.com/openvinotoolkit/anomalib.git
Cloning into '"'"'anomalib'"'"'...
remote: Enumerating objects: 9794, done.
remote: Counting objects: 100% (2052/2052), done.
remote: Compressing objects: 100% (688/688), done.
remote: Total 9794 (delta 1516), reused 1766 (delta 1349), pack-reused 7742
Receiving objects: 100% (9794/9794), 106.63 MiB | 5.92 MiB/s, done.
Resolving deltas: 100% (6947/6947), done.
(.venv) ❯ cd anomalib
(.venv) ❯ pip install -e .[full]
Installing collected packages: anomalib
  Running setup.py develop for anomalib
Successfully installed anomalib-0.0.0
(.venv) ❯ python -c '"'"'import anomalib; print(f"Anomalib version: {anomalib.__version__}")'"'"'
Anomalib version: 2.0.0'

echo -e "\nNote: The actual installation commands are commented out above."
echo "To install anomalib, uncomment the desired installation command by removing the '#' at the start of the line."
echo "Make sure to activate the virtual environment before running the installation commands."
