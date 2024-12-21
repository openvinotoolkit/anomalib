#!/usr/bin/env bash
# shellcheck shell=bash

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# This script demonstrates how to install anomalib using pip
# with different dependency options.

echo "=== Installing Anomalib using pip ==="

echo -e "\n1. Base Installation"
echo "# Install the base package with minimal dependencies"
echo "$ pip install anomalib"
# pip install anomalib

echo -e "\n2. Install with OpenVINO dependencies"
echo "$ pip install anomalib[openvino]"
# pip install anomalib[openvino]

echo -e "\n3. Install with full dependencies"
echo "$ pip install anomalib[full]"
# pip install anomalib[full]

echo -e "\n4. Install with development dependencies"
echo "$ pip install anomalib[dev]"
# pip install anomalib[dev]

echo -e "\n5. Install with multiple dependency groups"
echo "$ pip install anomalib[openvino,dev]"
# pip install anomalib[openvino,dev]

echo -e "\n=== Verifying Installation ==="
echo "$ python -c 'import anomalib; print(f\"Anomalib version: {anomalib.__version__}\")'"
# python -c 'import anomalib; print(f"Anomalib version: {anomalib.__version__}")'

echo -e "\nNote: The actual installation commands are commented out above."
echo "To install anomalib, uncomment the desired installation command by removing the '#' at the start of the line."
