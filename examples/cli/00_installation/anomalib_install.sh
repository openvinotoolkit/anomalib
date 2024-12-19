#!/usr/bin/env bash
# shellcheck shell=bash

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# This script demonstrates how to use the anomalib installer
# to install different dependency options.

echo "=== Installing Anomalib using the anomalib installer ==="

echo -e "\n1. Base Installation"
echo "# First, install the base package"
echo "$ pip install anomalib"
# pip install anomalib

echo -e "\n=== Anomalib Installer Help ==="
echo "$ anomalib install -h"
echo '
╭─ Arguments ───────────────────────────────────────────────────────────────────╮
│ Usage: anomalib install [-h] [-v] [--option {core,full,openvino,dev}]         │
│                                                                               │
│ Install the full-package for anomalib.                                        │
│                                                                               │
│ Options:                                                                      │
│   -h, --help            Show this help message and exit.                      │
│   -v, --verbose         Show verbose output during installation.              │
│   --option {core,full,openvino,dev}                                           │
│                         Installation option to use. Options are:              │
│                         - core: Install only core dependencies                │
│                         - full: Install all dependencies                      │
│                         - openvino: Install OpenVINO dependencies             │
│                         - dev: Install development dependencies               │
│                         (default: full)                                       │
╰───────────────────────────────────────────────────────────────────────────────╯'

echo -e "\n=== Installation Options ==="

echo -e "\n2. Install core dependencies only"
echo "# For basic training and evaluation via Torch and Lightning"
echo "$ anomalib install --option core"
# anomalib install --option core

echo -e "\n3. Install full dependencies"
echo "# Includes all optional dependencies"
echo "$ anomalib install --option full"
# anomalib install --option full

echo -e "\n4. Install OpenVINO dependencies"
echo "# For edge deployment with smaller wheel size"
echo "$ anomalib install --option openvino"
# anomalib install --option openvino

echo -e "\n5. Install development dependencies"
echo "# For contributing to anomalib"
echo "$ anomalib install --option dev"
# anomalib install --option dev

echo -e "\n6. Install with verbose output"
echo "# Shows detailed installation progress"
echo "$ anomalib install -v"
# anomalib install -v

echo -e "\n=== Example Installation Output ==="
echo '
❯ anomalib install --option full
Installing anomalib with full dependencies...
Successfully installed anomalib and all dependencies.'

echo -e "\nNote: The actual installation commands are commented out above."
echo "To install anomalib, uncomment the desired installation command by removing the '#' at the start of the line."
