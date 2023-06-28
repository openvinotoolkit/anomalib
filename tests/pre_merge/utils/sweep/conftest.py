"""Fixtures for the sweep tests."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

pytest_plugins = [
   "tests.pre_merge.deploy.test_inferencer"     # contains generate_results_dir fixture
]
