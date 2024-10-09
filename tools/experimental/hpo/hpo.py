"""Run HPO."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging

from anomalib.pipelines.hpo import HPO

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    HPO().run()
