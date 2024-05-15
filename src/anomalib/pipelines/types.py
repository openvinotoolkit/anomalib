"""Types."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Any

RUN_RESULTS = Any
GATHERED_RESULTS = Any
PREV_STAGE_RESULT = GATHERED_RESULTS | None
