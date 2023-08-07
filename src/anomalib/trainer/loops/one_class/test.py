"""Test loop for one-class classification."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from lightning.fabric.wrappers import _FabricDataLoader, _unwrap_objects
from lightning.pytorch.utilities.types import STEP_OUTPUT
from lightning_utilities import apply_to_collection

from anomalib import trainer
from anomalib.trainer.loops.base import BaseLoop

class TestLoop(BaseLoop):
    