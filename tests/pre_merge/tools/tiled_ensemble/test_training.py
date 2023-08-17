"""Test tiled ensemble training script"""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys
from tools.tiled_ensemble.train_ensemble import get_parser, train

sys.path.append("tools")


def test_train():
    """Test train.py."""
    # Test when model key is passed
    args = get_parser().parse_args(
        [
            "--model_config",
            "tests/pre_merge/tools/tiled_ensemble/dummy_padim_config.yaml",
            "--ensemble_config",
            "tests/pre_merge/tools/tiled_ensemble/dummy_ens_config.yaml",
        ]
    )
    train(args)
