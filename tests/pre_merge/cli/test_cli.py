"""Test Models using the new CLI."""

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

import subprocess
import tempfile
from pathlib import Path

import pytest

MODELS = ["cflow", "dfkde", "dfm", "ganomaly", "padim", "patchcore", "stfpm"]


@pytest.mark.parametrize("model", MODELS)
def test_train_model(model):
    """Test if the configs are parsed properly and train/test pipeline works."""
    with tempfile.TemporaryDirectory() as default_root_dir:
        config = f"anomalib/models/{model}/config/config.yaml"
        args = [
            "python",
            "trainer.py",
            "--config",
            config,
            "--save_images",
            "False",
            "--trainer.default_root_dir",
            default_root_dir,
            "--trainer.max_epochs",
            "1",
        ]
        # Run the trainer to perform train and test subcommands.
        result = subprocess.run(args, capture_output=True, check=True)

        # After running the trainer, a model weight file, called "model.ckpt", is expected
        # to be located in `default_root_dir`. The weight file is created within a directory
        # created with a time-stamp. Accessing to the timestamp would be hard, so instead,
        # we recursively check `default_root_dir` to find `model.ckpt` file.
        assert "model.ckpt" in [i.name for i in Path(default_root_dir).glob("**/*.ckpt")]
