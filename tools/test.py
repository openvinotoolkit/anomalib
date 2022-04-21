"""Test This script performs inference on the test dataset and saves the output visualizations into a directory."""

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

import warnings
from argparse import ArgumentParser, Namespace

from pytorch_lightning import Trainer

from anomalib.config import get_configurable_parameters
from anomalib.data import get_datamodule
from anomalib.models import get_model
from anomalib.utils.callbacks import get_callbacks


def get_args() -> Namespace:
    """Get CLI arguments.

    Returns:
        Namespace: CLI arguments.
    """
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="stfpm", help="Name of the algorithm to train/test")
    # --model_config_path will be deprecated in 0.2.8 and removed in 0.2.9
    parser.add_argument("--model_config_path", type=str, required=False, help="Path to a model config file")
    parser.add_argument("--config", type=str, required=False, help="Path to a model config file")
    parser.add_argument("--weight_file", type=str, default="weights/model.ckpt")

    args = parser.parse_args()
    if args.model_config_path is not None:
        warnings.warn(
            message="--model_config_path will be deprecated in v0.2.8 and removed in v0.2.9. Use --config instead.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        args.config = args.model_config_path

    return args


def test():
    """Test an anomaly classification and segmentation model that is initially trained via `tools/train.py`.

    The script is able to write the results into both filesystem and a logger such as Tensorboard.
    """
    args = get_args()
    config = get_configurable_parameters(
        model_name=args.model,
        config_path=args.config,
        weight_file=args.weight_file,
    )

    datamodule = get_datamodule(config)
    model = get_model(config)

    callbacks = get_callbacks(config)

    trainer = Trainer(callbacks=callbacks, **config.trainer)
    trainer.test(model=model, datamodule=datamodule)


if __name__ == "__main__":
    test()
