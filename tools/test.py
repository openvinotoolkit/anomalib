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

from argparse import ArgumentParser, Namespace

from pytorch_lightning import Trainer

from anomalib.config import get_configurable_parameters
from anomalib.core.callbacks import get_callbacks
from anomalib.data import get_datamodule
from anomalib.models import get_model


def get_args() -> Namespace:
    """Get CLI arguments.

    Returns:
        Namespace: CLI arguments.
    """
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="stfpm", help="Name of the algorithm to train/test")
    parser.add_argument("--model_config_path", type=str, required=False, help="Path to a model config file")
    parser.add_argument("--weight_file", type=str, default="weights/model.ckpt")
    parser.add_argument("--openvino", type=bool, default=False)

    return parser.parse_args()


def test():
    """Test an anomaly classification and segmentation model that is initially trained via `tools/train.py`.

    The script is able to write the results into both filesystem and a logger such as Tensorboard.
    """
    args = get_args()
    config = get_configurable_parameters(
        model_name=args.model,
        model_config_path=args.model_config_path,
        weight_file=args.weight_file,
        openvino=args.openvino,
    )

    datamodule = get_datamodule(config)
    model = get_model(config)

    callbacks = get_callbacks(config)

    trainer = Trainer(callbacks=callbacks, **config.trainer)
    trainer.test(model=model, datamodule=datamodule)


if __name__ == "__main__":
    test()
