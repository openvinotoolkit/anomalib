"""Anomalib CLI."""

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

import os
from datetime import datetime
from importlib import import_module

from omegaconf.omegaconf import OmegaConf
from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.cli import LightningArgumentParser, LightningCLI

from anomalib.utils.callbacks import (
    CdfNormalizationCallback,
    CompressModelCallback,
    LoadModelCallback,
    MinMaxNormalizationCallback,
    SaveToCSVCallback,
    TilerCallback,
    TimerCallback,
    VisualizerCallback,
)


class AnomalibCLI(LightningCLI):
    """Anomalib CLI."""

    def __init__(self):
        super().__init__(
            model_class=LightningModule,
            datamodule_class=LightningDataModule,
            subclass_mode_model=True,
            subclass_mode_data=True,
            seed_everything_default=0,
            save_config_callback=None,
            run=False,
        )

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        """Add default arguments.

        Args:
            parser (LightningArgumentParser): Lightning Argument Parser.
        """
        parser.add_argument("--save_to_csv", type=bool, default=False, help="Save results to a CSV")
        parser.add_argument("--save_images", type=bool, default=True, help="Flag to save output images locally.")
        # TODO: https://github.com/openvinotoolkit/anomalib/issues/19
        # TODO: https://github.com/openvinotoolkit/anomalib/issues/20
        parser.add_argument("--openvino", type=bool, default=False, help="Export to ONNX and OpenVINO IR format.")
        parser.add_argument("--nncf", type=str, help="Path to NNCF config to enable quantized training.")

        # NOTE: MyPy gives the following error:
        # Argument 1 to "add_lightning_class_args" of "LightningArgumentParser"
        # has incompatible type "Type[TilerCallback]"; expected "Union[Type[Trainer],
        # Type[LightningModule], Type[LightningDataModule]]"  [arg-type]
        parser.add_lightning_class_args(TilerCallback, "tiling")  # type: ignore
        parser.set_defaults({"tiling.enable": False})

    def before_instantiate_classes(self) -> None:
        """Modify the configuration to properly instantiate classes."""

        # 0. Get the root dir.
        root_dir = (
            self.config["trainer"]["default_root_dir"] if self.config["trainer"]["default_root_dir"] else "./results"
        )
        model_name = self.config["model"]["class_path"].split(".")[-1].lower()
        data_name = self.config["data"]["class_path"].split(".")[-1].lower()
        category = self.config["data"]["init_args"]["category"] if self.config["data"]["init_args"].keys() else ""
        time_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        default_root_dir = os.path.join(root_dir, model_name, data_name, category, time_stamp)
        os.makedirs(default_root_dir, exist_ok=True)
        self.config["trainer"]["default_root_dir"] = default_root_dir

        callbacks = []

        # 1. Model Checkpoint.
        monitor = None
        mode = "max"
        if self.config["trainer"]["callbacks"] is not None:
            # If trainer has callbacks defined from the config file, they have the
            # following format:
            # [{'class_path': 'pytorch_lightning.ca...lyStopping', 'init_args': {...}}]
            callbacks = self.config["trainer"]["callbacks"]

            # Convert to the following format to get `monitor` and `mode` variables
            # {'EarlyStopping': {'monitor': 'pixel_AUROC', 'mode': 'max', ...}}
            callback_args = {c["class_path"].split(".")[-1]: c["init_args"] for c in callbacks}
            if "EarlyStopping" in callback_args:
                monitor = callback_args["EarlyStopping"]["monitor"]
                mode = callback_args["EarlyStopping"]["mode"]

        checkpoint = ModelCheckpoint(
            dirpath=os.path.join(self.config["trainer"]["default_root_dir"], "weights"),
            filename="model",
            monitor=monitor,
            mode=mode,
            auto_insert_metric_name=False,
        )
        callbacks.append(checkpoint)

        # 2. LoadModel from Checkpoint.
        if self.config["trainer"]["resume_from_checkpoint"]:
            load_model = LoadModelCallback(self.config["trainer"]["resume_from_checkpoint"])
            callbacks.append(load_model)

        # 3. Add timing to the pipeline.
        callbacks.append(TimerCallback())

        # 4. Save Results to CSV
        if self.config["save_to_csv"]:
            callbacks.append(SaveToCSVCallback())

        # 5. Normalization.
        normalization = self.config["model"]["init_args"]["normalization"]
        if normalization:
            if normalization == "min_max":
                callbacks.append(MinMaxNormalizationCallback())
            elif normalization == "cdf":
                callbacks.append(CdfNormalizationCallback())
            else:
                raise ValueError(
                    f"Unknown normalization type {normalization}. \n" "Available types are either min_max or cdf"
                )

        # 6. Visualization
        if self.config["save_images"]:
            if self.config["model"]["init_args"]["task"] == "segmentation":
                # NOTE: Currently only segmentation tasks are supported for visualizaition.
                # NOTE: When ready, add wandb logger here.
                callbacks.append(VisualizerCallback(loggers=["local"]))

        # TODO: https://github.com/openvinotoolkit/anomalib/issues/19
        if self.config["openvino"] and self.config["nncf"]:
            raise ValueError("OpenVINO and NNCF cannot be set simultaneously.")

        # 7. Export to OpenVINO
        if self.config["openvino"]:
            callbacks.append(
                CompressModelCallback(
                    input_size=self.config["data"]["init_args"]["image_size"],
                    dirpath=os.path.join(self.config["trainer"]["default_root_dir"], "compressed"),
                    filename="model",
                )
            )
        if self.config["nncf"]:
            if os.path.isfile(self.config["nncf"]) and self.config["nncf"].endswith(".yaml"):
                nncf_module = import_module("anomalib.core.callbacks.nncf_callback")
                nncf_callback = getattr(nncf_module, "NNCFCallback")
                callbacks.append(
                    nncf_callback(
                        config=OmegaConf.load(self.config["nncf"]),
                        dirpath=os.path.join(self.config["trainer"]["default_root_dir"], "compressed"),
                        filename="model",
                    )
                )
            else:
                raise ValueError(
                    f"--nncf expects a path to nncf config which is a yaml file, but got {self.config['nncf']}"
                )

        self.config["trainer"]["callbacks"] = callbacks
