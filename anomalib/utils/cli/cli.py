"""Anomalib CLI."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Type, Union

from omegaconf import OmegaConf
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.utilities.cli import (
    LightningArgumentParser,
    LightningCLI,
    SaveConfigCallback,
)

from anomalib.deploy.export import ExportMode
from anomalib.models import AnomalyModule
from anomalib.pre_processing.tiler import TilerDecorator
from anomalib.utils.callbacks import (
    ImageVisualizerCallback,
    MetricsConfigurationCallback,
    PostProcessingConfigurationCallback,
)
from anomalib.utils.loggers import configure_logger

logger = logging.getLogger("anomalib.cli")


class AnomalibCLI(LightningCLI):
    """Implementation of a fully configurable CLI tool for anomalib.

    The advantage of this tool is its flexibility to configure the pipeline
    from both the CLI and a configuration file (.yaml or .json). It is even
    possible to use both the CLI and a configuration file simultaneously.
    For more details, the reader could refer to PyTorch Lightning CLI documentation.
    """

    def __init__(  # pylint: disable=too-many-function-args
        self,
        model_class: Optional[Union[Type[LightningModule], Callable[..., LightningModule]]] = None,
        datamodule_class: Optional[Union[Type[LightningDataModule], Callable[..., LightningDataModule]]] = None,
        save_config_callback: Optional[Type[SaveConfigCallback]] = SaveConfigCallback,
        save_config_filename: str = "config.yaml",
        save_config_overwrite: bool = False,
        save_config_multifile: bool = False,
        trainer_class: Union[Type[Trainer], Callable[..., Trainer]] = Trainer,
        trainer_defaults: Optional[Dict[str, Any]] = None,
        seed_everything_default: Optional[int] = None,
        description: str = "Anomalib trainer command line tool",
        env_prefix: str = "Anomalib",
        env_parse: bool = False,
        parser_kwargs: Optional[Union[Dict[str, Any], Dict[str, Dict[str, Any]]]] = None,
        subclass_mode_model: bool = False,
        subclass_mode_data: bool = False,
        run: bool = True,
        auto_registry: bool = True,
    ) -> None:
        super().__init__(
            model_class,
            datamodule_class,
            save_config_callback,
            save_config_filename,
            save_config_overwrite,
            save_config_multifile,
            trainer_class,
            trainer_defaults,
            seed_everything_default,
            description,
            env_prefix,
            env_parse,
            parser_kwargs,
            subclass_mode_model,
            subclass_mode_data,
            run,
            auto_registry,
        )

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        """Add custom arguments to the parser."""
        # TODO: design for explicit arguments
        parser.add_argument("--dataset.task", type=str, default="segmentation", required=False)
        # TODO: https://github.com/openvinotoolkit/anomalib/issues/20
        parser.add_argument(
            "--export_mode", type=ExportMode, default=None, help="Select export mode to ONNX or OpenVINO IR format."
        )
        parser.add_argument("--nncf", type=str, help="Path to NNCF config to enable quantized training.")
        parser.add_class_arguments(ImageVisualizerCallback, "visualization")
        parser.add_class_arguments(TilerDecorator, "tiling")
        parser.add_class_arguments(MetricsConfigurationCallback, "metrics")
        parser.add_class_arguments(PostProcessingConfigurationCallback, "post_processing")

    def __set_default_root_dir(self) -> None:
        """Sets the default root directory depending on the subcommand type. <train, fit, predict, tune.>."""
        # Get configs.
        subcommand = self.config["subcommand"]
        config = self.config[subcommand]

        # If `resume_from_checkpoint` is not specified, it means that the project has not been created before.
        # Therefore, we need to create the project directory first.
        if config.trainer.resume_from_checkpoint is None:
            root_dir = config.trainer.default_root_dir if config.trainer.default_root_dir else "./results"
            model_name = config.model.class_path.split(".")[-1].lower()
            data_name = config.data.class_path.split(".")[-1].lower()
            category = config.data.init_args.category if "category" in config.data.init_args else ""
            time_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            default_root_dir = os.path.join(root_dir, model_name, data_name, category, time_stamp)

        # Otherwise, the assumption is that the project directory has alrady been created.
        else:
            # By default, train subcommand saves the weights to
            #   ./results/<model>/<data>/time_stamp/weights/model.ckpt.
            # For this reason, we set the project directory to the parent directory
            #   that is two-level up.
            default_root_dir = str(Path(config.trainer.resume_from_checkpoint).parent.parent)

        if config.visualization.image_save_path == "":
            self.config[subcommand].visualization.image_save_path = default_root_dir + "/images"
        self.config[subcommand].trainer.default_root_dir = default_root_dir

    def __update_callback(self, callbacks: Dict, callback_name: str, update_dict: Dict[str, Any]) -> None:
        """Updates the callback with the given dictionary.

        If the callback exists in the callbacks dictionary, it will be updated with the passed keys.
        This ensure that the callback is not overwritten.
        """
        if callback_name in callbacks:
            callbacks[callback_name].update(update_dict)
        else:
            callbacks[callback_name] = update_dict

    def __set_callbacks(self) -> None:
        """Sets the default callbacks used within the pipeline.

        The callbacks are added to trainer.callbacks as a dictionary so that they can be serialized by
        SaveConfigCallback.
        """
        subcommand = self.config["subcommand"]
        config = self.config[subcommand]

        callbacks: Dict = {}

        # Convert trainer callbacks to a dictionary. It makes it easier to search and update values
        # {"anomalib.utils.callbacks.ImageVisualizerCallback":{'task':...}}
        if config.trainer.callbacks is not None:
            for callback in config.trainer.callbacks:
                callbacks[callback.class_path.split(".")[-1]] = dict(callback.init_args)

        monitor = callbacks.get("EarlyStopping", {}).get("monitor", None)
        mode = callbacks.get("EarlyStopping", {}).get("mode", "max")

        self.__update_callback(
            callbacks,
            "ModelCheckpoint",
            {
                "dirpath": os.path.join(config.trainer.default_root_dir, "weights"),
                "filename": "model",
                "monitor": monitor,
                "mode": mode,
                "auto_insert_metric_name": False,
            },
        )

        self.__update_callback(
            callbacks,
            "PostProcessingConfigurationCallback",
            {
                "normalization_method": config.post_processing.normalization_method,
                "manual_image_threshold": config.post_processing.get("manual_image_threshold", None),
                "manual_pixel_threshold": config.post_processing.get("manual_pixel_threshold", None),
            },
        )

        # Add metric configuration to the model via MetricsConfigurationCallback
        self.__update_callback(
            callbacks,
            "MetricsConfigurationCallback",
            {
                "task": config.dataset.task,
                "image_metrics": config.metrics.get("image_metrics", None),
                "pixel_metrics": config.metrics.get("pixel_metrics", None),
            },
        )

        # LoadModel from Checkpoint.
        if config.trainer.resume_from_checkpoint:
            self.__update_callback(
                callbacks,
                "LoadModelCallback",
                {
                    "weights_path": config.trainer.resume_from_checkpoint,
                },
            )

        # Add timing to the pipeline.
        self.__update_callback(callbacks, "TimerCallback", {})

        #  TODO: This could be set in PostProcessingConfiguration callback
        #   - https://github.com/openvinotoolkit/anomalib/issues/384
        # Normalization.
        normalization = config.post_processing.normalization_method
        if normalization:
            if normalization == "min_max":
                self.__update_callback(callbacks, "MinMaxNormalizationCallback", {})
            elif normalization == "cdf":
                self.__update_callback(callbacks, "CDFNormalizationCallback", {})
            else:
                raise ValueError(
                    f"Unknown normalization type {normalization}. \n" "Available types are either None, min_max or cdf"
                )

        # TODO Add visualization support
        # add_visualizer_callback(callbacks, config)
        # self.config[subcommand].visualization = config.visualization

        # Export to OpenVINO
        if config.export_mode is not None:
            logger.info("Setting model export to %s", config.export_mode)
            self.__update_callback(
                callbacks,
                "ExportCallback",
                {
                    "input_size": config.data.init_args.image_size,
                    "dirpath": os.path.join(config.trainer.default_root_dir, "compressed"),
                    "filename": "model",
                    "export_mode": config.export_mode,
                },
            )
        else:
            warnings.warn(f"Export option: {config.export_mode} not found. Defaulting to no model export")
        if config.nncf:
            if os.path.isfile(config.nncf) and config.nncf.endswith(".yaml"):
                self.__update_callback(
                    callbacks,
                    "anomalib.core.callbacks.nncf_callback.NNCFCallback",
                    {
                        "config": OmegaConf.load(config.nncf),
                        "dirpath": os.path.join(config.trainer.default_root_dir, "compressed"),
                        "filename": "model",
                    },
                )
            else:
                raise ValueError(f"--nncf expects a path to nncf config which is a yaml file, but got {config.nncf}")

        # Convert callbacks to dict format expected by pytorch-lightning.
        # eg [
        # {"class_path": "ModelCheckpoint", "init_args": {...}},
        # {"class_path": "PostProcessingConfigurationCallback", "init_args": {...}}
        # ]
        trainer_callbacks = []
        for class_path, init_args in callbacks.items():
            trainer_callbacks.append({"class_path": class_path, "init_args": init_args})

        self.config[subcommand].trainer.callbacks = trainer_callbacks

    def before_instantiate_classes(self) -> None:
        """Modify the configuration to properly instantiate classes and sets up tiler."""
        subcommand = self.config.subcommand

        TilerDecorator(**vars(self.config[subcommand].tiling))
        self.__set_default_root_dir()
        self.__set_callbacks()


def main() -> None:
    """Trainer via Anomalib CLI."""
    configure_logger()
    AnomalibCLI(model_class=AnomalyModule, subclass_mode_model=True, seed_everything_default=42)


if __name__ == "__main__":
    main()
