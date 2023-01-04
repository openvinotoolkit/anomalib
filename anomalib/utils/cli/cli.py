"""Anomalib CLI."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Type, Union

from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.utilities.cli import (
    LightningArgumentParser,
    LightningCLI,
    SaveConfigCallback,
)

from anomalib.utils.callbacks import (
    ImageVisualizerCallback,
    MetricsConfigurationCallback,
    PostProcessingConfigurationCallback,
    TilerConfigurationCallback,
    get_callbacks_dict,
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
        """Add default arguments.

        Args:
            parser (LightningArgumentParser): Lightning Argument Parser.
        """
        # TODO: https://github.com/openvinotoolkit/anomalib/issues/20
        parser.add_argument(
            "--export_format", type=str, default="", help="Select export mode to ONNX or OpenVINO IR format."
        )
        parser.add_argument("--nncf", type=str, help="Path to NNCF config to enable quantized training.")

        # ADD CUSTOM CALLBACKS TO CONFIG
        # NOTE: MyPy gives the following error:
        # Argument 1 to "add_lightning_class_args" of "LightningArgumentParser"
        # has incompatible type "Type[TilerCallback]"; expected "Union[Type[Trainer],
        # Type[LightningModule], Type[LightningDataModule]]"  [arg-type]
        parser.add_lightning_class_args(TilerConfigurationCallback, "tiling")  # type: ignore
        parser.set_defaults({"tiling.enable": False})

        parser.add_lightning_class_args(PostProcessingConfigurationCallback, "post_processing")  # type: ignore
        parser.set_defaults(
            {
                "post_processing.normalization_method": "min_max",
                "post_processing.threshold_method": "adaptive",
                "post_processing.manual_image_threshold": None,
                "post_processing.manual_pixel_threshold": None,
            }
        )

        # TODO: Assign these default values within the MetricsConfigurationCallback
        #   - https://github.com/openvinotoolkit/anomalib/issues/384
        parser.add_lightning_class_args(MetricsConfigurationCallback, "metrics")  # type: ignore
        parser.set_defaults(
            {
                "metrics.task": "segmentation",
                "metrics.image_metrics": ["F1Score", "AUROC"],
                "metrics.pixel_metrics": ["F1Score", "AUROC"],
            }
        )

        parser.add_lightning_class_args(ImageVisualizerCallback, "visualization")  # type: ignore
        parser.set_defaults(
            {
                "visualization.mode": "full",
                "visualization.task": "segmentation",
                "visualization.image_save_path": "",
                "visualization.save_images": False,
                "visualization.show_images": False,
                "visualization.log_images": False,
            }
        )

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

    def __set_callbacks(self) -> None:
        """Sets the default callbacks used within the pipeline."""
        subcommand = self.config["subcommand"]
        config = self.config[subcommand]

        callbacks = get_callbacks_dict(config)

        self.config[subcommand].trainer.callbacks = callbacks

    def before_instantiate_classes(self) -> None:
        """Modify the configuration to properly instantiate classes."""
        self.__set_default_root_dir()
        self.__set_callbacks()
        print("done.")


def main() -> None:
    """Trainer via Anomalib CLI."""
    configure_logger()
    AnomalibCLI()


if __name__ == "__main__":
    main()
