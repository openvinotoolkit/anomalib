"""Anomalib CLI."""

from typing import List

from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.cli import LightningArgumentParser, LightningCLI

from anomalib.core.callbacks import (
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
            # save_config_overwrite=False,
            run=False,
        )

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        """Add default arguments.

        Args:
            parser (LightningArgumentParser): Lightning Argument Parser.
        """
        parser.add_argument("--save_to_csv", type=bool, default=False, help="Save results to a CSV")
        parser.add_argument("--save_images", type=bool, default=True, help="Flag to save output images locally.")
        # NOTE: MyPy gives the following error:
        # Argument 1 to "add_lightning_class_args" of "LightningArgumentParser"
        # has incompatible type "Type[TilerCallback]"; expected "Union[Type[Trainer],
        # Type[LightningModule], Type[LightningDataModule]]"  [arg-type]
        parser.add_lightning_class_args(TilerCallback, "tiling")  # type: ignore
        parser.set_defaults({"tiling.enable": False})

    def before_instantiate_classes(self) -> None:
        """Modify the configuration to properly instantiate classes."""
        callbacks: List[Callback] = []

        # Add timing to the pipeline.
        callbacks.append(TimerCallback())

        if self.config["save_to_csv"]:
            callbacks.append(SaveToCSVCallback())

        # Visualization.
        if self.config["save_images"]:
            callbacks.append(VisualizerCallback(loggers=["local"]))

        self.config["trainer"]["callbacks"] = callbacks

        print("done.")
