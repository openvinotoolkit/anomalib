"""Anomalib CLI."""

from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.utilities.cli import LightningArgumentParser, LightningCLI

from anomalib.core.callbacks.tiler import TilerCallback


class AnomalibCLI(LightningCLI):
    """Anomalib CLI"""

    def __init__(self):
        super().__init__(
            model_class=LightningModule,
            datamodule_class=LightningDataModule,
            subclass_mode_model=True,
            subclass_mode_data=True,
            seed_everything_default=0,
            save_config_overwrite=False,
            run=False,
        )

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.add_lightning_class_args(TilerCallback, "tiling")
        parser.set_defaults(
            {
                "tiling.enable": False,
            }
        )
