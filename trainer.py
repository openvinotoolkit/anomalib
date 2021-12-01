"""Trainer."""

# from pytorch_lightning.core import datamodule
from pytorch_lightning.utilities.cli import LightningCLI
from anomalib.cli import AnomalibCLI

from anomalib.core.model import AnomalyModule
from anomalib.data import Mvtec
from anomalib.models import Padim

from pytorch_lightning import LightningDataModule, LightningModule

# Subclass Mode.
cli = LightningCLI(
    model_class=LightningModule,
    datamodule_class=LightningDataModule,
    subclass_mode_model=True,
    subclass_mode_data=True,
    seed_everything_default=0,
    save_config_overwrite=False,
    run=False,
)


# cli = AnomalibCLI(
#     model_class=PadimLightning,
#     datamodule_class=MVTecDataModule,
#     seed_everything_default=0,
#     save_config_callback=None,
#     run=False,
# )
# cli = AnomalibCLI(seed_everything_default=0, save_config_callback=None, run=False)

# model_name = cli.config["algorithm"]
# model_class = f"{model_name.capitalize()}Lightning"
# cli = AnomalibCLI(
#     model_class=model, datamodule_class=data, seed_everything_default=0, run=False, save_config_callback=None
# )

#
cli.trainer.fit(datamodule=cli.datamodule, model=cli.model)
cli.trainer.test(datamodule=cli.datamodule, model=cli.model)
