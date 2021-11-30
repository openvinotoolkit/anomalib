"""Trainer."""

# from pytorch_lightning.core import datamodule
# from pytorch_lightning.utilities.cli import LightningCLI

from anomalib.cli import AnomalibCLI

# from anomalib.core.model import AnomalyModule
# from anomalib.data import MVTecDataModule
# from anomalib.models.padim.model import PadimLightning

cli = AnomalibCLI(seed_everything_default=0, run=False, save_config_callback=None)

# model_name = cli.config["algorithm"]
# model_class = f"{model_name.capitalize()}Lightning"
# cli = AnomalibCLI(
#     model_class=model, datamodule_class=data, seed_everything_default=0, run=False, save_config_callback=None
# )

cli.trainer.fit(datamodule=cli.datamodule, model=cli.model)
cli.trainer.test(datamodule=cli.datamodule, model=cli.model)
