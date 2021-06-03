from typing import Union

from omegaconf import DictConfig, ListConfig
from pytorch_lightning import LightningDataModule

from .mvtec import MVTecDataModule


def get_datamodule(config: Union[DictConfig, ListConfig]):
    datamodule: LightningDataModule
    if config.dataset.name.lower() == "mvtec":
        datamodule = MVTecDataModule(
            root=config.dataset.path,
            category=config.dataset.category,
            batch_size=config.dataset.batch_size,
            num_workers=config.dataset.num_workers,
        )
    else:
        raise ValueError("Unknown dataset!")

    return datamodule
