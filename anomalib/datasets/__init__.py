"""
Anomalib Datasets
"""

from typing import Union

from omegaconf import DictConfig, ListConfig
from pytorch_lightning import LightningDataModule

from .mvtec import MVTecDataModule


def get_datamodule(config: Union[DictConfig, ListConfig]):
    """
    Get Anomaly Datamodule

    Args:
        config: Configuration of the anomaly model
        config: Union[DictConfig, ListConfig]:

    Returns:
        PyTorch Lightning DataModule

    """
    datamodule: LightningDataModule

    if config.dataset.format.lower() == "mvtec":
        datamodule = MVTecDataModule(
            root=config.dataset.path,
            category=config.dataset.category,
            image_size=config.dataset.image_size,
            crop_size=config.dataset.crop_size,
            batch_size=config.dataset.batch_size,
            num_workers=config.dataset.num_workers,
        )
    else:
        raise ValueError("Unknown dataset!")

    return datamodule
