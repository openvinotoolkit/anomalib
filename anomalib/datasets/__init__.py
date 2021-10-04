"""
Anomalib Datasets
"""

from typing import Union

from omegaconf import DictConfig, ListConfig
from pytorch_lightning import LightningDataModule

from .anomaly_dataset import AnomalyDataModule
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
            image_size=config.transform.image_size,
            crop_size=config.transform.crop_size,
            train_batch_size=config.dataset.train_batch_size,
            test_batch_size=config.dataset.test_batch_size,
            num_workers=config.dataset.num_workers,
        )
    elif config.dataset.format.lower() == "anomaly_dataset":
        datamodule = AnomalyDataModule(
            root=config.dataset.path,
            url=config.dataset.url,
            category=config.dataset.category,
            task=config.dataset.task,
            label_format=config.dataset.label_format,
            train_batch_size=config.dataset.train_batch_size,
            test_batch_size=config.dataset.test_batch_size,
            num_workers=config.dataset.num_workers,
            transform_config=config.transform,
        )
    else:
        raise ValueError("Unknown dataset!")

    return datamodule
