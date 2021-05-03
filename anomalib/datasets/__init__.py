from argparse import Namespace
from typing import Dict, Union
from omegaconf import DictConfig, ListConfig
from pytorch_lightning import LightningDataModule

from .mvtec import MVTecDataModule


def get_datamodule(args: Union[DictConfig, ListConfig]):
    datamodule: LightningDataModule
    if args.dataset.lower() == "mvtec":
        datamodule = MVTecDataModule(args.dataset_path, args.batch_size, args.num_workers)
    else:
        raise ValueError("Unknown dataset!")

    return datamodule
