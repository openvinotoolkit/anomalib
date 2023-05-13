"""EfficientAD: Accurate Visual Anomaly Detection at Millisecond-Level Latencies.
https://arxiv.org/pdf/2303.14535.pdf
"""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from pathlib import Path

import albumentations as A
import numpy as np
import torch
import tqdm
from albumentations.pytorch import ToTensorV2
from omegaconf import DictConfig, ListConfig
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor, optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from anomalib.data.utils import DownloadInfo, download_and_extract
from anomalib.models.components import AnomalyModule

from .torch_model import EfficientADModel

logger = logging.getLogger(__name__)

IMAGENETTE_DOWNLOAD_INFO = DownloadInfo(
    name="imagenette2.tgz",
    url="https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz",
    hash="fe2fc210e6bb7c5664d602c3cd71e612",
)


class TransformsWrapper:
    def __init__(self, t: A.Compose):
        self.transforms = t

    def __call__(self, img, *args, **kwargs):
        return self.transforms(image=np.array(img))


class EfficientAD(AnomalyModule):
    """PL Lightning Module for the EfficientAD algorithm."""

    def __init__(
        self,
        category_name: str,
        teacher_file_name: str,
        teacher_out_channels: int,
        pre_trained_dir: str,
        image_size: list,
        model_size: str = "M",
        lr: float = 0.0001,
        weight_decay: float = 0.00001,
        padding: bool = False,
    ) -> None:
        super().__init__()

        self.category = category_name
        self.pre_trained_dir = Path(pre_trained_dir)
        self.imagenet_dir = Path("./datasets/imagenette")
        self.model: EfficientADModel = EfficientADModel(
            teacher_path=self.pre_trained_dir / teacher_file_name,
            teacher_out_channels=teacher_out_channels,
            input_size=image_size,
            model_size=model_size,
            padding=padding,
        )

        self.data_transforms_imagenet = A.Compose(
            [  # We obtain an image P ∈ R 3×256×256 from ImageNet by choosing a random image,
                A.Resize(image_size[0] * 2, image_size[1] * 2),  # resizing it to 512 × 512,
                A.ToGray(p=0.3),  # converting it to gray scale with a probability of 0.3
                A.CenterCrop(image_size[0], image_size[1]),  # and cropping the center 256 × 256 pixels
                A.ToFloat(always_apply=False, p=1.0, max_value=255),
                ToTensorV2(),
            ]
        )

        self.prepare_imagenet_data()
        imagenet_dataset = ImageFolder(self.imagenet_dir, transform=TransformsWrapper(t=self.data_transforms_imagenet))
        self.imagenet_loader = DataLoader(imagenet_dataset, batch_size=1, shuffle=True, pin_memory=True)
        self.imagenet_iterator = iter(self.imagenet_loader)
        self.lr = lr
        self.weight_decay = weight_decay

    def prepare_imagenet_data(self) -> None:
        """Download the imagenet subset if not available."""
        if not self.imagenet_dir.is_dir():
            download_and_extract(self.imagenet_dir, IMAGENETTE_DOWNLOAD_INFO)
        else:
            logger.info("Found the dataset.")

    @torch.no_grad()
    def teacher_channel_mean_std(self, dataloader: DataLoader) -> dict[str, Tensor]:
        """Calculate the the mean and std of the teacher models activations.

        Args:
            dataloader (DataLoader): Dataloader of the respective dataset.

        Returns:
            dict[str, Tensor]: Dictionary of channel-wise mean and std
        """
        y_means = []
        teacher_outputs = []
        means_distance = []

        logger.info("Calculate teacher channel mean and std")
        for batch in tqdm.tqdm(dataloader, desc="Calculate teacher channel mean", position=0, leave=True):
            y = self.model.teacher(batch["image"].to(self.device))
            y_means.append(torch.mean(y, dim=[0, 2, 3]))
            teacher_outputs.append(y)

        channel_mean = torch.mean(torch.stack(y_means), dim=0)[None, :, None, None]

        for y in tqdm.tqdm(teacher_outputs, desc="Calculate teacher channel std", position=0, leave=True):
            distance = (y - channel_mean) ** 2
            means_distance.append(torch.mean(distance, dim=[0, 2, 3]))

        channel_var = torch.mean(torch.stack(means_distance), dim=0)[None, :, None, None]
        channel_std = torch.sqrt(channel_var)
        return {"mean": channel_mean, "std": channel_std}

    @torch.no_grad()
    def map_norm_quantiles(self, dataloader: DataLoader) -> dict[str, Tensor]:
        """Calculate 90% andf 99.5% quantiles of the student and autoencoder feature maps.

        Args:
            dataloader (DataLoader): Dataloader of the respective dataset.

        Returns:
            dict[str, Tensor]: Dictionary of both the 90% and 99.5% quantiles
            of both the student and autoencoder feature maps.
        """
        maps_st = []
        maps_ae = []
        logger.info("Calculate Validation Dataset Quantiles")
        for batch in tqdm.tqdm(dataloader, desc="Calculate Validation Dataset Quantiles", position=0, leave=True):
            for img, label in zip(batch["image"], batch["label"]):
                if label == 0:  # only use good images of validation set!
                    output = self.model(img.to(self.device))
                    map_st = output["map_st"]
                    map_ae = output["map_ae"]
                    maps_st.append(map_st)
                    maps_ae.append(map_ae)
        maps_st = torch.cat(maps_st)
        maps_ae = torch.cat(maps_ae)
        qa_st = torch.quantile(maps_st, q=0.9).to(self.device)
        qb_st = torch.quantile(maps_st, q=0.995).to(self.device)
        qa_ae = torch.quantile(maps_ae, q=0.9).to(self.device)
        qb_ae = torch.quantile(maps_ae, q=0.995).to(self.device)
        return {"qa_st": qa_st, "qa_ae": qa_ae, "qb_st": qb_st, "qb_ae": qb_ae}

    def configure_optimizers(self) -> optim.Optimizer:
        optimizer = optim.AdamW(
            list(self.model.student.parameters()) + list(self.model.ae.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        num_steps = max(
            self.trainer.max_steps, self.trainer.max_epochs * len(self.trainer.datamodule.train_dataloader())
        )
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=int(0.95 * num_steps), gamma=0.1)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def on_train_start(self) -> None:
        """Calculate or load the channel-wise mean and std of the training dataset and push to the model."""
        if not self.model.is_set(self.model.mean_std):
            channel_mean_std = self.teacher_channel_mean_std(self.trainer.datamodule.train_dataloader())
            self.model.mean_std.update(channel_mean_std)

    def training_step(self, batch: dict[str, str | Tensor], *args, **kwargs) -> dict[str, Tensor]:
        """Training step for EfficintAD returns the  student, autoencoder and combined loss.

        Args:
            batch (batch: dict[str, str | Tensor]): Batch containing image filename, image, label and mask

        Returns:
          Loss.
        """
        del args, kwargs  # These variables are not used.

        try:
            # infinite dataloader; [0] getting the image not the label
            batch_imagenet = next(self.imagenet_iterator)[0]["image"].to(self.device)
        except StopIteration:
            self.imagenet_iterator = iter(self.imagenet_loader)
            batch_imagenet = next(self.imagenet_iterator)[0]["image"].to(self.device)

        loss_st, loss_ae, loss_stae = self.model(batch=batch["image"], batch_imagenet=batch_imagenet)

        loss = loss_st + loss_ae + loss_stae
        self.log("train_st", loss_st.item(), on_epoch=True, prog_bar=True, logger=True)
        self.log("train_ae", loss_ae.item(), on_epoch=True, prog_bar=True, logger=True)
        self.log("train_stae", loss_stae.item(), on_epoch=True, prog_bar=True, logger=True)
        self.log("train_loss", loss.item(), on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss}

    def on_validation_start(self) -> None:
        """
        Calculate the feature map quantiles of the validation dataset and push to the model.
        """
        if (self.current_epoch + 1) == self.trainer.max_epochs:
            if not self.model.is_set(self.model.quantiles):
                map_norm_quantiles = self.map_norm_quantiles(self.trainer.datamodule.val_dataloader())
                self.model.quantiles.update(map_norm_quantiles)

    def validation_step(self, batch: dict[str, str | Tensor], *args, **kwargs) -> STEP_OUTPUT:
        """Validation Step of EfficientAD returns anomaly maps for the input image batch

        Args:
          batch (dict[str, str | Tensor]): Input batch

        Returns:
          Dictionary containing anomaly maps.
        """
        del args, kwargs  # These variables are not used.

        batch["anomaly_maps"] = self.model(batch["image"])["anomaly_map_combined"]

        return batch


class EfficientadLightning(EfficientAD):
    """PL Lightning Module for the EfficientAD Algorithm.

    Args:
        hparams (DictConfig | ListConfig): Model params
    """

    def __init__(self, hparams: DictConfig | ListConfig) -> None:
        super().__init__(
            category_name=hparams.dataset.category,
            teacher_file_name=hparams.model.teacher_file_name,
            pre_trained_dir=hparams.model.pre_trained_dir,
            teacher_out_channels=hparams.model.teacher_out_channels,
            model_size=hparams.model.model_size,
            lr=hparams.model.lr,
            weight_decay=hparams.model.weight_decay,
            padding=hparams.model.padding,
            image_size=hparams.dataset.image_size,
        )
        self.hparams: DictConfig | ListConfig  # type: ignore
        self.save_hyperparameters(hparams)
