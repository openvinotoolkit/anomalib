"""EfficientAd: Accurate Visual Anomaly Detection at Millisecond-Level Latencies.

https://arxiv.org/pdf/2303.14535.pdf.
"""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from pathlib import Path
from typing import Any

import torch
import tqdm
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms.v2 import CenterCrop, Compose, Normalize, RandomGrayscale, Resize, ToTensor, Transform

from anomalib import LearningType
from anomalib.data.utils import DownloadInfo, download_and_extract
from anomalib.models.components import AnomalyModule

from .torch_model import EfficientAdModel, EfficientAdModelSize, reduce_tensor_elems

logger = logging.getLogger(__name__)

IMAGENETTE_DOWNLOAD_INFO = DownloadInfo(
    name="imagenette2.tgz",
    url="https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz",
    hashsum="6cbfac238434d89fe99e651496f0812ebc7a10fa62bd42d6874042bf01de4efd",
)

WEIGHTS_DOWNLOAD_INFO = DownloadInfo(
    name="efficientad_pretrained_weights.zip",
    url="https://github.com/openvinotoolkit/anomalib/releases/download/efficientad_pretrained_weights/efficientad_pretrained_weights.zip",
    hashsum="c09aeaa2b33f244b3261a5efdaeae8f8284a949470a4c5a526c61275fe62684a",
)


class EfficientAd(AnomalyModule):
    """PL Lightning Module for the EfficientAd algorithm.

    Args:
        imagenet_dir (Path|str): directory path for the Imagenet dataset
            Defaults to ``./datasets/imagenette``.
        teacher_out_channels (int): number of convolution output channels
            Defaults to ``384``.
        model_size (str): size of student and teacher model
            Defaults to ``EfficientAdModelSize.S``.
        lr (float): learning rate
            Defaults to ``0.0001``.
        weight_decay (float): optimizer weight decay
            Defaults to ``0.00001``.
        padding (bool): use padding in convoluional layers
            Defaults to ``False``.
        pad_maps (bool): relevant if padding is set to False. In this case, pad_maps = True pads the
            output anomaly maps so that their size matches the size in the padding = True case.
            Defaults to ``True``.
    """

    def __init__(
        self,
        imagenet_dir: Path | str = "./datasets/imagenette",
        teacher_out_channels: int = 384,
        model_size: EfficientAdModelSize | str = EfficientAdModelSize.S,
        lr: float = 0.0001,
        weight_decay: float = 0.00001,
        padding: bool = False,
        pad_maps: bool = True,
    ) -> None:
        super().__init__()

        self.imagenet_dir = Path(imagenet_dir)
        if not isinstance(model_size, EfficientAdModelSize):
            model_size = EfficientAdModelSize(model_size)
        self.model_size: EfficientAdModelSize = model_size
        self.model: EfficientAdModel = EfficientAdModel(
            teacher_out_channels=teacher_out_channels,
            model_size=model_size,
            padding=padding,
            pad_maps=pad_maps,
        )
        self.batch_size: int = 1  # imagenet dataloader batch_size is 1 according to the paper
        self.lr: float = lr
        self.weight_decay: float = weight_decay

    def prepare_pretrained_model(self) -> None:
        """Prepare the pretrained teacher model."""
        pretrained_models_dir = Path("./pre_trained/")
        if not (pretrained_models_dir / "efficientad_pretrained_weights").is_dir():
            download_and_extract(pretrained_models_dir, WEIGHTS_DOWNLOAD_INFO)
        model_size_str = self.model_size.value if isinstance(self.model_size, EfficientAdModelSize) else self.model_size
        teacher_path = (
            pretrained_models_dir / "efficientad_pretrained_weights" / f"pretrained_teacher_{model_size_str}.pth"
        )
        logger.info(f"Load pretrained teacher model from {teacher_path}")
        self.model.teacher.load_state_dict(torch.load(teacher_path, map_location=torch.device(self.device)))

    def prepare_imagenette_data(self, image_size: tuple[int, int] | torch.Size) -> None:
        """Prepare ImageNette dataset transformations.

        Args:
            image_size (tuple[int, int] | torch.Size): Image size.
        """
        self.data_transforms_imagenet = Compose(
            [
                Resize((image_size[0] * 2, image_size[1] * 2)),
                RandomGrayscale(p=0.3),
                CenterCrop((image_size[0], image_size[1])),
                ToTensor(),
            ],
        )

        if not self.imagenet_dir.is_dir():
            download_and_extract(self.imagenet_dir, IMAGENETTE_DOWNLOAD_INFO)
        imagenet_dataset = ImageFolder(self.imagenet_dir, transform=self.data_transforms_imagenet)
        self.imagenet_loader = DataLoader(imagenet_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)
        self.imagenet_iterator = iter(self.imagenet_loader)

    @torch.no_grad()
    def teacher_channel_mean_std(self, dataloader: DataLoader) -> dict[str, torch.Tensor]:
        """Calculate the mean and std of the teacher models activations.

        Adapted from https://math.stackexchange.com/a/2148949

        Args:
            dataloader (DataLoader): Dataloader of the respective dataset.

        Returns:
            dict[str, torch.Tensor]: Dictionary of channel-wise mean and std
        """
        arrays_defined = False
        n: torch.Tensor | None = None
        chanel_sum: torch.Tensor | None = None
        chanel_sum_sqr: torch.Tensor | None = None

        for batch in tqdm.tqdm(dataloader, desc="Calculate teacher channel mean & std", position=0, leave=True):
            y = self.model.teacher(batch["image"].to(self.device))
            if not arrays_defined:
                _, num_channels, _, _ = y.shape
                n = torch.zeros((num_channels,), dtype=torch.int64, device=y.device)
                chanel_sum = torch.zeros((num_channels,), dtype=torch.float32, device=y.device)
                chanel_sum_sqr = torch.zeros((num_channels,), dtype=torch.float32, device=y.device)
                arrays_defined = True

            n += y[:, 0].numel()
            chanel_sum += torch.sum(y, dim=[0, 2, 3])
            chanel_sum_sqr += torch.sum(y**2, dim=[0, 2, 3])

        if n is None:
            msg = "The value of 'n' cannot be None."
            raise ValueError(msg)

        channel_mean = chanel_sum / n

        channel_std = (torch.sqrt((chanel_sum_sqr / n) - (channel_mean**2))).float()[None, :, None, None]
        channel_mean = channel_mean.float()[None, :, None, None]

        return {"mean": channel_mean, "std": channel_std}

    @torch.no_grad()
    def map_norm_quantiles(self, dataloader: DataLoader) -> dict[str, torch.Tensor]:
        """Calculate 90% and 99.5% quantiles of the student(st) and autoencoder(ae).

        Args:
            dataloader (DataLoader): Dataloader of the respective dataset.

        Returns:
            dict[str, torch.Tensor]: Dictionary of both the 90% and 99.5% quantiles
            of both the student and autoencoder feature maps.
        """
        maps_st = []
        maps_ae = []
        logger.info("Calculate Validation Dataset Quantiles")
        for batch in tqdm.tqdm(dataloader, desc="Calculate Validation Dataset Quantiles", position=0, leave=True):
            for img, label in zip(batch["image"], batch["label"], strict=True):
                if label == 0:  # only use good images of validation set!
                    output = self.model(img.to(self.device), normalize=False)
                    map_st = output["map_st"]
                    map_ae = output["map_ae"]
                    maps_st.append(map_st)
                    maps_ae.append(map_ae)

        qa_st, qb_st = self._get_quantiles_of_maps(maps_st)
        qa_ae, qb_ae = self._get_quantiles_of_maps(maps_ae)
        return {"qa_st": qa_st, "qa_ae": qa_ae, "qb_st": qb_st, "qb_ae": qb_ae}

    def _get_quantiles_of_maps(self, maps: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate 90% and 99.5% quantiles of the given anomaly maps.

        If the total number of elements in the given maps is larger than 16777216
        the returned quantiles are computed on a random subset of the given
        elements.

        Args:
            maps (list[torch.Tensor]): List of anomaly maps.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Two scalars - the 90% and the 99.5% quantile.
        """
        maps_flat = reduce_tensor_elems(torch.cat(maps))
        qa = torch.quantile(maps_flat, q=0.9).to(self.device)
        qb = torch.quantile(maps_flat, q=0.995).to(self.device)
        return qa, qb

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizers."""
        optimizer = torch.optim.Adam(
            list(self.model.student.parameters()) + list(self.model.ae.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        if self.trainer.max_epochs < 0 and self.trainer.max_steps < 0:
            msg = "A finite number of steps or epochs must be defined"
            raise ValueError(msg)

        # lightning stops training when either 'max_steps' or 'max_epochs' is reached (earliest),
        # so actual training steps need to be determined here
        if self.trainer.max_epochs < 0:
            # max_epochs not set
            num_steps = self.trainer.max_steps
        elif self.trainer.max_steps < 0:
            # max_steps not set -> determine steps as 'max_epochs' * 'steps in a single training epoch'
            num_steps = self.trainer.max_epochs * len(self.trainer.datamodule.train_dataloader())
        else:
            num_steps = min(
                self.trainer.max_steps,
                self.trainer.max_epochs * len(self.trainer.datamodule.train_dataloader()),
            )

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(0.95 * num_steps), gamma=0.1)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def on_train_start(self) -> None:
        """Called before the first training epoch.

        First check if EfficientAd-specific parameters are set correctly (train_batch_size of 1
        and no Imagenet normalization in transforms), then sets up the pretrained teacher model,
        then prepares the imagenette data, and finally calculates or loads
        the channel-wise mean and std of the training dataset and push to the model.
        """
        if self.trainer.datamodule.train_batch_size != 1:
            msg = "train_batch_size for EfficientAd should be 1."
            raise ValueError(msg)
        if self._transform and any(isinstance(transform, Normalize) for transform in self._transform.transforms):
            msg = "Transforms for EfficientAd should not contain Normalize."
            raise ValueError(msg)

        sample = next(iter(self.trainer.train_dataloader))
        image_size = sample["image"].shape[-2:]
        self.prepare_pretrained_model()
        self.prepare_imagenette_data(image_size)
        if not self.model.is_set(self.model.mean_std):
            channel_mean_std = self.teacher_channel_mean_std(self.trainer.datamodule.train_dataloader())
            self.model.mean_std.update(channel_mean_std)

    def training_step(self, batch: dict[str, str | torch.Tensor], *args, **kwargs) -> dict[str, torch.Tensor]:
        """Perform the training step for EfficientAd returns the student, autoencoder and combined loss.

        Args:
            batch (batch: dict[str, str | torch.Tensor]): Batch containing image filename, image, label and mask
            args: Additional arguments.
            kwargs: Additional keyword arguments.

        Returns:
          Loss.
        """
        del args, kwargs  # These variables are not used.

        try:
            # infinite dataloader; [0] getting the image not the label
            batch_imagenet = next(self.imagenet_iterator)[0].to(self.device)
        except StopIteration:
            self.imagenet_iterator = iter(self.imagenet_loader)
            batch_imagenet = next(self.imagenet_iterator)[0].to(self.device)

        loss_st, loss_ae, loss_stae = self.model(batch=batch["image"], batch_imagenet=batch_imagenet)

        loss = loss_st + loss_ae + loss_stae
        self.log("train_st", loss_st.item(), on_epoch=True, prog_bar=True, logger=True)
        self.log("train_ae", loss_ae.item(), on_epoch=True, prog_bar=True, logger=True)
        self.log("train_stae", loss_stae.item(), on_epoch=True, prog_bar=True, logger=True)
        self.log("train_loss", loss.item(), on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss}

    def on_validation_start(self) -> None:
        """Calculate the feature map quantiles of the validation dataset and push to the model."""
        map_norm_quantiles = self.map_norm_quantiles(self.trainer.datamodule.val_dataloader())
        self.model.quantiles.update(map_norm_quantiles)

    def validation_step(self, batch: dict[str, str | torch.Tensor], *args, **kwargs) -> STEP_OUTPUT:
        """Perform the validation step of EfficientAd returns anomaly maps for the input image batch.

        Args:
          batch (dict[str, str | torch.Tensor]): Input batch
          args: Additional arguments.
          kwargs: Additional keyword arguments.

        Returns:
          Dictionary containing anomaly maps.
        """
        del args, kwargs  # These variables are not used.

        batch["anomaly_maps"] = self.model(batch["image"])["anomaly_map"]

        return batch

    @property
    def trainer_arguments(self) -> dict[str, Any]:
        """Return EfficientAD trainer arguments."""
        return {"num_sanity_val_steps": 0}

    @property
    def learning_type(self) -> LearningType:
        """Return the learning type of the model.

        Returns:
            LearningType: Learning type of the model.
        """
        return LearningType.ONE_CLASS

    @staticmethod
    def configure_transforms(image_size: tuple[int, int] | None = None) -> Transform:
        """Default transform for EfficientAd. Imagenet normalization applied in forward."""
        image_size = image_size or (256, 256)
        return Compose(
            [
                Resize(image_size, antialias=True),
            ],
        )
