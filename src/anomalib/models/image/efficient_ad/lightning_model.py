"""EfficientAd: Accurate Visual Anomaly Detection at Millisecond-Level Latencies.

This module implements the EfficientAd model for fast and accurate anomaly
detection. EfficientAd uses a student-teacher architecture with a pre-trained
EfficientNet backbone to achieve state-of-the-art performance with
millisecond-level inference times.

The model consists of:
    - A pre-trained EfficientNet teacher network
    - A lightweight student network
    - Knowledge distillation training
    - Anomaly detection via feature comparison

Example:
    >>> from anomalib.data import MVTecAD
    >>> from anomalib.models import EfficientAd
    >>> from anomalib.engine import Engine

    >>> datamodule = MVTecAD()
    >>> model = EfficientAd()
    >>> engine = Engine()

    >>> engine.fit(model, datamodule=datamodule)  # doctest: +SKIP
    >>> predictions = engine.predict(model, datamodule=datamodule)  # doctest: +SKIP

Paper:
    "EfficientAd: Accurate Visual Anomaly Detection at
    Millisecond-Level Latencies"
    https://arxiv.org/pdf/2303.14535.pdf

See Also:
    :class:`anomalib.models.image.efficient_ad.torch_model.EfficientAdModel`:
        PyTorch implementation of the EfficientAd model architecture.
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
from torchvision.transforms.v2 import CenterCrop, Compose, Normalize, RandomGrayscale, Resize, ToTensor

from anomalib import LearningType
from anomalib.data import Batch
from anomalib.data.transforms.utils import extract_transforms_by_type
from anomalib.data.utils import DownloadInfo, download_and_extract
from anomalib.metrics import Evaluator
from anomalib.models.components import AnomalibModule
from anomalib.post_processing import PostProcessor
from anomalib.pre_processing import PreProcessor
from anomalib.visualization import Visualizer

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


class EfficientAd(AnomalibModule):
    """PL Lightning Module for the EfficientAd algorithm.

    The EfficientAd model uses a student-teacher architecture with a pretrained
    EfficientNet backbone for fast and accurate anomaly detection.

    Args:
        imagenet_dir (Path | str): Directory path for the Imagenet dataset.
            Defaults to ``"./datasets/imagenette"``.
        teacher_out_channels (int): Number of convolution output channels.
            Defaults to ``384``.
        model_size (EfficientAdModelSize | str): Size of student and teacher model.
            Defaults to ``EfficientAdModelSize.S``.
        lr (float): Learning rate.
            Defaults to ``0.0001``.
        weight_decay (float): Optimizer weight decay.
            Defaults to ``0.00001``.
        padding (bool): Use padding in convolutional layers.
            Defaults to ``False``.
        pad_maps (bool): Relevant if ``padding=False``. If ``True``, pads the output
            anomaly maps to match size of ``padding=True`` case.
            Defaults to ``True``.
        pre_processor (PreProcessor | bool, optional): Pre-processor used to transform
            input data before passing to model.
            Defaults to ``True``.
        post_processor (PostProcessor | bool, optional): Post-processor used to process
            model predictions.
            Defaults to ``True``.
        evaluator (Evaluator | bool, optional): Evaluator used to compute metrics.
            Defaults to ``True``.
        visualizer (Visualizer | bool, optional): Visualizer used to create
            visualizations.
            Defaults to ``True``.

    Example:
        >>> from anomalib.models import EfficientAd
        >>> model = EfficientAd(
        ...     imagenet_dir="./datasets/imagenette",
        ...     model_size="s",
        ...     lr=1e-4
        ... )

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
        pre_processor: PreProcessor | bool = True,
        post_processor: PostProcessor | bool = True,
        evaluator: Evaluator | bool = True,
        visualizer: Visualizer | bool = True,
    ) -> None:
        super().__init__(
            pre_processor=pre_processor,
            post_processor=post_processor,
            evaluator=evaluator,
            visualizer=visualizer,
        )

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
        """Prepare the pretrained teacher model.

        Downloads and loads pretrained weights for the teacher model if not already
        present.
        """
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

        Sets up data transforms and downloads ImageNette dataset if not present.

        Args:
            image_size (tuple[int, int] | torch.Size): Target image size for
                transforms.
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
        """Calculate channel-wise mean and std of teacher model activations.

        Computes running mean and standard deviation of teacher model feature maps
        over the full dataset.

        Args:
            dataloader (DataLoader): Dataloader for the dataset.

        Returns:
            dict[str, torch.Tensor]: Dictionary containing:
                - ``mean``: Channel-wise means of shape ``(1, C, 1, 1)``
                - ``std``: Channel-wise standard deviations of shape
                  ``(1, C, 1, 1)``

        Raises:
            ValueError: If no data is provided (``n`` remains ``None``).
        """
        arrays_defined = False
        n: torch.Tensor | None = None
        chanel_sum: torch.Tensor | None = None
        chanel_sum_sqr: torch.Tensor | None = None

        for batch in tqdm.tqdm(dataloader, desc="Calculate teacher channel mean & std", position=0, leave=True):
            y = self.model.teacher(batch.image.to(self.device))
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
        """Calculate quantiles of student and autoencoder feature maps.

        Computes the 90% and 99.5% quantiles of the feature maps from both the
        student network and autoencoder on normal (good) validation samples.

        Args:
            dataloader (DataLoader): Validation dataloader.

        Returns:
            dict[str, torch.Tensor]: Dictionary containing:
                - ``qa_st``: 90% quantile of student maps
                - ``qa_ae``: 90% quantile of autoencoder maps
                - ``qb_st``: 99.5% quantile of student maps
                - ``qb_ae``: 99.5% quantile of autoencoder maps
        """
        maps_st = []
        maps_ae = []
        logger.info("Calculate Validation Dataset Quantiles")
        for batch in tqdm.tqdm(dataloader, desc="Calculate Validation Dataset Quantiles", position=0, leave=True):
            for img, label in zip(batch.image, batch.gt_label, strict=True):
                if label == 0:  # only use good images of validation set!
                    map_st, map_ae = self.model.get_maps(img.to(self.device), normalize=False)
                    maps_st.append(map_st)
                    maps_ae.append(map_ae)

        qa_st, qb_st = self._get_quantiles_of_maps(maps_st)
        qa_ae, qb_ae = self._get_quantiles_of_maps(maps_ae)
        return {"qa_st": qa_st, "qa_ae": qa_ae, "qb_st": qb_st, "qb_ae": qb_ae}

    def _get_quantiles_of_maps(self, maps: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate quantiles of anomaly maps.

        Computes the 90% and 99.5% quantiles of the given anomaly maps. If total
        number of elements exceeds 16777216, uses a random subset.

        Args:
            maps (list[torch.Tensor]): List of anomaly maps.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple containing:
                - 90% quantile scalar
                - 99.5% quantile scalar
        """
        maps_flat = reduce_tensor_elems(torch.cat(maps))
        qa = torch.quantile(maps_flat, q=0.9).to(self.device)
        qb = torch.quantile(maps_flat, q=0.995).to(self.device)
        return qa, qb

    @classmethod
    def configure_pre_processor(cls, image_size: tuple[int, int] | None = None) -> PreProcessor:
        """Configure default pre-processor for EfficientAd.

        Note that ImageNet normalization is applied in the forward pass, not here.

        Args:
            image_size (tuple[int, int] | None, optional): Target image size.
                Defaults to ``(256, 256)``.

        Returns:
            PreProcessor: Configured pre-processor with resize transform.
        """
        image_size = image_size or (256, 256)
        transform = Compose([Resize(image_size, antialias=True)])
        return PreProcessor(transform=transform)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizers for training.

        Sets up Adam optimizer with learning rate scheduler that decays LR by 0.1
        at 95% of training.

        Returns:
            dict: Dictionary containing:
                - ``optimizer``: Adam optimizer
                - ``lr_scheduler``: StepLR scheduler

        Raises:
            ValueError: If neither ``max_epochs`` nor ``max_steps`` is defined.
        """
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
        """Set up model before training begins.

        Performs the following steps:
        1. Validates training parameters (batch size=1, no normalization)
        2. Sets up pretrained teacher model
        3. Prepares ImageNette dataset
        4. Calculates channel statistics

        Raises:
            ValueError: If ``train_batch_size != 1`` or transforms contain
                normalization.
        """
        if self.trainer.datamodule.train_batch_size != 1:
            msg = "train_batch_size for EfficientAd should be 1."
            raise ValueError(msg)

        if self.pre_processor and extract_transforms_by_type(self.pre_processor.transform, Normalize):
            msg = "Transforms for EfficientAd should not contain Normalize."
            raise ValueError(msg)

        sample = next(iter(self.trainer.train_dataloader))
        image_size = sample.image.shape[-2:]
        self.prepare_pretrained_model()
        self.prepare_imagenette_data(image_size)
        if not self.model.is_set(self.model.mean_std):
            channel_mean_std = self.teacher_channel_mean_std(self.trainer.datamodule.train_dataloader())
            self.model.mean_std.update(channel_mean_std)

    def training_step(self, batch: Batch, *args, **kwargs) -> dict[str, torch.Tensor]:
        """Perform training step.

        Computes student, autoencoder and combined losses using both the input
        batch and a batch from ImageNette.

        Args:
            batch (Batch): Input batch containing image and labels
            *args: Additional arguments (unused)
            **kwargs: Additional keyword arguments (unused)

        Returns:
            dict[str, torch.Tensor]: Dictionary containing total loss
        """
        del args, kwargs  # These variables are not used.

        try:
            # infinite dataloader; [0] getting the image not the label
            batch_imagenet = next(self.imagenet_iterator)[0].to(self.device)
        except StopIteration:
            self.imagenet_iterator = iter(self.imagenet_loader)
            batch_imagenet = next(self.imagenet_iterator)[0].to(self.device)

        loss_st, loss_ae, loss_stae = self.model(batch=batch.image, batch_imagenet=batch_imagenet)

        loss = loss_st + loss_ae + loss_stae
        self.log("train_st", loss_st.item(), on_epoch=True, prog_bar=True, logger=True)
        self.log("train_ae", loss_ae.item(), on_epoch=True, prog_bar=True, logger=True)
        self.log("train_stae", loss_stae.item(), on_epoch=True, prog_bar=True, logger=True)
        self.log("train_loss", loss.item(), on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss}

    def on_validation_start(self) -> None:
        """Calculate feature map statistics before validation.

        Computes quantiles of feature maps on validation set and updates model.
        """
        map_norm_quantiles = self.map_norm_quantiles(self.trainer.datamodule.val_dataloader())
        self.model.quantiles.update(map_norm_quantiles)

    def validation_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        """Perform validation step.

        Generates anomaly maps for the input batch.

        Args:
            batch (Batch): Input batch
            *args: Additional arguments (unused)
            **kwargs: Additional keyword arguments (unused)

        Returns:
            STEP_OUTPUT: Batch with added predictions
        """
        del args, kwargs  # These variables are not used.

        predictions = self.model(batch.image)
        return batch.update(**predictions._asdict())

    @property
    def trainer_arguments(self) -> dict[str, Any]:
        """Get trainer arguments.

        Returns:
            dict[str, Any]: Dictionary with trainer arguments:
                - ``num_sanity_val_steps``: 0
        """
        return {"num_sanity_val_steps": 0}

    @property
    def learning_type(self) -> LearningType:
        """Get model's learning type.

        Returns:
            LearningType: Always ``LearningType.ONE_CLASS``
        """
        return LearningType.ONE_CLASS
