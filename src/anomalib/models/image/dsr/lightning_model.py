"""DSR - A Dual Subspace Re-Projection Network for Surface Anomaly Detection.

This module implements the DSR model for surface anomaly detection. DSR uses a dual
subspace re-projection approach to detect anomalies by comparing input images with
their reconstructions in two different subspaces.

The model consists of three training phases:
1. A discrete model pre-training phase (using pre-trained weights)
2. Training of the main reconstruction and anomaly detection modules
3. Training of the upsampling module

Paper: https://link.springer.com/chapter/10.1007/978-3-031-19821-2_31

Example:
    >>> from anomalib.models.image import Dsr
    >>> model = Dsr(
    ...     latent_anomaly_strength=0.2,
    ...     upsampling_train_ratio=0.7
    ... )

The model can be used with any of the supported datasets and task modes in
anomalib.

Notes:
    The model requires pre-trained weights for the discrete model which are
    downloaded automatically during training.

See Also:
    :class:`anomalib.models.image.dsr.torch_model.DsrModel`:
        PyTorch implementation of the DSR model architecture.
"""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from pathlib import Path
from typing import Any

import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torchvision.transforms.v2 import Compose, Resize

from anomalib import LearningType
from anomalib.data import Batch
from anomalib.data.utils import DownloadInfo, download_and_extract
from anomalib.data.utils.generators.perlin import PerlinAnomalyGenerator
from anomalib.metrics import Evaluator
from anomalib.models.components import AnomalibModule
from anomalib.models.image.dsr.anomaly_generator import DsrAnomalyGenerator
from anomalib.models.image.dsr.loss import DsrSecondStageLoss, DsrThirdStageLoss
from anomalib.models.image.dsr.torch_model import DsrModel
from anomalib.post_processing import PostProcessor
from anomalib.pre_processing import PreProcessor
from anomalib.visualization import Visualizer

__all__ = ["Dsr"]

logger = logging.getLogger(__name__)

WEIGHTS_DOWNLOAD_INFO = DownloadInfo(
    name="vq_model_pretrained_128_4096.pckl",
    url="https://github.com/openvinotoolkit/anomalib/releases/download/"
    "dsr_pretrained_weights/dsr_vq_model_pretrained.zip",
    hashsum="52fe7504ec8e9df70b4382f287ab26269dcfe000cd7a7e146a52c6f146f34afb",
)


class Dsr(AnomalibModule):
    """DSR: A Dual Subspace Re-Projection Network for Surface Anomaly Detection.

    The model uses a dual subspace approach with three training phases:
    1. Pre-trained discrete model (loaded from weights)
    2. Training of reconstruction and anomaly detection modules
    3. Training of the upsampling module for final anomaly map generation

    Args:
        latent_anomaly_strength (float, optional): Strength of the generated
            anomalies in the latent space. Defaults to ``0.2``.
        upsampling_train_ratio (float, optional): Ratio of training steps for
            the upsampling module. Defaults to ``0.7``.
        pre_processor (PreProcessor | bool, optional): Pre-processor instance or
            flag to use default. Defaults to ``True``.
        post_processor (PostProcessor | bool, optional): Post-processor instance
            or flag to use default. Defaults to ``True``.
        evaluator (Evaluator | bool, optional): Evaluator instance or flag to
            use default. Defaults to ``True``.
        visualizer (Visualizer | bool, optional): Visualizer instance or flag to
            use default. Defaults to ``True``.

    Example:
        >>> from anomalib.models.image import Dsr
        >>> model = Dsr(
        ...     latent_anomaly_strength=0.2,
        ...     upsampling_train_ratio=0.7
        ... )
        >>> model.trainer_arguments
        {'num_sanity_val_steps': 0}
    """

    def __init__(
        self,
        latent_anomaly_strength: float = 0.2,
        upsampling_train_ratio: float = 0.7,
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

        self.automatic_optimization = False
        self.upsampling_train_ratio = upsampling_train_ratio

        self.quantized_anomaly_generator = DsrAnomalyGenerator()
        self.perlin_generator = PerlinAnomalyGenerator()
        self.model = DsrModel(latent_anomaly_strength)
        self.second_stage_loss = DsrSecondStageLoss()
        self.third_stage_loss = DsrThirdStageLoss()

        self.second_phase: int

    @staticmethod
    def prepare_pretrained_model() -> Path:
        """Download pre-trained models if they don't exist.

        Returns:
            Path: Path to the downloaded pre-trained model weights.

        Example:
            >>> model = Dsr()
            >>> weights_path = model.prepare_pretrained_model()
            >>> weights_path.name
            'vq_model_pretrained_128_4096.pckl'
        """
        pretrained_models_dir = Path("./pre_trained/")
        if not (pretrained_models_dir / "vq_model_pretrained_128_4096.pckl").is_file():
            download_and_extract(pretrained_models_dir, WEIGHTS_DOWNLOAD_INFO)
        return pretrained_models_dir / "vq_model_pretrained_128_4096.pckl"

    def configure_optimizers(
        self,
    ) -> OptimizerLRScheduler:
        """Configure the Adam optimizer for training phases 2 and 3.

        Does not train the discrete model (phase 1)

        Returns:
            dict[str, torch.optim.Optimizer | torch.optim.lr_scheduler.LRScheduler]:
                Dictionary containing optimizers and schedulers.

        Example:
            >>> model = Dsr()
            >>> optimizers = model.configure_optimizers()
            >>> isinstance(optimizers, tuple)
            True
            >>> len(optimizers)
            2
        """
        num_steps = max(
            self.trainer.max_steps // len(self.trainer.datamodule.train_dataloader()),
            self.trainer.max_epochs,
        )
        self.second_phase = int(num_steps * self.upsampling_train_ratio)
        anneal = int(0.8 * self.second_phase)
        optimizer_d = torch.optim.Adam(
            params=list(self.model.image_reconstruction_network.parameters())
            + list(self.model.subspace_restriction_module_hi.parameters())
            + list(self.model.subspace_restriction_module_lo.parameters())
            + list(self.model.anomaly_detection_module.parameters()),
            lr=0.0002,
        )
        scheduler_d = torch.optim.lr_scheduler.StepLR(optimizer_d, anneal, gamma=0.1)

        optimizer_u = torch.optim.Adam(params=self.model.upsampling_module.parameters(), lr=0.0002)

        return ({"optimizer": optimizer_d, "lr_scheduler": scheduler_d}, {"optimizer": optimizer_u})

    def on_train_start(self) -> None:
        """Load pretrained weights of the discrete model when starting training."""
        ckpt: Path = self.prepare_pretrained_model()
        self.model.load_pretrained_discrete_model_weights(ckpt, self.device)

    def on_train_epoch_start(self) -> None:
        """Display a message when starting to train the upsampling module."""
        if self.current_epoch == self.second_phase:
            logger.info("Now training upsampling module.")

    def training_step(self, batch: Batch) -> STEP_OUTPUT:
        """Training Step of DSR.

        During the first phase, feeds the original image and simulated anomaly
        mask. During second phase, feeds a generated anomalous image to train
        the upsampling module.

        Args:
            batch (Batch): Input batch containing image, label and mask

        Returns:
            STEP_OUTPUT: Dictionary containing the loss value

        Example:
            >>> from anomalib.data import Batch
            >>> model = Dsr()
            >>> batch = Batch(
            ...     image=torch.randn(8, 3, 256, 256),
            ...     label=torch.zeros(8)
            ... )
            >>> output = model.training_step(batch)
            >>> isinstance(output, dict)
            True
            >>> "loss" in output
            True
        """
        ph1_opt, ph2_opt = self.optimizers()

        if self.current_epoch < self.second_phase:
            # we are not yet training the upsampling module: we are only using
            # the first optimizer
            input_image = batch.image
            # Create anomaly masks
            anomaly_mask = self.quantized_anomaly_generator.augment_batch(input_image)
            # Generate model prediction
            model_outputs = self.model(input_image, anomaly_mask)
            # Compute loss
            loss = self.second_stage_loss(
                model_outputs["recon_feat_hi"],
                model_outputs["recon_feat_lo"],
                model_outputs["embedding_bot"],
                model_outputs["embedding_top"],
                input_image,
                model_outputs["obj_spec_image"],
                model_outputs["anomaly_map"],
                model_outputs["true_anomaly_map"],
            )

            # compute manual optimizer step
            ph1_opt.zero_grad()
            self.manual_backward(loss)
            ph1_opt.step()

        else:
            # we are training the upsampling module
            input_image = batch.image
            # Generate anomalies
            input_image, anomaly_maps = self.perlin_generator(input_image)
            # Get model prediction
            model_outputs = self.model(input_image)
            # Calculate loss
            loss = self.third_stage_loss(model_outputs["anomaly_map"], anomaly_maps)

            # compute manual optimizer step
            ph2_opt.zero_grad()
            self.manual_backward(loss)
            ph2_opt.step()

        self.log("train_loss", loss.item(), on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss}

    def validation_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        """Validation step of DSR.

        The Softmax predictions of the anomalous class are used as anomaly map.

        Args:
            batch (Batch): Input batch containing image, label and mask
            *args: Additional positional arguments (unused)
            **kwargs: Additional keyword arguments (unused)

        Returns:
            STEP_OUTPUT: Dictionary containing predictions and batch information

        Example:
            >>> from anomalib.data import Batch
            >>> model = Dsr()
            >>> batch = Batch(
            ...     image=torch.randn(8, 3, 256, 256),
            ...     label=torch.zeros(8)
            ... )
            >>> output = model.validation_step(batch)
            >>> isinstance(output, Batch)
            True
        """
        del args, kwargs  # These variables are not used.

        predictions = self.model(batch.image)
        return batch.update(**predictions._asdict())

    @property
    def trainer_arguments(self) -> dict[str, Any]:
        """Required trainer arguments.

        Returns:
            dict[str, Any]: Dictionary of trainer arguments

        Example:
            >>> model = Dsr()
            >>> model.trainer_arguments
            {'num_sanity_val_steps': 0}
        """
        return {"num_sanity_val_steps": 0}

    @property
    def learning_type(self) -> LearningType:
        """Return the learning type of the model.

        Returns:
            LearningType: Learning type of the model.

        Example:
            >>> model = Dsr()
            >>> model.learning_type
            <LearningType.ONE_CLASS: 'one_class'>
        """
        return LearningType.ONE_CLASS

    @classmethod
    def configure_pre_processor(cls, image_size: tuple[int, int] | None = None) -> PreProcessor:
        """Configure default pre-processor for DSR.

        Note:
            Imagenet normalization is not used in this model.

        Args:
            image_size (tuple[int, int] | None, optional): Target image size.
                Defaults to ``(256, 256)``.

        Returns:
            PreProcessor: Configured pre-processor with resize transform.
        """
        image_size = image_size or (256, 256)
        transform = Compose([Resize(image_size, antialias=True)])
        return PreProcessor(transform=transform)
