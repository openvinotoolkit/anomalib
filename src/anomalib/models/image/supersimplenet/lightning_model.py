"""SuperSimpleNet: Unifying Unsupervised and Supervised Learning for Fast and Reliable Surface Defect Detection.

This module implements the SuperSimpleNet model for surface defect / anomaly detection.
SuperSimpleNet is a simple yet strong discriminative model consisting of a pretrained feature extractor with upscaling,
feature adaptor, train-time feature-level synthetic anomaly generation module, and segmentation-detection module.

Using the adapted features, the model predicts an anomaly map via the segmentation head
and an anomaly score using the classification head.
It delivers strong performance while maintaining fast inference.

Example:
    >>> from anomalib.data import MVTecAD
    >>> from anomalib.models import Supersimplenet
    >>> from anomalib.engine import Engine

    >>> datamodule = MVTecAD()
    >>> model = Supersimplenet()
    >>> engine = Engine()

    >>> engine.fit(model, datamodule=datamodule)  # doctest: +SKIP
    >>> predictions = engine.predict(model, datamodule=datamodule)  # doctest: +SKIP


Paper:
    Title: SuperSimpleNet: Unifying Unsupervised and Supervised Learning for Fast and Reliable Surface Defect Detection.
    URL: https://arxiv.org/pdf/2408.03143

Notes:
    This implementation supports both unsupervised and supervised setting,
    but Anomalib currently supports only unsupervised learning.

See Also:
    :class:`anomalib.models.image.supersimplenet.torch_model.SupersimplenetModel`:
        PyTorch implementation of the SuperSimpleNet model.
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from torchvision.transforms.v2 import Compose, Normalize, Resize

from anomalib import LearningType
from anomalib.data import Batch
from anomalib.metrics import Evaluator
from anomalib.models import AnomalibModule
from anomalib.post_processing import PostProcessor
from anomalib.pre_processing import PreProcessor
from anomalib.visualization import Visualizer

from .loss import SSNLoss
from .torch_model import SupersimplenetModel


class Supersimplenet(AnomalibModule):
    """PL Lightning Module for the SuperSimpleNet model.

    Args:
        perlin_threshold (float): threshold value for Perlin noise thresholding during anomaly generation.
        backbone (str): backbone name
        layers (list[str]): backbone layers utilised
        supervised (bool): whether the model will be trained in supervised mode. False by default (unsupervised).
        pre_processor (PreProcessor | bool, optional): Pre-processor instance or
            flag to use default. Defaults to ``True``.
        post_processor (PostProcessor | bool, optional): Post-processor instance
            or flag to use default. Defaults to ``True``.
        evaluator (Evaluator | bool, optional): Evaluator instance or flag to use
            default. Defaults to ``True``.
        visualizer (Visualizer | bool, optional): Visualizer instance or flag to
            use default. Defaults to ``True``.
    """

    def __init__(
        self,
        perlin_threshold: float = 0.2,
        backbone: str = "wide_resnet50_2",
        layers: list[str] = ["layer2", "layer3"],  # noqa: B006
        supervised: bool = False,
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
        self.supervised = supervised
        # stop grad in unsupervised
        if supervised:
            stop_grad = False
            self.norm_clip_val = 1
        else:
            stop_grad = True
            self.norm_clip_val = 0

        self.model = SupersimplenetModel(
            perlin_threshold=perlin_threshold,
            backbone=backbone,
            layers=layers,
            stop_grad=stop_grad,
        )
        self.loss = SSNLoss()

    def training_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        """Perform the training step input and return the loss.

        Args:
            batch (batch: dict[str, str | torch.Tensor]): Input batch
            args: Additional arguments.
            kwargs: Additional keyword arguments.

        Returns:
            STEP_OUTPUT: Dictionary containing the loss value.
        """
        del args, kwargs  # These variables are not used.

        anomaly_map, anomaly_score, masks, labels = self.model(
            images=batch.image,
            masks=batch.gt_mask,
            labels=batch.gt_label,
        )
        loss = self.loss(pred_map=anomaly_map, pred_score=anomaly_score, target_mask=masks, target_label=labels)
        self.log("train_loss", loss.item(), on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss}

    def validation_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        """Perform the validation step and return the anomaly map and anomaly score.

        Args:
            batch (dict[str, str | torch.Tensor]): Input batch
            args: Additional arguments.
            kwargs: Additional keyword arguments.

        Returns:
            STEP_OUTPUT | None: batch dictionary containing anomaly-maps.
        """
        # These variables are not used.
        del args, kwargs

        # Get anomaly maps and predicted scores from the model.
        predictions = self.model(batch.image)

        return batch.update(**predictions._asdict())

    @property
    def trainer_arguments(self) -> dict[str, Any]:
        """Return SuperSimpleNet trainer arguments."""
        return {"gradient_clip_val": self.norm_clip_val, "num_sanity_val_steps": 0}

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Configure AdamW optimizer and MultiStepLR scheduler."""
        optimizer = AdamW(
            [
                {
                    "params": self.model.adaptor.parameters(),
                    "lr": 0.0001,
                },
                {
                    "params": self.model.segdec.parameters(),
                    "lr": 0.0002,
                    "weight_decay": 0.00001,
                },
            ],
        )
        scheduler = MultiStepLR(
            optimizer,
            milestones=[int(self.trainer.max_epochs * 0.8), int(self.trainer.max_epochs * 0.9)],
            gamma=0.4,
        )
        return [optimizer], [scheduler]

    @property
    def learning_type(self) -> LearningType:
        """Return the learning type of the model.

        This is subject to change in the future when support for supervised training is introduced.

        Returns:
            LearningType: Learning type of the model.
        """
        return LearningType.ONE_CLASS

    @classmethod
    def configure_pre_processor(cls, image_size: tuple[int, int] | None = None) -> PreProcessor:
        """Configure the default pre-processor for SuperSimpleNet.

        Pre-processor resizes images and normalizes using ImageNet statistics.

        Args:
            image_size (tuple[int, int] | None, optional): Target size for
                resizing. Defaults to ``(256, 256)``.

        Returns:
            PreProcessor: Configured SuperSimpleNet pre-processor
        """
        image_size = image_size or (256, 256)
        return PreProcessor(
            transform=Compose([
                Resize(image_size, antialias=True),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]),
        )
