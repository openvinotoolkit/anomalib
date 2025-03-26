"""WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation.

This module implements the WinCLIP model for zero-shot and few-shot anomaly
detection using CLIP embeddings and a sliding window approach.

The model can perform both anomaly classification and segmentation tasks by
comparing image regions with normal reference examples through CLIP embeddings.

Example:
    >>> from anomalib.data import MVTecAD
    >>> from anomalib.engine import Engine
    >>> from anomalib.models.image import WinClip

    >>> datamodule = MVTecAD(root="./datasets/MVTecAD")  # doctest: +SKIP
    >>> model = WinClip()  # doctest: +SKIP

    >>> Engine.test(model=model, datamodule=datamodule)  # doctest: +SKIP

Paper:
    WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation
    https://arxiv.org/abs/2303.14814

See Also:
    - :class:`WinClip`: Main model class for WinCLIP-based anomaly detection
    - :class:`WinClipModel`: PyTorch implementation of the WinCLIP model
"""

# Copyright (C) 2024-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import Compose, InterpolationMode, Normalize, Resize

from anomalib import LearningType
from anomalib.data import Batch
from anomalib.data.predict import PredictDataset
from anomalib.metrics import Evaluator
from anomalib.models.components import AnomalibModule
from anomalib.post_processing import PostProcessor
from anomalib.pre_processing import PreProcessor
from anomalib.visualization import Visualizer

from .torch_model import WinClipModel

logger = logging.getLogger(__name__)

__all__ = ["WinClip"]


class WinClip(AnomalibModule):
    """WinCLIP Lightning model.

    This model implements the WinCLIP algorithm for zero-/few-shot anomaly detection using CLIP
    embeddings and a sliding window approach. The model can perform both anomaly classification
    and segmentation by comparing image regions with normal reference examples.

    Args:
        class_name (str | None, optional): Name of the object class used in the prompt
            ensemble. If not provided, will try to infer from the datamodule or use "object"
            as default. Defaults to ``None``.
        k_shot (int, optional): Number of reference images to use for few-shot inference.
            If 0, uses zero-shot approach. Defaults to ``0``.
        scales (tuple[int], optional): Scales of sliding windows used for multiscale anomaly
            detection. Defaults to ``(2, 3)``.
        few_shot_source (str | Path | None, optional): Path to folder containing reference
            images for few-shot inference. If not provided, reference images are sampled from
            training data. Defaults to ``None``.
        pre_processor (PreProcessor | bool, optional): Pre-processor instance or flag to use
            default. Used to pre-process input data before model inference. Defaults to
            ``True``.
        post_processor (PostProcessor | bool, optional): Post-processor instance or flag to
            use default. Used to post-process model predictions. Defaults to ``True``.
        evaluator (Evaluator | bool, optional): Evaluator instance or flag to use default.
            Used to compute metrics. Defaults to ``True``.
        visualizer (Visualizer | bool, optional): Visualizer instance or flag to use default.
            Used to create visualizations. Defaults to ``True``.

    Example:
        >>> from anomalib.models.image import WinClip
        >>> # Zero-shot approach
        >>> model = WinClip()  # doctest: +SKIP
        >>> # Few-shot with 5 reference images
        >>> model = WinClip(k_shot=5)  # doctest: +SKIP
        >>> # Custom class name
        >>> model = WinClip(class_name="transistor")  # doctest: +SKIP

    Notes:
        - Input image size is fixed at 240x240 and cannot be modified
        - Uses a custom normalization transform specific to CLIP

    See Also:
        - :class:`WinClipModel`: PyTorch implementation of the core model
        - :class:`PostProcessor`: Default post-processor used by WinCLIP
    """

    def __init__(
        self,
        class_name: str | None = None,
        k_shot: int = 0,
        scales: tuple = (2, 3),
        few_shot_source: Path | str | None = None,
        pre_processor: nn.Module | bool = True,
        post_processor: nn.Module | bool = True,
        evaluator: Evaluator | bool = True,
        visualizer: Visualizer | bool = True,
    ) -> None:
        super().__init__(
            pre_processor=pre_processor,
            post_processor=post_processor,
            evaluator=evaluator,
            visualizer=visualizer,
        )

        self.model = WinClipModel(scales=scales, apply_transform=False)
        self.class_name = class_name
        self.k_shot = k_shot
        self.few_shot_source = Path(few_shot_source) if few_shot_source else None
        self.is_setup = False

    def setup(self, stage: str) -> None:
        """Setup WinCLIP model.

        This method:
        - Sets the class name used in the prompt ensemble
        - Collects text embeddings for zero-shot inference
        - Collects reference images for few-shot inference if ``k_shot > 0``

        Note:
            This hook is called before the model is moved to the target device.
        """
        del stage
        if self.is_setup:
            return

        # get class name
        self.class_name = self._get_class_name()
        ref_images = None

        # get reference images
        if self.k_shot:
            if self.few_shot_source:
                logger.info("Loading reference images from %s", self.few_shot_source)
                reference_dataset = PredictDataset(
                    self.few_shot_source,
                    transform=self.pre_processor.test_transform if self.pre_processor else None,
                )
                dataloader = DataLoader(reference_dataset, batch_size=1, shuffle=False)
            else:
                logger.info("Collecting reference images from training dataset")
                dataloader = self.trainer.datamodule.train_dataloader()

            ref_images = self.collect_reference_images(dataloader)

        # call setup to initialize the model
        self.model.setup(self.class_name, ref_images)
        self.is_setup = True

    def _get_class_name(self) -> str:
        """Get the class name used in the prompt ensemble.

        The class name is determined in the following order:
        1. Use class name provided in initialization
        2. Use category name from datamodule if available
        3. Use default value "object"

        Returns:
            str: Class name to use in prompts
        """
        if self.class_name is not None:
            logger.info("Using class name from init args: %s", self.class_name)
            return self.class_name
        if getattr(self, "_trainer", None) and hasattr(self.trainer.datamodule, "category"):
            logger.info("No class name provided, using category from datamodule: %s", self.trainer.datamodule.category)
            return self.trainer.datamodule.category
        logger.info("No class name provided and no category name found in datamodule using default: object")
        return "object"

    def collect_reference_images(self, dataloader: DataLoader) -> torch.Tensor:
        """Collect reference images for few-shot inference.

        Iterates through the training dataset until the required number of reference images
        (specified by ``k_shot``) are collected.

        Args:
            dataloader (DataLoader): DataLoader to collect reference images from

        Returns:
            torch.Tensor: Tensor containing the collected reference images
        """
        ref_images = torch.Tensor()
        for batch in dataloader:
            images = batch.image[: self.k_shot - ref_images.shape[0]]
            ref_images = torch.cat((ref_images, images))
            if self.k_shot == ref_images.shape[0]:
                break
        return ref_images

    @staticmethod
    def configure_optimizers() -> None:
        """Configure optimizers.

        WinCLIP doesn't require optimization, therefore returns no optimizers.
        """
        return

    def validation_step(self, batch: Batch, *args, **kwargs) -> dict:
        """Validation Step of WinCLIP.

        Args:
            batch (Batch): Input batch
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments

        Returns:
            dict: Dictionary containing the batch updated with predictions
        """
        del args, kwargs  # These variables are not used.
        predictions = self.model(batch.image)
        return batch.update(**predictions._asdict())

    @property
    def trainer_arguments(self) -> dict[str, int | float]:
        """Get model-specific trainer arguments.

        Returns:
            dict[str, int | float]: Empty dictionary as WinCLIP needs no special arguments
        """
        return {}

    @property
    def learning_type(self) -> LearningType:
        """Get the learning type of the model.

        Returns:
            LearningType: ``LearningType.FEW_SHOT`` if ``k_shot > 0``, else
                ``LearningType.ZERO_SHOT``
        """
        return LearningType.FEW_SHOT if self.k_shot else LearningType.ZERO_SHOT

    @classmethod
    def configure_pre_processor(cls, image_size: tuple[int, int] | None = None) -> PreProcessor:
        """Configure the default pre-processor used by the model.

        Args:
            image_size (tuple[int, int] | None, optional): Not used as WinCLIP has fixed
                input size. Defaults to ``None``.

        Returns:
            PreProcessor: Configured pre-processor with CLIP-specific transforms
        """
        if image_size is not None:
            logger.warning("Image size is not used in WinCLIP. The input image size is determined by the model.")

        transform = Compose([
            Resize((240, 240), antialias=True, interpolation=InterpolationMode.BICUBIC),
            Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
        ])
        return PreProcessor(transform=transform)

    @staticmethod
    def configure_post_processor() -> PostProcessor:
        """Configure the default post-processor for WinCLIP.

        Returns:
            PostProcessor: Default post-processor instance
        """
        return PostProcessor()
