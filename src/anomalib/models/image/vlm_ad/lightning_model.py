"""Vision Language Model (VLM) based Anomaly Detection.

This module implements anomaly detection using Vision Language Models (VLMs) like
GPT-4V, LLaVA, etc. The models use natural language prompting to detect anomalies
in images by comparing them with reference normal images.

The module supports both zero-shot and few-shot learning approaches:

- Zero-shot: No reference images needed
- Few-shot: Uses ``k`` reference normal images for better context

Example:
    >>> from anomalib.models.image import VlmAd
    >>> from anomalib.data import MVTecAD
    >>> from anomalib.engine import Engine

    >>> model = VlmAd(  # doctest: +SKIP
    ...     model="gpt-4-vision-preview",
    ...     api_key="YOUR_API_KEY",
    ...     k_shot=3
    ... )
    >>> datamodule = MVTecAD()

    >>> engine = Engine()
    >>> predictions = engine.predict(model=model, datamodule=datamodule)  # doctest: +SKIP

See Also:
    - :class:`VlmAd`: Main model class for VLM-based anomaly detection
    - :mod:`.backends`: Different VLM backend implementations
    - :mod:`.utils`: Utility functions for prompting and responses
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging

import torch
from torch.utils.data import DataLoader

from anomalib import LearningType
from anomalib.data import ImageBatch
from anomalib.metrics import Evaluator, F1Score
from anomalib.models import AnomalibModule
from anomalib.post_processing import PostProcessor

from .backends import Backend, ChatGPT, Huggingface, Ollama
from .utils import ModelName, Prompt

logger = logging.getLogger(__name__)


class VlmAd(AnomalibModule):
    """Vision Language Model (VLM) based anomaly detection model.

    This model uses VLMs like GPT-4V, LLaVA, etc. to detect anomalies in images by
    comparing them with reference normal images through natural language prompting.

    Args:
        model (ModelName | str): Name of the VLM model to use. Can be one of:
            - ``ModelName.LLAMA_OLLAMA``
            - ``ModelName.GPT_4O_MINI``
            - ``ModelName.VICUNA_7B_HF``
            - ``ModelName.VICUNA_13B_HF``
            - ``ModelName.MISTRAL_7B_HF``
            Defaults to ``ModelName.LLAMA_OLLAMA``.
        api_key (str | None, optional): API key for models that require
            authentication. Defaults to None.
        k_shot (int, optional): Number of reference normal images to use for
            few-shot learning. If 0, uses zero-shot approach. Defaults to 0.

    Example:
        >>> from anomalib.models.image import VlmAd
        >>> # Zero-shot approach
        >>> model = VlmAd(  # doctest: +SKIP
        ...     model="gpt-4-vision-preview",
        ...     api_key="YOUR_API_KEY"
        ... )
        >>> # Few-shot approach with 3 reference images
        >>> model = VlmAd(  # doctest: +SKIP
        ...     model="gpt-4-vision-preview",
        ...     api_key="YOUR_API_KEY",
        ...     k_shot=3
        ... )

    Raises:
        ValueError: If an unsupported VLM model is specified.
    """

    def __init__(
        self,
        model: ModelName | str = ModelName.LLAMA_OLLAMA,
        api_key: str | None = None,
        k_shot: int = 0,
    ) -> None:
        super().__init__()
        self.k_shot = k_shot
        model = ModelName(model)
        self.vlm_backend: Backend = self._setup_vlm_backend(model, api_key)

    @staticmethod
    def _setup_vlm_backend(model_name: ModelName, api_key: str | None) -> Backend:
        if model_name == ModelName.LLAMA_OLLAMA:
            return Ollama(model_name=model_name.value)
        if model_name == ModelName.GPT_4O_MINI:
            return ChatGPT(api_key=api_key, model_name=model_name.value)
        if model_name in {ModelName.VICUNA_7B_HF, ModelName.VICUNA_13B_HF, ModelName.MISTRAL_7B_HF}:
            return Huggingface(model_name=model_name.value)

        msg = f"Unsupported VLM model: {model_name}"
        raise ValueError(msg)

    def _setup(self) -> None:
        if self.k_shot > 0 and self.vlm_backend.num_reference_images != self.k_shot:
            logger.info("Collecting reference images from training dataset.")
            dataloader = self.trainer.datamodule.train_dataloader()
            self.collect_reference_images(dataloader)

    def collect_reference_images(self, dataloader: DataLoader) -> None:
        """Collect reference images for few-shot inference.

        Args:
            dataloader (DataLoader): DataLoader containing normal images for
                reference.
        """
        for batch in dataloader:
            for img_path in batch.image_path:
                self.vlm_backend.add_reference_images(img_path)
                if self.vlm_backend.num_reference_images == self.k_shot:
                    return

    @property
    def prompt(self) -> Prompt:
        """Get the prompt for VLM interaction.

        Returns:
            Prompt: Object containing prompts for prediction and few-shot learning.
        """
        return Prompt(
            predict=(
                "You are given an image. It is either normal or anomalous."
                " First say 'YES' if the image is anomalous, or 'NO' if it is normal.\n"
                "Then give the reason for your decision.\n"
                "For example, 'YES: The image has a crack on the wall.'"
            ),
            few_shot=(
                "These are a few examples of normal picture without any anomalies."
                " You have to use these to determine if the image I provide in the next"
                " chat is normal or anomalous."
            ),
        )

    def validation_step(self, batch: ImageBatch, *args, **kwargs) -> ImageBatch:
        """Perform validation step.

        Args:
            batch (ImageBatch): Batch of images to validate.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            ImageBatch: Batch with predictions and explanations added.
        """
        del args, kwargs  # These variables are not used.
        assert batch.image_path is not None
        responses = [(self.vlm_backend.predict(img_path, self.prompt)) for img_path in batch.image_path]
        batch.explanation = responses
        batch.pred_label = torch.tensor([1.0 if r.startswith("Y") else 0.0 for r in responses], device=self.device)
        return batch

    @property
    def learning_type(self) -> LearningType:
        """Get the learning type of the model.

        Returns:
            LearningType: ZERO_SHOT if k_shot=0, else FEW_SHOT.
        """
        return LearningType.ZERO_SHOT if self.k_shot == 0 else LearningType.FEW_SHOT

    @property
    def trainer_arguments(self) -> dict[str, int | float]:
        """Get trainer arguments.

        Returns:
            dict[str, int | float]: Empty dict as no training is needed.
        """
        return {}

    @staticmethod
    def configure_transforms(image_size: tuple[int, int] | None = None) -> None:
        """Configure image transforms.

        Args:
            image_size (tuple[int, int] | None, optional): Ignored as each backend
                has its own transforms. Defaults to None.
        """
        if image_size is not None:
            logger.warning("Ignoring image_size argument as each backend has its own transforms.")

    @classmethod
    def configure_post_processor(cls) -> PostProcessor | None:
        """Configure post processor.

        Returns:
            PostProcessor | None: None as post processing is not required.
        """
        return None

    @staticmethod
    def configure_evaluator() -> Evaluator:
        """Configure default evaluator.

        Returns:
            Evaluator: Evaluator configured with F1Score metric.
        """
        image_f1score = F1Score(fields=["pred_label", "gt_label"], prefix="image_")
        return Evaluator(test_metrics=image_f1score)

    @staticmethod
    def _export_not_supported_message() -> None:
        logging.warning("Exporting the model is not supported for VLM-AD model. Skipping...")

    def to_torch(self, *_, **__) -> None:  # type: ignore[override]
        """Skip export to torch."""
        return self._export_not_supported_message()

    def to_onnx(self, *_, **__) -> None:  # type: ignore[override]
        """Skip export to onnx."""
        return self._export_not_supported_message()

    def to_openvino(self, *_, **__) -> None:  # type: ignore[override]
        """Skip export to openvino."""
        return self._export_not_supported_message()
