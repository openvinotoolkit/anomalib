"""Llm: Zero-/Few-Shot Anomaly Classification and Segmentation.

Paper No paper
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import base64
import json
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any

import requests
import torch
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import Compose, InterpolationMode, Normalize, Resize, Transform

from anomalib import LearningType
from anomalib.data.predict import PredictDataset
from anomalib.models.components import AnomalyModule

from ollama import generate

# from .torch_model import openAI # TODO: This is necesary

logger = logging.getLogger(__name__)

__all__ = ["Llmollama"]


def api_call(prompt, image) -> str:
    prompt = "Describe me if this image has an obious anomaly or not. if yes say 'YES:', follow by a description, and if not say 'NO' and finish."

    # Function to encode the image
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    # Path to your image

    # Getting the base64 string
    base64_image = encode_image(image)
    # base64_image = base64.b64encode(image).decode('utf-8')




    response = generate('llava', f'{prompt}', images=[base64_image], stream=False)

    return response


class Llmollama(AnomalyModule):
    """Llmollama Lightning model.

    Args:
        openai_key(str): The key to interact with openai,
                         https://platform.openai.com/docs/quickstart/step-2-set-up-your-api-key .
    """

    def __init__(
        self,
    ) -> None:
        super().__init__()

    def training_step(self, batch: dict[str, str | torch.Tensor], *args, **kwargs) -> None:
        """train Step of LLM."""
        del args, kwargs  # These variables are not used.
        # no train on llm
        return batch

    @staticmethod
    def configure_optimizers() -> None:
        """WinCLIP doesn't require optimization, therefore returns no optimizers."""
        return

    def validation_step(self, batch: dict[str, str | torch.Tensor], *args, **kwargs) -> dict:
        """Validation Step of WinCLIP."""
        del args, kwargs  # These variables are not used.
        bsize = len(batch["image_path"])
        # long_text = "This is a very long text that might not fit well in a single line in a subplot. So it needs to be wrapped properly to ensure it fits within the plotting area without looking cluttered.This is a very long text that might not fit well in a single line in a subplot. So it needs to be wrapped properly to ensure it fits within the plotting area without looking cluttered.This is a very long text that might not fit well in a single line in a subplot. So it needs to be wrapped properly to ensure it fits within the plotting area without looking cluttered."

        #batch["str_output"] =[f'{long_text}']*bsize
        batch["str_output"] =[api_call( "", batch["image_path"][0])]*bsize  # the first img of the batch
        batch["pred_scores"] = torch.tensor([0.9]*bsize)
        return batch

    @property
    def trainer_arguments(self) -> dict[str, int | float]:
        """Set model-specific trainer arguments."""
        return {}

    @property
    def learning_type(self) -> LearningType:
        """The learning type of the model.

        Llm is a zero-/few-shot model, depending on the user configuration. Therefore, the learning type is
        set to ``LearningType.FEW_SHOT`` when ``k_shot`` is greater than zero and ``LearningType.ZERO_SHOT`` otherwise.
        """
        return LearningType.ZERO_SHOT

    def configure_transforms(self, image_size: tuple[int, int] | None = None) -> Transform:
        """Configure the default transforms used by the model."""
        if image_size is not None:
            logger.warning("Image size is not used in WinCLIP. The input image size is determined by the model.")
        return Compose(
            [
                Resize((240, 240), antialias=True, interpolation=InterpolationMode.BICUBIC),
                Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
            ],
        )
