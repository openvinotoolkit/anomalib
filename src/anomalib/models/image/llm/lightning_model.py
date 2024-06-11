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

# from .torch_model import openAI # TODO: This is necesary

logger = logging.getLogger(__name__)

__all__ = ["Llm"]


def api_call(key, prompt, image) -> str:
    prompt = "Describe me if this image has an obious anomaly or not. if yes say 'YES:', follow by a description, and if not say 'NO' and finish."

    # Function to encode the image
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    # Path to your image

    # Getting the base64 string
    base64_image = encode_image(image)
    # base64_image = base64.b64encode(image).decode('utf-8')

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}",
    }

    payload = {
        "model": "gpt-4-turbo",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "low", # low, high
                        },
                    },
                ],
            },
        ],
        "max_tokens": 300,
    }
    #print(headers)

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    #print(json.loads(response.content))
    if not response.content or not json.loads(response.content).get("choices"):
        print("error")
        print(json.loads(response.content))
        return "No : ERROR on url"

    return json.loads(response.content)["choices"][-1]["message"]["content"]

    return json.dumps(headers)


class Llm(AnomalyModule):
    """Llm Lightning model.

    Args:
        openai_key(str): The key to interact with openai,
                         https://platform.openai.com/docs/quickstart/step-2-set-up-your-api-key .
    """

    def __init__(
        self,
        openai_key: str = "",
    ) -> None:
        super().__init__()
        # self.model = openAI()
        # OpenAI API Key
        self.openai_key = openai_key

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
        out_list: list[str] = []
        pred_list = []
        for x in range(bsize):
            o = str(api_call( self.openai_key,  "", batch["image_path"][x])).strip()
            p = 0.0 if o.startswith("N") else 1.0
            out_list.append(o)
            pred_list.append(p)
            print(o)
            print(p)

        batch["str_output"] = str(out_list)
        batch["pred_scores"] = torch.tensor(pred_list)
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
