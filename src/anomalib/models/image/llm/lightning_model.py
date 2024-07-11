"""Llm: Zero-/Few-Shot Anomaly Classification and Segmentation.

Paper No paper
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from lightning.pytorch.utilities.types import STEP_OUTPUT

import base64
import json
import logging

from anomalib.metrics.threshold import ManualThreshold
import requests
import torch
from torchvision.transforms.v2 import (
    Compose,
    InterpolationMode,
    Normalize,
    Resize,
    Transform,
)

from anomalib import LearningType
from torch.utils.data import DataLoader
from anomalib.models.components import AnomalyModule
from anomalib.engine.engine import UnassignedError

# from .torch_model import openAI # TODO: This is necesary

logger = logging.getLogger(__name__)

__all__ = ["Llm"]

model_str = "openAI"


def api_call_few_shot(key, pre_img, prompt, image) -> str:

    # Function to encode the image
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    # Path to your image

    # Getting the base64 string
    base64_image = encode_image(image)
    base64_image_pre = []
    for i in pre_img:
        base64_image_pre.append(encode_image(i))

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}",
    }

    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You will receive an image that is going to be an example of the typical image without any anomaly, \nand the last image that you need to decide if it has an anomaly or not.\nAnswer with a 'NO' if it does not have any anomalies and 'YES: description' where description is a description of the anomaly provided, position.",
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image_pre[0]}",
                            "detail": "high",  # low, high
                        },
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}",
                            "detail": "high",  # low, high
                        },
                    },
                ],
            },
        ],
        "max_tokens": 300,
    }

    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=10,
        )
        response.raise_for_status()  # This will raise an HTTPError if the response was an HTTP error
        return json.loads(response.content)["choices"][-1]["message"]["content"]
    except requests.exceptions.RequestException as e:
        # Handle all types of request exceptions here
        print(f"An error occurred: {e}")
        raise


def api_call(key, prompt, image) -> str:
    prompt = """
Examine the provided image carefully to determine if there is an obvious anomaly present. Anomalies may include mechanical malfunctions, unexpected objects, safety hazards, structural damages, or unusual patterns or defects in the objects.

Instructions:

1. Thoroughly inspect the image for any irregularities or deviations from normal operating conditions.

2. Clearly state if an obvious anomaly is detected.
   - If an anomaly is detected, begin with 'YES,' followed by a detailed description of the anomaly.
   - If no anomaly is detected, simply state 'NO' and end the analysis.

Example Output Structure:

'YES:
- Description: Conveyor belt misalignment causing potential blockages. This may result in production delays and equipment damage. Immediate realignment and inspection are recommended.'

'NO'

Considerations:

- Ensure accuracy in identifying anomalies to prevent overlooking critical issues.
- Provide clear and concise descriptions for any detected anomalies.
- Focus on obvious anomalies that could impact final use of the object operation or safety.
"""

    # Function to encode the image
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    # Path to your image

    # Getting the base64 string
    base64_image = encode_image(image)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}",
    }

    payload = {
        "model": "gpt-4o",
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
                            "detail": "high",  # low, high
                        },
                    },
                ],
            },
        ],
        "max_tokens": 300,
    }

    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=10,
        )
        response.raise_for_status()  # This will raise an HTTPError if the response was an HTTP error
        return json.loads(response.content)["choices"][-1]["message"]["content"]
    except requests.exceptions.RequestException as e:
        # Handle all types of request exceptions here
        print(f"An error occurred: {e}")
        raise


class Llm(AnomalyModule):
    """Llm Lightning model.

    Args:
        openai_key(str): The key to interact with openai,
                         https://platform.openai.com/docs/quickstart/step-2-set-up-your-api-key .
    """

    def __init__(
        self,
        k_shot=0,
        openai_key: None | str = "",
    ) -> None:
        super().__init__()

        self.k_shot = k_shot
        self.model_str = model_str
        # OpenAI API Key
        if not openai_key:
            msg = "OpenAI key not found."
            raise UnassignedError(msg)

        self.openai_key = openai_key
        self.image_threshold = ManualThreshold()

    def _setup(self):
        dataloader = self.trainer.datamodule.train_dataloader()
        pre_images = self.collect_reference_images(dataloader)
        self.pre_images = pre_images

    def training_step(
        self, batch: dict[str, str | torch.Tensor], *args, **kwargs
    ) -> None:
        """Train Step of LLM."""
        del args, kwargs  # These variables are not used.
        # no train on llm
        return batch

    @staticmethod
    def configure_optimizers() -> None:
        """WinCLIP doesn't require optimization, therefore returns no optimizers."""
        return

    def validation_step(
        self, batch: dict[str, str | list[str] | torch.Tensor], *args, **kwargs
    ) -> STEP_OUTPUT:
        """Get batch of anomaly maps from input image batch.

        Args:
            batch (dict[str, str | list[str] | torch.Tensor]): Batch containing image filename, image, label and mask
            args: Additional arguments.
            kwargs: Additional keyword arguments.

        Returns:
            dict[str, Any]: str_otput and pred_scores, the output of the Llm and pred_scores 1.0 if is an anomaly image.
        """
        del args, kwargs  # These variables are not used.
        bsize = len(batch["image_path"])
        out_list: list[str] = []
        pred_list: list[float] = []
        for i in range(bsize):
            try:
                if self.k_shot > 0:
                    output = str(
                        api_call_few_shot(
                            self.openai_key,
                            self.pre_images,
                            "",
                            batch["image_path"][i],
                        ),
                    ).strip()
                else:
                    output = str(
                        api_call(self.openai_key, "", batch["image_path"][i]),
                    ).strip()
            except Exception as e:
                print(f"Error:img_path:{batch['image_path']}")
                logging.exception(
                    f"Error calling openAI API for image {batch['image_path'][i]}: {e}",
                )
                output = "Error"

            prediction = 0.5
            if output.startswith("N"):

                prediction = 0.0
            elif output.startswith("Y"):
                prediction = 1.0

            out_list.append(output)
            pred_list.append(prediction)
            logging.debug(f"Output: {output}, Prediction: {prediction}")
            # print( f"img_path:{batch['image_path'][i]}, Output: {output}, Prediction: {prediction}")

        batch["str_output"] = out_list
        batch["pred_scores"] = torch.tensor(pred_list).to(self.device)
        batch["pred_labels"] = torch.tensor(pred_list).to(self.device)
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
        return LearningType.ZERO_SHOT if self.k_shot == 0 else LearningType.FEW_SHOT

    def collect_reference_images(self, dataloader: DataLoader) -> torch.Tensor:
        """Collect reference images for few-shot inference.

        The reference images are collected by iterating the training dataset until the required number of images are
        collected.

        Returns:
            ref_images (Tensor): A tensor containing the reference images.
        """
        ref_images = []
        for batch in dataloader:
            images = batch["image_path"][: self.k_shot - len(ref_images)]
            ref_images.extend(images)
            if self.k_shot == len(ref_images):
                break
        return ref_images
