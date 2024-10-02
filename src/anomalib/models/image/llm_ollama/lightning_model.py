"""Llm: Zero-/Few-Shot Anomaly Classification and Segmentation.

Paper No paper
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import base64
import logging

import ollama
import torch
from ollama import generate
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import Compose, InterpolationMode, Normalize, Resize, Transform

from anomalib import LearningType
from anomalib.models.components import AnomalyModule

# from .torch_model import openAI # TODO: This is necesary

logger = logging.getLogger(__name__)

__all__ = ["Llmollama"]

# model_str = "llava:34b"
model_str = "llava:latest"


def api_call_fewShot(preImages, prompt, image) -> str:
    prompt = "Describe me if this image has an obious anomaly or not. if yes say 'YES:', follow by a description, and if not say 'NO' and finish."

    # Function to encode the image
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    # Path to your image

    # Getting the base64 string
    base64_image = encode_image(image)
    base64_image_pre = []
    for i in preImages:
        base64_image_pre.append(encode_image(i))
    # base64_image = base64.b64encode(image).decode('utf-8')

    # response = generate('llava', f'{prompt}', images=[base64_image], stream=False)
    # response = generate('llava:34b', f'{prompt}', images=[base64_image], stream=False)

    res = ollama.chat(
        model=model_str,
        messages=[
            {
                "role": "user",
                "images": base64_image_pre,
                "content": "",
            },
            {
                "role": "user",
                "images": [],
                "content": "This is a sample of a normal picture without any anomalies.",
            },
            {
                "role": "user",
                "images": [base64_image],
                "content": "",
            },
            {
                "role": "user",
                "images": [],
                "content": prompt,
            },
        ],
    )

    return res


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

    response = generate(model_str, f"{prompt}", images=[base64_image], stream=False)
    # response = generate('llava:34b', f'{prompt}', images=[base64_image], stream=False)

    return response


class Llmollama(AnomalyModule):
    """Llmollama Lightning model.

    Args:
        openai_key(str): The key to interact with openai,
                         https://platform.openai.com/docs/quickstart/step-2-set-up-your-api-key .
    """

    def __init__(
        self,
        k_shot=0,
    ) -> None:
        super().__init__()
        self.k_shot = k_shot
        self.model_str = model_str

    def _setup(self):
        dataloader = self.trainer.datamodule.train_dataloader()
        pre_images = self.collect_reference_images(dataloader)
        self.pre_images = pre_images

    def training_step(self, batch: dict[str, str | torch.Tensor], *args, **kwargs) -> None:
        """Train Step of LLM."""
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
            o = "NO - default"
            if self.k_shot > 0:
                o = str(api_call_fewShot(self.pre_images, "", batch["image_path"][x])["message"]["content"]).strip()
            else:
                o = str(api_call("", batch["image_path"][x])["response"]).strip()
            p = 0.0 if o.startswith("N") else 1.0
            out_list.append(o)
            pred_list.append(p)

        batch["str_output"] = out_list
        # [api_call( "", batch["image_path"][0])]*bsize  # the first img of the batch
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
        return LearningType.FEW_SHOT if self.k_shot else LearningType.ZERO_SHOT

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
