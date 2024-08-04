"""OpenAI Visual Large Model: Zero-/Few-Shot Anomaly Classification.

Paper (No paper)
"""
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import base64
import logging
from pathlib import Path

import openai
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch.utils.data import DataLoader

from anomalib import LearningType
from anomalib.metrics.threshold import ManualThreshold
from anomalib.models.components import AnomalyModule

logger = logging.getLogger(__name__)

__all__ = ["OpenaiVlm"]


class OpenaiVlm(AnomalyModule):
    """OpenaiVlm Lightning model using OpenAI's GPT-4 for image anomaly detection.

    Args:
        key (str): API key for OpenAI.
                         https://platform.openai.com/docs/quickstart/step-2-set-up-your-api-key
        k_shot(int): The number of images that will compare to detect if it is an anomaly.
    """

    def __init__(
        self,
        k_shot: int = 0,
        openai_key: None | str = "",
    ) -> None:
        super().__init__()

        self.k_shot = k_shot

        # OpenAI API Key
        if not openai_key:
            from anomalib.engine.engine import UnassignedError

            msg = "OpenAI key not found."
            raise UnassignedError(msg)

        self.openai_key = openai_key
        self.openai = openai.OpenAI(api_key=openai_key)
        self.openai.api_key = self.openai_key
        self.image_threshold = ManualThreshold()

    def _setup(self) -> None:
        dataloader = self.trainer.datamodule.train_dataloader()
        pre_images = self.collect_reference_images(dataloader)
        self.pre_images = pre_images

    def api_call_few_shot(self, pre_img: str, image: str) -> str:
        """Makes an API call to OpenAI's GPT-4 model to detect anomalies in an image.

        Args:
            key (str): API key for OpenAI.
            pre_img (list): List of paths to images that serve as examples of typical images without anomalies.
            prompt (str): The prompt to provide to the GPT-4 model (not used in the current implementation).
            image (str): Path to the image that needs to be checked for anomalies.

        Returns:
            str: The response from the GPT-4 model indicating whether the image has anomalies or not.
                  It returns 'NO' if there are no anomalies and 'YES: description' if there are anomalies,
                  where 'description' provides details of the anomaly and its position.

        Raises:
            openai.error.OpenAIError: If there is an error during the API call.
        """

        # Function to encode the image
        def encode_image(image_path: str) -> str:
            path = Path(image_path)
            with path.open("rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")

        # Getting the base64 string
        base64_image = encode_image(image)
        base64_image_pre = [encode_image(img) for img in pre_img]
        prompt = """
         You will receive an image that is going to be an example of the typical image without any anomaly,
         and the last image that you need to decide if it has an anomaly or not.
         Answer with a 'NO' if it does not have any anomalies and 'YES: description'
         where description is a description of the anomaly provided, position.
        """
        messages = [
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image_pre[0]}"},
                        "detail": "high",
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                        "detail": "high",
                    },
                ],
            },
        ]

        try:
            # Make the API call using the openai library
            response = self.openai.chat.completions.create(
                model="gpt-4o-mini-2024-07-18",
                messages=messages,
                max_tokens=300,
            )
            return response.choices[-1].message.content
        except openai.APIConnectionError as e:
            print("The server could not be reached")
            print(e.__cause__)  # an underlying Exception, likely raised within httpx.
            raise
        except openai.RateLimitError as e:
            print("A 429 status code was received; we should back off a bit.")
            print(e.__cause__)
            raise
        except openai.APIStatusError as e:
            print("Another non-200-range status code was received")
            print(e.status_code)
            print(e.response)
            raise

    def api_call(self, image: str) -> str:
        """Makes an API call to OpenAI's GPT-4 model to detect anomalies in an image.

        Args:
            key (str): API key for OpenAI.
            prompt (str): The prompt to provide to the GPT-4 model (not used in the current implementation).
            image (str): Path to the image that needs to be checked for anomalies.

        Returns:
            str: The response from the GPT-4 model indicating whether the image has anomalies or not.
                  It returns 'NO' if there are no anomalies and 'YES: description' if there are anomalies,
                  where 'description' provides details of the anomaly and its position.

        Raises:
            openai.error.OpenAIError: If there is an error during the API call.
        """
        prompt = """
        Examine the provided image carefully to determine if there is an obvious anomaly present.
        Anomalies may include mechanical malfunctions, unexpected objects, safety hazards, structural damages,
        or unusual patterns or defects in the objects.

        Instructions:

        1. Thoroughly inspect the image for any irregularities or deviations from normal operating conditions.

        2. Clearly state if an obvious anomaly is detected.
        - If an anomaly is detected, begin with 'YES,' followed by a detailed description of the anomaly.
        - If no anomaly is detected, simply state 'NO' and end the analysis.

        Example Output Structure:

        'YES:
        - Description: Conveyor belt misalignment causing potential blockages.
        This may result in production delays and equipment damage.
        Immediate realignment and inspection are recommended.'

        'NO'

        Considerations:

        - Ensure accuracy in identifying anomalies to prevent overlooking critical issues.
        - Provide clear and concise descriptions for any detected anomalies.
        - Focus on obvious anomalies that could impact final use of the object operation or safety.
        """

        # Function to encode the image
        def encode_image(image_path: str) -> str:
            path = Path(image_path)
            with path.open("rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")

        # Getting the base64 string
        base64_image = encode_image(image)

        messages = [
            {
                "role": "system",
                "content": f"{prompt}",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                        "detail": "high",
                    },
                ],
            },
        ]

        try:
            # Make the API call using the openai library
            response = self.openai.chat.completions.create(
                model="gpt-4o-mini-2024-07-18",
                messages=messages,
                max_tokens=300,
            )
            return response.choices[-1].message.content
        except openai.APIConnectionError as e:
            print("The server could not be reached")
            print(e.__cause__)  # an underlying Exception, likely raised within httpx.
            raise
        except openai.RateLimitError:
            print("A 429 status code was received; we should back off a bit.")
            raise
        except openai.APIStatusError as e:
            print("Another non-200-range status code was received")
            print(e.status_code)
            print(e.response)
            raise

    def training_step(self, batch: dict[str, str | torch.Tensor], *args, **kwargs) -> dict[str, str | torch.Tensor]:
        """Train Step of LLM."""
        del args, kwargs  # These variables are not used.
        # no train on llm
        return batch

    @staticmethod
    def configure_optimizers() -> None:
        """WinCLIP doesn't require optimization, therefore returns no optimizers."""
        return

    def validation_step(
        self,
        batch: dict[str, str | list[str] | torch.Tensor],
        *args,
        **kwargs,
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
                        self.api_call_few_shot(self.pre_images, batch["image_path"][i]),
                    ).strip()
                else:
                    output = str(self.api_call(batch["image_path"][i])).strip()
            except Exception:
                print(f"Error:img_path:{batch['image_path']}")
                logging.exception(
                    f"Error calling openAI API for image {batch['image_path'][i]}",
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
        ref_images: list[str] = []
        for batch in dataloader:
            images = batch["image_path"][: self.k_shot - len(ref_images)]
            ref_images.extend(images)
            if self.k_shot == len(ref_images):
                break
        return ref_images
