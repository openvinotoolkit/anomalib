"""OpenAI Visual Large Model: Zero-/Few-Shot Anomaly Classification.

Paper (No paper)
"""
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import base64
import logging
from pathlib import Path

import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch.utils.data import DataLoader

from anomalib import LearningType
from anomalib.metrics.threshold import ManualThreshold
from anomalib.models.components import AnomalyModule

from .chatgpt import ChatGPTWrapper

logger = logging.getLogger(__name__)

__all__ = ["ChatGPTVision"]


class ChatGPTVision(AnomalyModule):
    """OpenAI VLM Lightning model using OpenAI's GPT-4 for image anomaly detection.

    Args:
        k_shot(int): The number of images that will compare to detect if it is an anomaly.
        model_name (str): The openAI vlm for the image detection.
        detail (bool): The detail of the input in the vlm for the image detection 'high'(true) 'low'(false).
    """

    def __init__(
        self,
        k_shot: int = 0,
        model_name: str = "gpt-4o-mini-2024-07-18",
        detail: bool = True,
    ) -> None:
        super().__init__()

        self.k_shot = k_shot

        self.model_name = model_name
        self.detail = detail
        self.image_threshold = ManualThreshold()
        self.vlm = ChatGPTWrapper(model_name=self.model_name, detail=self.detail)

    def _setup(self) -> None:
        dataloader = self.trainer.datamodule.train_dataloader()
        pre_images = self.collect_reference_images(dataloader)
        self.pre_images = pre_images

    # Function to encode the image
    def _encode_image(self, image_path: str) -> str:
        path = Path(image_path)
        with path.open("rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def training_step(self, batch: dict[str, str | torch.Tensor], *args, **kwargs) -> dict[str, str | torch.Tensor]:
        """Train Step of LLM."""
        del args, kwargs  # These variables are not used.
        # no train on llm
        return batch

    @staticmethod
    def configure_optimizers() -> None:
        """OpenaiVlm doesn't require optimization, therefore returns no optimizers."""
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
        batch_size = len(batch["image_path"])
        out_list: list[str] = []
        pred_list: list[float] = []
        for i in range(batch_size):
            # Getting the base64 string
            base64_images = [self._encode_image(img) for img in self.pre_images]
            base64_images.append(self._encode_image(batch["image_path"][i]))

            try:
                output = self.vlm.api_call(base64_images)
            except Exception:
                logging.exception(
                    f"Error calling openAI API for image {batch['image_path'][i]}",
                )
                output = "Error"

            # set an error and get to normal if not followed
            prediction = 0.0
            if output.startswith("N"):
                prediction = 0.0
            elif output.startswith("Y"):
                prediction = 1.0
            else:
                logging.warning(
                    f"(Set predition to '0' Normal)Could not identify if there is anomaly by the output:\n{output}",
                )

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

    def collect_reference_images(self, dataloader: DataLoader) -> list[str]:
        """Collect reference images for few-shot inference.

        The reference images are collected by iterating the training dataset until the required number of images are
        collected.

        Returns:
            ref_images list[str]: A list containing the reference images path.
        """
        reference_images_paths: list[str] = []
        for batch in dataloader:
            image_paths = batch["image_path"][: self.k_shot - len(reference_images_paths)]
            reference_images_paths.extend(image_paths)
            if self.k_shot == len(reference_images_paths):
                break
        return reference_images_paths
