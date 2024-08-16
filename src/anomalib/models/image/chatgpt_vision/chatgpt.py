"""Wrapper for the OpenAI calls to the VLM model."""

import logging
import os
from typing import Any

import openai


class ChatGPTWrapper:
    """A wrapper class for making API calls to OpenAI's GPT-4 model to detect anomalies in images.

    Environment variable OPENAI_API_KEY (str): API key for OpenAI.
    https://platform.openai.com/docs/quickstart/step-2-set-up-your-api-key

    Args:
        model_name (str): Model name for OpenAI API VLM. Default "gpt-4o"
        detail (bool): If the images will be sended with high detail or low detail.

    """

    def __init__(self, model_name: str = "gpt-4o", detail: bool = True) -> None:
        openai_key = os.getenv("OPENAI_API_KEY")
        self.model_name = model_name
        self.detail = detail
        if not openai_key:
            from anomalib.engine.engine import UnassignedError

            msg = "OpenAI environment key not found.(OPENAI_API_KEY)"
            raise UnassignedError(msg)

    def api_call(
        self,
        images: list[str],
        extension: str = "png",
    ) -> str:
        """Makes an API call to OpenAI's GPT-4 model to detect anomalies in an image.

        Args:
            images (list[str]): List of base64 images that serve as examples and last one to check for anomalies.
            extension (str): Extension of the group of images that needs to be checked for anomalies. Default = 'png'

        Returns:
            str: The response from the GPT-4 model indicating whether the image has anomalies or not.
                  It returns 'NO' if there are no anomalies and 'YES: description' if there are anomalies,
                  where 'description' provides details of the anomaly and its position.

        Raises:
            openai.error.OpenAIError: If there is an error during the API call.
        """
        prompt: str = ""
        if len(images) > 0:
            prompt = """
             You will receive an image that is going to be an example of the typical image without any anomaly,
             and the last image that you need to decide if it has an anomaly or not.
             Answer with a 'NO' if it does not have any anomalies and 'YES: description'
             where description is a description of the anomaly provided, position.
            """
        else:
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

        detail_img = "high" if self.detail else "low"
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": prompt,
            },
        ]
        for image in images:
            image_message = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{extension};base64,{image}",
                                "detail": detail_img,
                            },
                        },
                    ],
                },
            ]
            messages.extend(image_message)

        try:
            # Make the API call using the openai library
            response = openai.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=300,
            )
            return response.choices[-1].message.content or ""
        except Exception:
            msg = "The openai API trow an exception."
            logging.exception(msg)
            raise
