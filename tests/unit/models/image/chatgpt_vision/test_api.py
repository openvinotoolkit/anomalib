"""Unit tests for OpenAI Api funtions."""

import os
import unittest
from unittest.mock import MagicMock, Mock, patch

from anomalib.models.image.chatgpt_vision.chatgpt import ChatGPTWrapper


class TestChatGPTWrapper(unittest.TestCase):
    """Unit tests for api_call."""

    @patch.dict(os.environ, {"OPENAI_API_KEY": "fake-api-key"})
    @patch("anomalib.models.image.chatgpt_vision.chatgpt.openai.chat.completions.create")
    def test_api_call(self, mock_openai_create: Mock) -> None:
        """Tests for api_call positive response and few shot."""
        # Set up the mock response from OpenAI
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="YES: Anomaly detected."))]
        mock_openai_create.return_value = mock_response

        # Initialize the ChatGPTWrapper instance
        wrapper = ChatGPTWrapper(model_name="gpt-4o-mini-2024-07-18", detail=True)

        # Prepare test images (simulated base64 encoded strings)
        test_images = ["base64encodedimage1", "base64encodedimage2"]

        # Call the api_call method
        response = wrapper.api_call(images=test_images)

        # Check if the response matches the expected output
        assert response, "YES: Anomaly detected."

        # Check if the openai API was called with the expected parameters
        mock_openai_create.assert_called_once_with(
            model="gpt-4o-mini-2024-07-18",
            messages=unittest.mock.ANY,  # Ignore specific messages content in this check
            max_tokens=300,
        )

    @patch.dict(os.environ, {"OPENAI_API_KEY": "fake-api-key"})
    @patch("anomalib.models.image.chatgpt_vision.chatgpt.openai.chat.completions.create")
    def test_api_call_no_anomaly(self, mock_openai_create: Mock) -> None:
        """Testsfor api_call negative response and zero shot."""
        # Set up the mock response from OpenAI
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="NO"))]
        mock_openai_create.return_value = mock_response

        # Initialize the ChatGPTWrapper instance
        wrapper = ChatGPTWrapper(model_name="gpt-4o-mini-2024-07-18", detail=False)

        # Prepare test images (simulated base64 encoded strings)
        test_images = ["base64encodedimage1"]

        # Call the api_call method
        response = wrapper.api_call(images=test_images)

        # Check if the response matches the expected output
        assert response, "NO"

        # Check if the openai API was called correctly
        mock_openai_create.assert_called_once()
