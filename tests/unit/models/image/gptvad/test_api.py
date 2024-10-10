"""Unit tests for GptVad OpenAI Api funtions."""
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

import pytest
from pytest_mock import MockerFixture

from anomalib.models.image.gptvad.chatgpt import GPTWrapper


class TestGPTWrapper:
    """Unit tests for api_call."""

    @pytest.fixture(autouse=True)
    def _mock_env(self, mocker: MockerFixture) -> None:
        """Fixture to automatically patch environment variables."""
        mocker.patch.dict(os.environ, {"OPENAI_API_KEY": "fake-api-key"})

    def test_api_call(self, mocker: MockerFixture) -> None:
        """Tests for api_call positive response and few shot."""
        # Set up the mock response from OpenAI
        mock_response = mocker.MagicMock()
        mock_response.choices = [mocker.MagicMock(message=mocker.MagicMock(content="YES: Anomaly detected."))]

        # Mock the openai.chat.completions.create function
        mock_openai_create = mocker.patch("anomalib.models.image.gptvad.chatgpt.openai.chat.completions.create")
        mock_openai_create.return_value = mock_response

        # Initialize the GPTWrapper instance
        wrapper = GPTWrapper(model_name="gpt-4o-mini-2024-07-18", detail=True)

        # Prepare test images (simulated base64 encoded strings)
        test_images = ["base64encodedimage1", "base64encodedimage2"]

        # Call the api_call method
        response = wrapper.api_call(images=test_images)

        # Check if the response matches the expected output
        assert response == "YES: Anomaly detected."

        # Check if the openai API was called with the expected parameters
        mock_openai_create.assert_called_once_with(
            model="gpt-4o-mini-2024-07-18",
            messages=mocker.ANY,  # Ignore specific messages content in this check
            max_tokens=300,
        )

    def test_api_call_no_anomaly(self, mocker: MockerFixture) -> None:
        """Tests for api_call negative response and zero shot."""
        # Set up the mock response from OpenAI
        mock_response = mocker.MagicMock()
        mock_response.choices = [mocker.MagicMock(message=mocker.MagicMock(content="NO"))]
        # Mock the openai.chat.completions.create function
        mock_openai_create = mocker.patch("anomalib.models.image.gptvad.chatgpt.openai.chat.completions.create")
        mock_openai_create.return_value = mock_response

        # Initialize the GPTWrapper instance
        wrapper = GPTWrapper(model_name="gpt-4o-mini-2024-07-18", detail=False)

        # Prepare test images (simulated base64 encoded strings)
        test_images = ["base64encodedimage1"]

        # Call the api_call method
        response = wrapper.api_call(images=test_images)

        # Check if the response matches the expected output
        assert response == "NO"

        # Check if the openai API was called correctly
        mock_openai_create.assert_called_once()
