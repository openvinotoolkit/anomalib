"""Unit tests for OpenAI Api funtions."""

from unittest.mock import MagicMock, Mock, patch

from anomalib.models.image.openai_vlm.lightning_model import OpenaiVlm


@patch("openai.OpenAI")  # Mock the OpenAI class instantiation
def test_api_call_few_shot(mock_openai: Mock) -> None:
    """Test few shot wrapper."""
    # Mock the return value for chat.completions.create().choices[-1].message.content
    mock_content = MagicMock(name="message.content", return_value="YES: anomaly detected")

    # Set up the mock chain
    mock_openai_instance = mock_openai.return_value
    mock_openai_instance.chat.completions.create.return_value.choices = [MagicMock(message=MagicMock(content=""))]
    mock_openai_instance.chat.completions.create.return_value.choices[-1].message.content = mock_content.return_value

    # Initialize the OpenaiVlm object
    openai_key = "fake-key"  # This won't be used since we're mocking
    vlm_model = OpenaiVlm(k_shot=1, openai_key=openai_key)

    # Create some dummy image paths
    pre_img = ["path/to/pre_image_1.png"]
    image = "path/to/test_image.png"

    # Mock the encode_image function to avoid actual file I/O
    with patch.object(vlm_model, "_encode_image", return_value="dummy_base64_string"):
        response = vlm_model.api_call_few_shot(pre_img=pre_img, image=image)

    # Check that the API was called correctly and the mock returned the expected result
    assert response == "YES: anomaly detected"
    mock_openai_instance.chat.completions.create.assert_called_once_with(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {
                "role": "system",
                "content": """
         You will receive an image that is going to be an example of the typical image without any anomaly,
         and the last image that you need to decide if it has an anomaly or not.
         Answer with a 'NO' if it does not have any anomalies and 'YES: description'
         where description is a description of the anomaly provided, position.
        """,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,dummy_base64_string",
                            "detail": "high",
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
                            "url": "data:image/png;base64,dummy_base64_string",
                            "detail": "high",
                        },
                    },
                ],
            },
        ],
        max_tokens=300,
    )
    mock_openai.assert_called_once_with(api_key=openai_key)


# Example test for the api_call_few_shot function
@patch("openai.OpenAI")  # Mock the OpenAI class instantiation
def test_api_call_zero_shot(mock_openai: Mock) -> None:
    """Test zero shot wrapper."""
    # Mock the return value for chat.completions.create().choices[-1].message.content
    mock_content = MagicMock(name="message.content", return_value="YES: anomaly detected")

    # Set up the mock chain
    mock_openai_instance = mock_openai.return_value
    mock_openai_instance.chat.completions.create.return_value.choices = [MagicMock(message=MagicMock(content=""))]
    mock_openai_instance.chat.completions.create.return_value.choices[-1].message.content = mock_content.return_value

    # Initialize the OpenaiVlm object
    openai_key = "fake-key"  # This won't be used since we're mocking
    vlm_model = OpenaiVlm(k_shot=0, openai_key=openai_key)

    # Create some dummy image paths
    image = "path/to/test_image.png"

    # Mock the encode_image function to avoid actual file I/O
    with patch.object(vlm_model, "_encode_image", return_value="dummy_base64_string"):
        response = vlm_model.api_call(image=image)

    # Check that the API was called correctly and the mock returned the expected result
    assert response == "YES: anomaly detected"
    mock_openai_instance.chat.completions.create.assert_called_once_with(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {
                "role": "system",
                "content": """
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
        """,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/jpeg;base64,dummy_base64_string",
                            "detail": "high",
                        },
                    },
                ],
            },
        ],
        max_tokens=300,
    )
    mock_openai.assert_called_once_with(api_key=openai_key)
