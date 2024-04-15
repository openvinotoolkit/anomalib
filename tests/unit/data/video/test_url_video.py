"""asdf."""

import pytest
import requests

from anomalib.data.video import avenue, shanghaitech, ucsd_ped


def test_ucsdped_url_is_active() -> None:
    """Test if the URL of the ucsd_ped datamodule is active."""
    url = ucsd_ped.DOWNLOAD_INFO.url
    try:
        response = requests.head(url, timeout=5)  # timeout in seconds
        assert response.status_code == 200
    except requests.exceptions.RequestException as e:
        pytest.fail(f"Request failed: {e}")


def test_shangaitech_url_is_active() -> None:
    """Test if the URL of the ShanghaiTech datamodule is active."""
    url = shanghaitech.DATASET_DOWNLOAD_INFO.url
    try:
        response = requests.head(url, timeout=5)  # timeout in seconds
        assert response.status_code == 200
    except requests.exceptions.RequestException as e:
        pytest.fail(f"Request failed: {e}")


def test_avenue_url_is_active() -> None:
    """Test if the URL of the Avenue datamodule is active."""
    url = avenue.DATASET_DOWNLOAD_INFO.url
    try:
        response = requests.head(url, timeout=5)  # timeout in seconds
        assert response.status_code == 200
    except requests.exceptions.RequestException as e:
        pytest.fail(f"Request failed: {e}")

    url = avenue.ANNOTATIONS_DOWNLOAD_INFO.url
    try:
        response = requests.head(url, timeout=5)  # timeout in seconds
        assert response.status_code == 200
    except requests.exceptions.RequestException as e:
        pytest.fail(f"Request failed: {e}")
