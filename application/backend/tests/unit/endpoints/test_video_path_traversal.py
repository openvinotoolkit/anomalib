# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for path traversal at the HTTP endpoint layer.

Verifies that validate_video_file() returns HTTP 400 for filenames that contain
path separators or parent-directory references before the request reaches
VideoService.upload_video().
"""

import io
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from main import app

client = TestClient(app, raise_server_exceptions=False)

PROJECT_ID = str(uuid4())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _upload(filename: str, content: bytes = b"fake") -> int:
    """POST to the video upload endpoint with the given filename; return status code."""
    response = client.post(
        f"/api/projects/{PROJECT_ID}/videos",
        files={"file": (filename, io.BytesIO(content), "video/mp4")},
    )
    return response.status_code


# ---------------------------------------------------------------------------
# Traversal filenames — all must be rejected with HTTP 400
# ---------------------------------------------------------------------------


TRAVERSAL_FILENAMES = [
    # Classic Unix traversal
    "../../../../tmp/pwned.mp4",
    # PoC payload from the vulnerability report
    "../../../../../../../../tmp/anomalib_pwned.mp4",
    # Single step up
    "../sibling_project/video.mp4",
    # Traversal inside a subdirectory segment
    "subdir/../../outside.mp4",
    # Absolute path
    "/tmp/absolute.mp4",
    # Windows-style separators
    "..\\..\\outside.mp4",
    # Forward slash only (no traversal but still a path component)
    "sub/video.mp4",
    # Null byte injection attempt
    "video%00.mp4",
]


@pytest.mark.parametrize("traversal_filename", TRAVERSAL_FILENAMES)
def test_upload_rejects_traversal_filename(traversal_filename):
    """validate_video_file must return 400 for any filename with path components.

    VideoService.upload_video must never be reached.
    """
    with patch("api.endpoints.video_endpoints.VideoService.upload_video", new_callable=AsyncMock) as mock_upload:
        code = _upload(traversal_filename)

    assert code == status.HTTP_400_BAD_REQUEST, (
        f"Expected 400 for traversal filename {traversal_filename!r}, got {code}"
    )
    mock_upload.assert_not_called()


# ---------------------------------------------------------------------------
# Safe filenames — must reach the service layer (service is mocked away)
# ---------------------------------------------------------------------------


SAFE_FILENAMES = [
    "video.mp4",
    "my_recording.avi",
    "clip-2026-07-20.mov",
    "test.mkv",
    "capture.webm",
]


@pytest.mark.parametrize("safe_filename", SAFE_FILENAMES)
def test_upload_accepts_safe_filename(safe_filename):
    """A plain filename (no path separators) must pass validate_video_file and reach the service."""
    mock_video = {
        "project_id": PROJECT_ID,
        "filename": safe_filename,
        "video_path": f"/data/videos/projects/{PROJECT_ID}/{safe_filename}",
        "size": 4,
        "created_at": "2026-07-20T00:00:00",
    }

    with patch("api.endpoints.video_endpoints.VideoService.upload_video", new_callable=AsyncMock) as mock_upload:
        mock_upload.return_value = mock_video
        code = _upload(safe_filename)

    # A safe filename must pass validation and reach the service, resulting in a successful create.
    assert code == status.HTTP_201_CREATED, f"Expected 201 for safe filename {safe_filename!r}, got {code}"
    mock_upload.assert_called_once()


# ---------------------------------------------------------------------------
# Extension validation — must still reject non-video extensions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_filename", ["malware.exe", "config.yaml", "shell.sh", "payload"])
def test_upload_rejects_invalid_extension(bad_filename):
    """Files with unsupported extensions must return 400 regardless of path safety."""
    code = _upload(bad_filename)
    assert code == status.HTTP_400_BAD_REQUEST
