# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from io import BytesIO

import pytest
from fastapi import Request, UploadFile

from api.media_rest_validator import MediaRestValidator
from exceptions import InvalidMediaException, PayloadTooLargeException


@pytest.fixture
def fxt_empty_request() -> Request:
    return Request(scope={"type": "http", "headers": []})


def test_validate_image_file_none(fxt_empty_request: Request) -> None:
    """Test validation returns None when no file is uploaded."""
    assert MediaRestValidator.validate_image_file(fxt_empty_request, file=None) is None


def test_validate_image_file_empty_file_with_size_zero(fxt_empty_request: Request) -> None:
    """Test validation succeeds for an empty file (size == 0) without Content-Length header.

    Regression test for Issue #3785: file.size == 0 is falsy, which previously triggered
    an unhandled KeyError on request.headers['content-length'] when the header was absent.
    """
    file = UploadFile(file=BytesIO(b""), filename="empty.png", size=0)
    result = MediaRestValidator.validate_image_file(fxt_empty_request, file=file)
    assert result is file


def test_validate_image_file_valid_with_size(fxt_empty_request: Request) -> None:
    """Test validation succeeds for a normal file with size specified."""
    file = UploadFile(file=BytesIO(b"testcontent"), filename="sample.jpg", size=11)
    result = MediaRestValidator.validate_image_file(fxt_empty_request, file=file)
    assert result is file


def test_validate_image_file_fallback_to_content_length() -> None:
    """Test validation falls back to Content-Length header when file.size is None."""
    request = Request(scope={"type": "http", "headers": [(b"content-length", b"1024")]})
    file = UploadFile(file=BytesIO(b"a" * 1024), filename="image.png", size=None)
    result = MediaRestValidator.validate_image_file(request, file=file)
    assert result is file


def test_validate_image_file_missing_size_and_content_length(fxt_empty_request: Request) -> None:
    """Test validation raises InvalidMediaException when both file.size and Content-Length are missing."""
    file = UploadFile(file=BytesIO(b"a"), filename="image.png", size=None)
    with pytest.raises(InvalidMediaException) as exc_info:
        MediaRestValidator.validate_image_file(fxt_empty_request, file=file)
    assert "Could not determine the size of the uploaded file." in str(exc_info.value)


def test_validate_image_file_invalid_content_length_header() -> None:
    """Test validation raises InvalidMediaException when Content-Length header is not a valid integer."""
    request = Request(scope={"type": "http", "headers": [(b"content-length", b"not-a-number")]})
    file = UploadFile(file=BytesIO(b"a"), filename="image.png", size=None)
    with pytest.raises(InvalidMediaException) as exc_info:
        MediaRestValidator.validate_image_file(request, file=file)
    assert "Invalid Content-Length header." in str(exc_info.value)


def test_validate_image_file_payload_too_large(fxt_empty_request: Request) -> None:
    """Test validation raises PayloadTooLargeException when file size exceeds MAX_BYTES_SIZE."""
    excessive_size = int(MediaRestValidator.MAX_BYTES_SIZE + 1)
    file = UploadFile(file=BytesIO(b""), filename="huge.png", size=excessive_size)
    with pytest.raises(PayloadTooLargeException):
        MediaRestValidator.validate_image_file(fxt_empty_request, file=file)


@pytest.mark.parametrize("empty_filename", [None, ""])
def test_validate_image_file_empty_filename(fxt_empty_request: Request, empty_filename: str | None) -> None:
    """Test validation raises InvalidMediaException when filename is empty or None."""
    file = UploadFile(file=BytesIO(b"data"), filename=empty_filename, size=4)
    with pytest.raises(InvalidMediaException) as exc_info:
        MediaRestValidator.validate_image_file(fxt_empty_request, file=file)
    assert "Filename can't be empty." in str(exc_info.value)


def test_validate_image_file_unsupported_format(fxt_empty_request: Request) -> None:
    """Test validation raises InvalidMediaException for unsupported image extensions."""
    file = UploadFile(file=BytesIO(b"data"), filename="document.pdf", size=4)
    with pytest.raises(InvalidMediaException) as exc_info:
        MediaRestValidator.validate_image_file(fxt_empty_request, file=file)
    assert "Not a valid image format." in str(exc_info.value)
