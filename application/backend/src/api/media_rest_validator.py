# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

from fastapi import File, Request, UploadFile

from exceptions import InvalidMediaException, PayloadTooLargeException
from pydantic_models import ImageExtension


class MediaRestValidator:
    SUPPORTED_IMAGE_TYPES = [extension.value for extension in ImageExtension]
    MAX_BYTES_SIZE = 4.7 * 1024**3

    @staticmethod
    def validate_image_file(request: Request, file: UploadFile = File(None)) -> UploadFile | None:
        """
        Validates a request to upload an image, if the file is not present, returns None.

        Args:
            request: FastAPI request that was made to upload the file
            file: uploaded image file

        Raises:
            InvalidMediaException: if the file name is empty or the file extension is
                not in the supported file extensions
            PayloadTooLargeException: when the total size of the request exceeds MAX_BYTES_SIZE

        Returns:
            UploadFile if valid, None if file is not present
        """
        if file is None:
            return None

        if file.size is not None:
            file_size = file.size
        else:
            content_length = request.headers.get("content-length")
            if content_length is None:
                raise InvalidMediaException("Could not determine the size of the uploaded file.")
            try:
                file_size = int(content_length)
            except (ValueError, TypeError):
                raise InvalidMediaException("Invalid Content-Length header.")

        if file_size > MediaRestValidator.MAX_BYTES_SIZE:
            raise PayloadTooLargeException(MediaRestValidator.MAX_BYTES_SIZE / (1024**2))

        if not file.filename:
            raise InvalidMediaException("Filename can't be empty.")

        extension = list(os.path.splitext(file.filename)).pop().lower()
        if extension not in MediaRestValidator.SUPPORTED_IMAGE_TYPES:
            raise InvalidMediaException(
                f"Not a valid image format. Received {extension}, but only the following are allowed: "
                f"{str(MediaRestValidator.SUPPORTED_IMAGE_TYPES)}",
            )
        return file
