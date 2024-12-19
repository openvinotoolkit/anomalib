"""Custom exceptions for anomalib data validation.

This module provides custom exception classes for handling data validation errors
in anomalib.
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


class MisMatchError(Exception):
    """Exception raised when a data mismatch is detected.

    This exception is raised when there is a mismatch between expected and actual
    data formats or values during validation.

    Args:
        message (str): Custom error message. Defaults to "Mismatch detected."

    Attributes:
        message (str): Explanation of the error.

    Examples:
        >>> raise MisMatchError()  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
            ...
        MisMatchError: Mismatch detected.
        >>> raise MisMatchError("Image dimensions do not match")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
            ...
        MisMatchError: Image dimensions do not match
    """

    def __init__(self, message: str = "") -> None:
        if message:
            self.message = message
        else:
            self.message = "Mismatch detected."
        super().__init__(self.message)
