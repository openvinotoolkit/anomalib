"""Custom Exception Class for Mismatch Detection (MisMatchError)."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from anomalib.utils import create_class_alias_with_deprecation_warning


class MismatchError(Exception):
    """Exception raised when a mismatch is detected.

    Attributes:
        message (str): Explanation of the error.
    """

    def __init__(self, message: str = "") -> None:
        if message:
            self.message = message
        else:
            self.message = "Mismatch detected."
        super().__init__(self.message)


# NOTE: This is deprecated and will be removed in a future release.
MisMatchError = create_class_alias_with_deprecation_warning(MismatchError, "MisMatchError")
