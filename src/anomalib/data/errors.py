"""Custom Exception Class for Mismatch Detection (MisMatchError)."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


class MisMatchError(Exception):
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
