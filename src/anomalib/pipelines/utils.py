# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Utility functions."""

import functools
import io
import sys
from collections.abc import Callable
from typing import Any


def hide_output(func: Callable[..., Any]) -> Callable[..., Any]:
    """Hide output of the function.

    Args:
        func (function): Hides output of this function.

    Raises:
        Exception: In case the execution of function fails, it raises an exception.

    Returns:
        object of the called function
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:  # noqa: ANN401
        std_out = sys.stdout
        sys.stdout = buf = io.StringIO()
        try:
            value = func(*args, **kwargs)
        # NOTE: A generic exception is used here to catch all exceptions.
        except Exception as exception:  # noqa: BLE001
            raise Exception(buf.getvalue()) from exception  # noqa: TRY002
        sys.stdout = std_out
        return value

    return wrapper
