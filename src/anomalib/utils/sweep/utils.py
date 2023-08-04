"""Additional utils for sweep."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import functools
import io
import logging
import sys
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


def redirect_output(func: Callable) -> Callable[..., dict[str, Any]]:
    """Decorator to redirect output of the function.

    Args:
        func (function): Hides output of this function.

    Raises:
        Exception: Incase the execution of function fails, it raises an exception.

    Returns:
        object of the called function
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> dict[str, Any]:
        std_out = sys.stdout
        sys.stdout = buf = io.StringIO()
        try:
            value = func(*args, **kwargs)
            logger.info(buf.getvalue())
            logger.info(value)
        except Exception as exp:
            logger.exception(
                "Error occurred while computing benchmark %s. Buffer: %s." "\n Method %s, args %s, kwargs %s",
                exp,
                buf.getvalue(),
                func,
                args,
                kwargs,
            )
            value = {}
        sys.stdout = std_out
        return value

    return wrapper


class Status(str, Enum):
    """Status of the benchmarking run."""

    SUCCESS = "success"
    FAILED = "failed"


class Result:
    def __init__(self, value: Any, status=Status.SUCCESS):
        self.value = value
        self.status = status

    def __bool__(self):
        return self.status == Status.SUCCESS


def exception_wrapper(func: Callable) -> Callable[..., Result]:
    """Wrapper method to handle exceptions.

    Args:
        func (function): Function to be wrapped.

    Raises:
        Exception: Incase the execution of function fails, it raises an exception.

    Example:
        >>> @exception_wrapper
        ... def func():
        ...     raise Exception("Exception occurred")
        >>> func()
        Exception: Exception occurred

    Returns:
        object of the called function
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Result:
        try:
            value = Result(value=func(*args, **kwargs))
        except Exception as exp:
            logger.exception(
                "Error occurred while computing benchmark %s. Method %s, args %s, kwargs %s",
                exp,
                func,
                args,
                kwargs,
            )
            value = Result(False, Status.FAILED)
        return value

    return wrapper
