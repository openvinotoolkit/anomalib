"""Logging Utility functions."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import functools
import io
import logging
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any


class LoggerRedirectError(Exception):
    """Exception occurred when executing function with outputs redirected to logger."""


def hide_output(func: Callable[..., Any]) -> Callable[..., Any]:
    """Hide output of the function.

    Args:
        func (function): Hides output from all streams of this function.

    Raises:
        Exception: In case the execution of function fails, it raises an exception.

    Returns:
        object of the called function
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        """Wrapper function."""
        # redirect stdout and stderr to logger
        stdout = sys.stdout
        stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        try:
            value = func(*args, **kwargs)
        except Exception as exception:  # noqa: BLE001
            msg = f"Error occurred while executing {func.__name__}"
            raise LoggerRedirectError(msg) from exception
        finally:
            sys.stdout = stdout
            sys.stderr = stderr
        return value

    return wrapper


def redirect_logs(log_file: str) -> None:
    """Add file handler to logger."""
    Path(log_file).parent.mkdir(exist_ok=True, parents=True)
    logger_file_handler = logging.FileHandler(log_file)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(format=format_string, level=logging.DEBUG, handlers=[logger_file_handler])
    logging.captureWarnings(capture=True)
    # remove stream handler from all loggers
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for _logger in loggers:
        _logger.handlers = [logger_file_handler]
