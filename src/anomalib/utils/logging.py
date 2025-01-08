"""Logging utility functions for anomaly detection.

This module provides utilities for logging and output management. The key components include:

    - ``LoggerRedirectError``: Custom exception for logging redirection failures
    - ``hide_output``: Decorator to suppress function output streams
    - Helper functions for redirecting output to loggers

Example:
    >>> from anomalib.utils.logging import hide_output
    >>> @hide_output
    >>> def my_function():
    ...     print("This output will be hidden")
    >>> my_function()

The module ensures consistent logging behavior by:
    - Providing decorators for output control
    - Handling both stdout and stderr redirection
    - Supporting exception propagation
    - Offering flexible output management

Note:
    The logging utilities are designed to work with both standard Python logging
    and custom logging implementations.
"""

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
    """Exception raised when redirecting function output to logger fails.

    This exception is raised when there is an error while redirecting the output
    streams (stdout/stderr) of a function to a logger. It typically occurs in
    functions decorated with ``@hide_output``.

    Example:
        >>> @hide_output
        >>> def problematic_function():
        ...     raise ValueError("Something went wrong")
        >>> problematic_function()
        Traceback (most recent call last):
            ...
        LoggerRedirectError: Error occurred while executing problematic_function

    Note:
        This exception wraps the original exception that caused the redirection
        failure, which can be accessed through the ``__cause__`` attribute.
    """


def hide_output(func: Callable[..., Any]) -> Callable[..., Any]:
    """Hide output of a function by redirecting stdout and stderr.

    This decorator captures and discards any output that would normally be printed
    to stdout or stderr when the decorated function executes. The function's
    return value is preserved.

    Args:
        func (Callable[..., Any]): Function whose output should be hidden.
            All output streams from this function will be captured.

    Returns:
        Callable[..., Any]: Wrapped function that executes silently.

    Raises:
        LoggerRedirectError: If an error occurs during function execution. The
            original exception can be accessed via ``__cause__``.

    Example:
        Basic usage to hide print statements:

        >>> @hide_output
        ... def my_function():
        ...     print("This will not be printed")
        >>> my_function()  # No output will appear

        Exceptions are still propagated:

        >>> @hide_output
        ... def my_function():
        ...     1/0  # doctest: +IGNORE_EXCEPTION_DETAIL
        >>> my_function()
        Traceback (most recent call last):
            ...
        LoggerRedirectError: Error occurred while executing my_function

    Note:
        - The decorator preserves the function's metadata using ``functools.wraps``
        - Both ``stdout`` and ``stderr`` streams are captured
        - Original streams are always restored, even if an exception occurs
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
        except Exception as exception:
            msg = f"Error occurred while executing {func.__name__}"
            raise LoggerRedirectError(msg) from exception
        finally:
            sys.stdout = stdout
            sys.stderr = stderr
        return value

    return wrapper


def redirect_logs(log_file: str) -> None:
    """Add file handler to logger and remove other handlers.

    This function sets up file-based logging by:
        - Creating a file handler for the specified log file
        - Setting a standard format for log messages
        - Removing all other handlers from existing loggers
        - Configuring warning capture

    Args:
        log_file: Path to the log file where messages will be written.
            Parent directories will be created if they don't exist.

    Example:
        >>> from pathlib import Path
        >>> log_path = Path("logs/app.log")
        >>> redirect_logs(str(log_path))  # doctest: +SKIP
        >>> import logging
        >>> logger = logging.getLogger(__name__)
        >>> logger.info("Test message")  # Message written to logs/app.log

    Note:
        - The log format includes timestamp, logger name, level and message
        - All existing handlers are removed from loggers to ensure logs only go
          to file
        - This function does not work well with multiprocessing - logs from
          child processes will not be redirected
        - The function captures Python warnings in addition to regular logs
    """
    Path(log_file).parent.mkdir(exist_ok=True, parents=True)
    logger_file_handler = logging.FileHandler(log_file)
    format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(format=format_string, handlers=[logger_file_handler])
    logging.captureWarnings(capture=True)
    # remove other handlers from all loggers
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for _logger in loggers:
        _logger.handlers = [logger_file_handler]
