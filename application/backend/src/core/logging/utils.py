# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import sys
import threading
from collections.abc import Generator
from contextlib import ContextDecorator, contextmanager
from types import TracebackType
from typing import IO, Any, Self
from uuid import UUID

from loguru import logger

from core.logging.handlers import InterceptHandler, LoggerStdoutWriter
from core.logging.setup import global_log_config

# Also logs the weights if they are being downloaded.
_ML_LOGGER_NAMES = (
    "timm",
    "huggingface_hub",
)


class _ThreadLocalStream:
    """Stream wrapper that delegates writes to a per-thread target.

    When no thread-local override is active, writes fall through to the
    original stream that was wrapped at installation time.  Each thread
    can independently push/pop its own override via :meth:`push` /
    :meth:`pop`, so concurrent ``CaptureOutput`` contexts in different
    threads never interfere with each other.  The stack also supports
    nesting within a single thread.
    """

    def __init__(self, original: IO[str]) -> None:
        self._original = original
        self._local: threading.local = threading.local()

    # -- per-thread stream stack ----------------------------------------

    def push(self, stream: IO[str]) -> None:
        """Set *stream* as the active target for the calling thread."""
        if not hasattr(self._local, "stack"):
            self._local.stack = []  # list[IO[str]] per thread
        self._local.stack.append(stream)

    def pop(self) -> None:
        """Remove the most recent override for the calling thread."""
        stack: list[IO[str]] | None = getattr(self._local, "stack", None)
        if stack:
            stack.pop()

    @property
    def _target(self) -> IO[str]:
        """Return the stream the calling thread should write to."""
        stack: list[IO[str]] | None = getattr(self._local, "stack", None)
        if stack:
            return stack[-1]
        return self._original

    # -- stream interface -----------------------------------------------

    def write(self, msg: str) -> int:
        return self._target.write(msg)

    def flush(self) -> None:
        self._target.flush()

    def fileno(self) -> int:
        # Always return the real FD so subprocess / C-extension code works.
        return self._original.fileno()

    def isatty(self) -> bool:
        return self._original.isatty()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target, name)


class CaptureOutput(ContextDecorator):
    """Redirect stdout, stderr, and ML library logging into loguru.

    Usable as a decorator or context manager. While active, all print()
    output, tqdm progress bars, and standard-logging calls from common ML
    libraries are forwarded to loguru so they appear in per-job log files.

    Thread-safety:
        Redirection is **thread-local** — each thread that enters this
        context gets its own capture without affecting other threads.
        Threads that have not entered ``CaptureOutput`` continue to write
        to the original ``sys.stdout`` / ``sys.stderr`` as usual.  Nesting
        within a single thread is also supported (stack-based).

    Note:
        ML-library logger interception (``logging.getLogger`` handlers)
        is still process-global.  This is acceptable because loguru sinks
        are already thread-safe, but be aware that handler lists are shared
        across threads.

    Example (decorator)::

        @CaptureOutput()
        def train(model): ...

    Example (context manager)::

        with CaptureOutput():
            engine.fit(model, datamodule)
    """

    _lock: threading.Lock = threading.Lock()
    _stdout_wrapper: _ThreadLocalStream | None = None
    _stderr_wrapper: _ThreadLocalStream | None = None

    @classmethod
    def _ensure_wrappers_installed(cls) -> None:
        """Install thread-local stream wrappers on sys.stdout/stderr (idempotent)."""
        if cls._stdout_wrapper is not None:
            return
        with cls._lock:
            # Double-checked locking.
            if cls._stdout_wrapper is None:
                cls._stdout_wrapper = _ThreadLocalStream(sys.stdout)
                sys.stdout = cls._stdout_wrapper  # type: ignore[assignment]
            if cls._stderr_wrapper is None:
                cls._stderr_wrapper = _ThreadLocalStream(sys.stderr)
                sys.stderr = cls._stderr_wrapper  # type: ignore[assignment]

    def __enter__(self) -> Self:
        self._ensure_wrappers_installed()

        self._log_writer = LoggerStdoutWriter()
        assert self._stdout_wrapper is not None  # guaranteed by _ensure_wrappers_installed  # noqa: S101
        assert self._stderr_wrapper is not None  # noqa: S101
        self._stdout_wrapper.push(self._log_writer)  # type: ignore[arg-type]
        self._stderr_wrapper.push(self._log_writer)  # type: ignore[arg-type]

        # Intercept standard logging from ML libraries.
        self._original_handlers: dict[str, list[logging.Handler]] = {}
        self._original_levels: dict[str, int] = {}
        intercept = InterceptHandler()
        for name in _ML_LOGGER_NAMES:
            lib_logger = logging.getLogger(name)
            self._original_handlers[name] = lib_logger.handlers[:]
            self._original_levels[name] = lib_logger.level
            lib_logger.handlers = [intercept]
            lib_logger.setLevel(logging.DEBUG)
            lib_logger.propagate = False

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        # Restore ML library loggers.
        for name in _ML_LOGGER_NAMES:
            lib_logger = logging.getLogger(name)
            lib_logger.handlers = self._original_handlers[name]
            lib_logger.level = self._original_levels[name]
            # Restore the original propagate flag if it was captured; fall back to True.
            original_propagate = getattr(self, "_original_propagate", {}).get(name, True)
            lib_logger.propagate = original_propagate

        assert self._stderr_wrapper is not None  # noqa: S101
        assert self._stdout_wrapper is not None  # noqa: S101
        self._stderr_wrapper.pop()
        self._stdout_wrapper.pop()


def _validate_job_id(job_id: str | UUID) -> str | UUID:
    """Validate job_id to prevent path traversal attacks.

    Args:
        job_id: The job identifier to validate

    Returns:
        Validated job_id

    Raises:
        ValueError: If job_id is not a valid UUID
    """
    try:
        UUID(str(job_id))
    except ValueError as e:
        raise ValueError(
            f"Invalid job_id '{job_id}'. Only alphanumeric characters, hyphens, and underscores are allowed.",
        ) from e
    return job_id


def get_job_logs_path(job_id: str | UUID) -> str:
    """Get the path to the logs folder for a specific job.

    Args:
        job_id: Unique identifier for the job

    Returns:
        str: Path to the job's logs folder (logs/jobs/{job_id})

    Raises:
        ValueError: If job_id contains invalid characters

    Example:
        >>> get_job_logs_path(job_id="544bb0ab-de53-4612-b785-3e42ae6e83c1")
        'logs/jobs/544bb0ab-de53-4612-b785-3e42ae6e83c1'
    """
    job_id = _validate_job_id(job_id)
    jobs_folder = os.path.join(global_log_config.log_folder, "jobs")
    try:
        os.makedirs(jobs_folder, exist_ok=True)
    except OSError as e:
        raise RuntimeError(f"Failed to create jobs log directory: {e}") from e
    return os.path.join(jobs_folder, f"{job_id}.log")


@contextmanager
def job_logging_ctx(job_id: str | UUID) -> Generator[str]:
    """Add a temporary log sink for a specific job.

    Captures all logs emitted during the context to logs/jobs/{job_id}.log.
    The sink is automatically removed on exit, but the log file persists.
    Logs also continue to go to other configured sinks.

    Args:
        job_id: Unique identifier for the job, used as the log filename

    Yields:
        str: Path to the created log file (logs/jobs/{job_id}.log)

    Raises:
        ValueError: If job_id contains invalid characters
        RuntimeError: If log directory creation or sink addition fails

    Example:
        >>> with job_logging_ctx(job_id="foo-123"):
        ...     logger.info("bar")  # All logs saved to logs/jobs/train-123.log
    """
    job_id = _validate_job_id(job_id)

    log_file = get_job_logs_path(job_id)

    try:
        sink_id = logger.add(
            log_file,
            rotation=global_log_config.rotation,
            retention=global_log_config.retention,
            level=global_log_config.level,
            serialize=global_log_config.serialize,
            enqueue=True,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to add log sink for job {job_id}: {e}") from e

    try:
        logger.info(f"Started logging to {log_file}")
        yield log_file
    finally:
        logger.info(f"Stopped logging to {log_file}")
        logger.remove(sink_id)
