"""Custom rich methods."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Generator, Iterable
from typing import TYPE_CHECKING, Any

from rich import get_console
from rich.progress import track

if TYPE_CHECKING:
    from rich.live import Live


class CacheRichLiveState:
    """Cache the live state of the console.

    Note: This is a bit dangerous as it accesses private attributes of the console.
    Use this with caution.
    """

    def __init__(self) -> None:
        self.console = get_console()
        self.live: Live | None = None

    def __enter__(self) -> None:
        """Save the live state of the console."""
        # Need to access private attribute to get the live state
        with self.console._lock:  # noqa: SLF001
            self.live = self.console._live  # noqa: SLF001
            self.console.clear_live()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:  # noqa: ANN401
        """Restore the live state of the console."""
        if self.live:
            self.console.clear_live()
            self.console.set_live(self.live)


def safe_track(*args, **kwargs) -> Generator[Iterable, Any, Any]:
    """Wraps ``rich.progress.track`` with a context manager to cache the live state.

    For parameters look at ``rich.progress.track``.
    """
    with CacheRichLiveState():
        yield from track(*args, **kwargs)
