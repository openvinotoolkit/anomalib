"""Helper to show progress bars with `urlretrieve`, check hash of file."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import hashlib
import io
from pathlib import Path
from typing import Dict, Iterable, Optional, Union

from tqdm import tqdm


class DownloadProgressBar(tqdm):
    """Create progress bar for urlretrieve. Subclasses `tqdm`.

    For information about the parameters in constructor, refer to `tqdm`'s documentation.

    Args:
        iterable (Optional[Iterable]): Iterable to decorate with a progressbar.
                            Leave blank to manually manage the updates.
        desc (Optional[str]): Prefix for the progressbar.
        total (Optional[Union[int, float]]): The number of expected iterations. If unspecified,
                                            len(iterable) is used if possible. If float("inf") or as a last
                                            resort, only basic progress statistics are displayed
                                            (no ETA, no progressbar).
                                            If `gui` is True and this parameter needs subsequent updating,
                                            specify an initial arbitrary large positive number,
                                            e.g. 9e9.
        leave (Optional[bool]): upon termination of iteration. If `None`, will leave only if `position` is `0`.
        file (Optional[Union[io.TextIOWrapper, io.StringIO]]): Specifies where to output the progress messages
                                                            (default: sys.stderr). Uses `file.write(str)` and
                                                            `file.flush()` methods.  For encoding, see
                                                            `write_bytes`.
        ncols (Optional[int]): The width of the entire output message. If specified,
                            dynamically resizes the progressbar to stay within this bound.
                            If unspecified, attempts to use environment width. The
                            fallback is a meter width of 10 and no limit for the counter and
                            statistics. If 0, will not print any meter (only stats).
        mininterval (Optional[float]): Minimum progress display update interval [default: 0.1] seconds.
        maxinterval (Optional[float]): Maximum progress display update interval [default: 10] seconds.
                                    Automatically adjusts `miniters` to correspond to `mininterval`
                                    after long display update lag. Only works if `dynamic_miniters`
                                    or monitor thread is enabled.
        miniters (Optional[Union[int, float]]): Minimum progress display update interval, in iterations.
                                            If 0 and `dynamic_miniters`, will automatically adjust to equal
                                            `mininterval` (more CPU efficient, good for tight loops).
                                            If > 0, will skip display of specified number of iterations.
                                            Tweak this and `mininterval` to get very efficient loops.
                                            If your progress is erratic with both fast and slow iterations
                                            (network, skipping items, etc) you should set miniters=1.
        use_ascii (Optional[Union[bool, str]]): If unspecified or False, use unicode (smooth blocks) to fill
                                        the meter. The fallback is to use ASCII characters " 123456789#".
        disable (Optional[bool]): Whether to disable the entire progressbar wrapper
                                    [default: False]. If set to None, disable on non-TTY.
        unit (Optional[str]): String that will be used to define the unit of each iteration
                            [default: it].
        unit_scale (Union[bool, int, float]): If 1 or True, the number of iterations will be reduced/scaled
                            automatically and a metric prefix following the
                            International System of Units standard will be added
                            (kilo, mega, etc.) [default: False]. If any other non-zero
                            number, will scale `total` and `n`.
        dynamic_ncols (Optional[bool]): If set, constantly alters `ncols` and `nrows` to the
                                        environment (allowing for window resizes) [default: False].
        smoothing (Optional[float]): Exponential moving average smoothing factor for speed estimates
                                    (ignored in GUI mode). Ranges from 0 (average speed) to 1
                                    (current/instantaneous speed) [default: 0.3].
        bar_format (Optional[str]):  Specify a custom bar string formatting. May impact performance.
                                    [default: '{l_bar}{bar}{r_bar}'], where
                                    l_bar='{desc}: {percentage:3.0f}%|' and
                                    r_bar='| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, '
                                    '{rate_fmt}{postfix}]'
                                    Possible vars: l_bar, bar, r_bar, n, n_fmt, total, total_fmt,
                                    percentage, elapsed, elapsed_s, ncols, nrows, desc, unit,
                                    rate, rate_fmt, rate_noinv, rate_noinv_fmt,
                                    rate_inv, rate_inv_fmt, postfix, unit_divisor,
                                    remaining, remaining_s, eta.
                                    Note that a trailing ": " is automatically removed after {desc}
                                    if the latter is empty.
        initial (Optional[Union[int, float]]): The initial counter value. Useful when restarting a progress
                                            bar [default: 0]. If using float, consider specifying `{n:.3f}`
                                            or similar in `bar_format`, or specifying `unit_scale`.
        position (Optional[int]): Specify the line offset to print this bar (starting from 0)
                                    Automatic if unspecified.
                                    Useful to manage multiple bars at once (eg, from threads).
        postfix (Optional[Dict]): Specify additional stats to display at the end of the bar.
                                    Calls `set_postfix(**postfix)` if possible (dict).
        unit_divisor (Optional[float]): [default: 1000], ignored unless `unit_scale` is True.
        write_bytes (Optional[bool]): If (default: None) and `file` is unspecified,
                                    bytes will be written in Python 2. If `True` will also write
                                    bytes. In all other cases will default to unicode.
        lock_args (Optional[tuple]): Passed to `refresh` for intermediate output
                                    (initialisation, iterating, and updating).
                                    nrows (Optional[int]): The screen height. If specified, hides nested bars
                                    outside this bound. If unspecified, attempts to use environment height.
                                    The fallback is 20.
        colour (Optional[str]): Bar colour (e.g. 'green', '#00ff00').
        delay (Optional[float]): Don't display until [default: 0] seconds have elapsed.
        gui (Optional[bool]): WARNING: internal parameter - do not use.
                                Use tqdm.gui.tqdm(...) instead. If set, will attempt to use
                                matplotlib animations for a graphical output [default: False].


    Example:
        >>> with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as p_bar:
        >>>         urllib.request.urlretrieve(url, filename=output_path, reporthook=p_bar.update_to)
    """

    def __init__(
        self,
        iterable: Optional[Iterable] = None,
        desc: Optional[str] = None,
        total: Optional[Union[int, float]] = None,
        leave: Optional[bool] = True,
        file: Optional[Union[io.TextIOWrapper, io.StringIO]] = None,
        ncols: Optional[int] = None,
        mininterval: Optional[float] = 0.1,
        maxinterval: Optional[float] = 10.0,
        miniters: Optional[Union[int, float]] = None,
        use_ascii: Optional[Union[bool, str]] = None,
        disable: Optional[bool] = False,
        unit: Optional[str] = "it",
        unit_scale: Optional[Union[bool, int, float]] = False,
        dynamic_ncols: Optional[bool] = False,
        smoothing: Optional[float] = 0.3,
        bar_format: Optional[str] = None,
        initial: Optional[Union[int, float]] = 0,
        position: Optional[int] = None,
        postfix: Optional[Dict] = None,
        unit_divisor: Optional[float] = 1000,
        write_bytes: Optional[bool] = None,
        lock_args: Optional[tuple] = None,
        nrows: Optional[int] = None,
        colour: Optional[str] = None,
        delay: Optional[float] = 0,
        gui: Optional[bool] = False,
        **kwargs,
    ):
        super().__init__(
            iterable=iterable,
            desc=desc,
            total=total,
            leave=leave,
            file=file,
            ncols=ncols,
            mininterval=mininterval,
            maxinterval=maxinterval,
            miniters=miniters,
            ascii=use_ascii,
            disable=disable,
            unit=unit,
            unit_scale=unit_scale,
            dynamic_ncols=dynamic_ncols,
            smoothing=smoothing,
            bar_format=bar_format,
            initial=initial,
            position=position,
            postfix=postfix,
            unit_divisor=unit_divisor,
            write_bytes=write_bytes,
            lock_args=lock_args,
            nrows=nrows,
            colour=colour,
            delay=delay,
            gui=gui,
            **kwargs,
        )
        self.total: Optional[Union[int, float]]

    def update_to(self, chunk_number: int = 1, max_chunk_size: int = 1, total_size=None):
        """Progress bar hook for tqdm.

        Based on https://stackoverflow.com/a/53877507
        The implementor does not have to bother about passing parameters to this as it gets them from urlretrieve.
        However the context needs a few parameters. Refer to the example.

        Args:
            chunk_number (int, optional): The current chunk being processed. Defaults to 1.
            max_chunk_size (int, optional): Maximum size of each chunk. Defaults to 1.
            total_size ([type], optional): Total download size. Defaults to None.
        """
        if total_size is not None:
            self.total = total_size
        self.update(chunk_number * max_chunk_size - self.n)


def hash_check(file_path: Path, expected_hash: str):
    """Raise assert error if hash does not match the calculated hash of the file.

    Args:
        file_path (Path): Path to file.
        expected_hash (str): Expected hash of the file.
    """
    with open(file_path, "rb") as hash_file:
        assert (
            hashlib.md5(hash_file.read()).hexdigest() == expected_hash
        ), f"Downloaded file {file_path} does not match the required hash."
