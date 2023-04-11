"""Helper to show progress bars with `urlretrieve`, check hash of file."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import io
import logging
import os
import tarfile
from dataclasses import dataclass
from pathlib import Path
from tarfile import TarError, TarFile
from typing import Iterable
from urllib.request import urlretrieve
from zipfile import ZipFile

from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class DownloadInfo:
    """Info needed to download a dataset from a url."""

    name: str
    url: str
    hash: str


class DownloadProgressBar(tqdm):
    """Create progress bar for urlretrieve. Subclasses `tqdm`.

    For information about the parameters in constructor, refer to `tqdm`'s documentation.

    Args:
        iterable (Iterable | None): Iterable to decorate with a progressbar.
                            Leave blank to manually manage the updates.
        desc (str | None): Prefix for the progressbar.
        total (int | float | None): The number of expected iterations. If unspecified,
                                            len(iterable) is used if possible. If float("inf") or as a last
                                            resort, only basic progress statistics are displayed
                                            (no ETA, no progressbar).
                                            If `gui` is True and this parameter needs subsequent updating,
                                            specify an initial arbitrary large positive number,
                                            e.g. 9e9.
        leave (bool | None): upon termination of iteration. If `None`, will leave only if `position` is `0`.
        file (io.TextIOWrapper |  io.StringIO | None): Specifies where to output the progress messages
                                                            (default: sys.stderr). Uses `file.write(str)` and
                                                            `file.flush()` methods.  For encoding, see
                                                            `write_bytes`.
        ncols (int | None): The width of the entire output message. If specified,
                            dynamically resizes the progressbar to stay within this bound.
                            If unspecified, attempts to use environment width. The
                            fallback is a meter width of 10 and no limit for the counter and
                            statistics. If 0, will not print any meter (only stats).
        mininterval (float | None): Minimum progress display update interval [default: 0.1] seconds.
        maxinterval (float | None): Maximum progress display update interval [default: 10] seconds.
                                    Automatically adjusts `miniters` to correspond to `mininterval`
                                    after long display update lag. Only works if `dynamic_miniters`
                                    or monitor thread is enabled.
        miniters (int | float | None): Minimum progress display update interval, in iterations.
                                            If 0 and `dynamic_miniters`, will automatically adjust to equal
                                            `mininterval` (more CPU efficient, good for tight loops).
                                            If > 0, will skip display of specified number of iterations.
                                            Tweak this and `mininterval` to get very efficient loops.
                                            If your progress is erratic with both fast and slow iterations
                                            (network, skipping items, etc) you should set miniters=1.
        use_ascii (str | bool | None): If unspecified or False, use unicode (smooth blocks) to fill
                                        the meter. The fallback is to use ASCII characters " 123456789#".
        disable (bool | None): Whether to disable the entire progressbar wrapper
                                    [default: False]. If set to None, disable on non-TTY.
        unit (str | None): String that will be used to define the unit of each iteration
                            [default: it].
        unit_scale (int | float | bool): If 1 or True, the number of iterations will be reduced/scaled
                            automatically and a metric prefix following the
                            International System of Units standard will be added
                            (kilo, mega, etc.) [default: False]. If any other non-zero
                            number, will scale `total` and `n`.
        dynamic_ncols (bool | None): If set, constantly alters `ncols` and `nrows` to the
                                        environment (allowing for window resizes) [default: False].
        smoothing (float | None): Exponential moving average smoothing factor for speed estimates
                                    (ignored in GUI mode). Ranges from 0 (average speed) to 1
                                    (current/instantaneous speed) [default: 0.3].
        bar_format (str | None):  Specify a custom bar string formatting. May impact performance.
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
        initial (int | float | None): The initial counter value. Useful when restarting a progress
                                            bar [default: 0]. If using float, consider specifying `{n:.3f}`
                                            or similar in `bar_format`, or specifying `unit_scale`.
        position (int | None): Specify the line offset to print this bar (starting from 0)
                                    Automatic if unspecified.
                                    Useful to manage multiple bars at once (eg, from threads).
        postfix (dict | None): Specify additional stats to display at the end of the bar.
                                    Calls `set_postfix(**postfix)` if possible (dict).
        unit_divisor (float | None): [default: 1000], ignored unless `unit_scale` is True.
        write_bytes (bool | None): If (default: None) and `file` is unspecified,
                                    bytes will be written in Python 2. If `True` will also write
                                    bytes. In all other cases will default to unicode.
        lock_args (tuple | None): Passed to `refresh` for intermediate output
                                    (initialisation, iterating, and updating).
                                    nrows (int | None): The screen height. If specified, hides nested bars
                                    outside this bound. If unspecified, attempts to use environment height.
                                    The fallback is 20.
        colour (str | None): Bar colour (e.g. 'green', '#00ff00').
        delay (float | None): Don't display until [default: 0] seconds have elapsed.
        gui (bool | None): WARNING: internal parameter - do not use.
                                Use tqdm.gui.tqdm(...) instead. If set, will attempt to use
                                matplotlib animations for a graphical output [default: False].


    Example:
        >>> with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as p_bar:
        >>>         urllib.request.urlretrieve(url, filename=output_path, reporthook=p_bar.update_to)
    """

    def __init__(
        self,
        iterable: Iterable | None = None,
        desc: str | None = None,
        total: int | float | None = None,
        leave: bool | None = True,
        file: io.TextIOWrapper | io.StringIO | None = None,
        ncols: int | None = None,
        mininterval: float | None = 0.1,
        maxinterval: float | None = 10.0,
        miniters: int | float | None = None,
        use_ascii: bool | str | None = None,
        disable: bool | None = False,
        unit: str | None = "it",
        unit_scale: bool | int | float | None = False,
        dynamic_ncols: bool | None = False,
        smoothing: float | None = 0.3,
        bar_format: str | None = None,
        initial: int | float | None = 0,
        position: int | None = None,
        postfix: dict | None = None,
        unit_divisor: float | None = 1000,
        write_bytes: bool | None = None,
        lock_args: tuple | None = None,
        nrows: int | None = None,
        colour: str | None = None,
        delay: float | None = 0,
        gui: bool | None = False,
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
        self.total: int | float | None

    def update_to(self, chunk_number: int = 1, max_chunk_size: int = 1, total_size=None) -> None:
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


def hash_check(file_path: Path, expected_hash: str) -> None:
    """Raise assert error if hash does not match the calculated hash of the file.

    Args:
        file_path (Path): Path to file.
        expected_hash (str): Expected hash of the file.
    """
    with file_path.open("rb") as hash_file:
        assert (
            hashlib.md5(hash_file.read()).hexdigest() == expected_hash
        ), f"Downloaded file {file_path} does not match the required hash."


def download_and_extract(root: Path, info: DownloadInfo) -> None:
    """Download and extract a dataset.

    Args:
        root (Path): Root directory where the dataset will be stored.
        info (DownloadInfo): Info needed to download the dataset.
    """
    root.mkdir(parents=True, exist_ok=True)

    # save the compressed file in the specified root directory, using the same file name as on the server
    downloaded_file_path = root / info.url.split("/")[-1]
    if downloaded_file_path.exists():
        logger.info("Existing dataset archive found. Skipping download stage.")
    else:
        logger.info("Downloading the %s dataset.", info.name)
        with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=info.name) as progress_bar:
            urlretrieve(  # nosec - suppress bandit warning (urls are hardcoded)
                url=f"{info.url}",
                filename=downloaded_file_path,
                reporthook=progress_bar.update_to,
            )
        logger.info("Checking the hash of the downloaded file.")
        hash_check(downloaded_file_path, info.hash)

    logger.info("Extracting dataset into root folder.")
    if downloaded_file_path.suffix == ".zip":
        with ZipFile(downloaded_file_path, "r") as zip_file:
            zip_file.extractall(root)
    elif downloaded_file_path.suffix in (".tar", ".gz", ".xz"):
        with tarfile.open(downloaded_file_path) as tar_file:
            safe_extract(tar_file, root)
    else:
        raise ValueError(f"Unrecognized file format: {downloaded_file_path}")

    logger.info("Cleaning up files.")
    (downloaded_file_path).unlink()


def is_within_directory(directory: Path, target: Path):
    """Checks if a target path is located within a given directory.

    Args:
        directory (Path): path of the parent directory
        target (Path): path of the target
    Returns:
        (bool): True if the target is within the directory, False otherwise
    """
    abs_directory = directory.resolve()
    abs_target = target.resolve()

    # TODO: replace with pathlib is_relative_to after switching to Python 3.10
    prefix = os.path.commonprefix([abs_directory, abs_target])
    return prefix == str(abs_directory)


def safe_extract(tar_file: TarFile, path: str | Path = "."):
    """Extract a tar file safely by first checking for attempted path traversal.

    Args:
        tar_file (TarFile): Tar file to be extracted
        path (str | Path): path in which the extracted files will be placed
    """
    path = Path(path)
    for member in tar_file.getmembers():
        member_path = path / member.name
        if not is_within_directory(path, member_path):
            raise TarError("Attempted Path Traversal in Tar File")

    tar_file.extractall(path)
