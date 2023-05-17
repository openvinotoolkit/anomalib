"""File system logger.

This is responsible for logging images to the file system and writing metrics in a csv file.
"""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from matplotlib.figure import Figure
from pytorch_lightning.loggers import CSVLogger

from .base import ImageLoggerBase


class FileSystemLogger(ImageLoggerBase, CSVLogger):
    def __init__(self, save_dir: str | Path, name: str = "anomalib_logs", version: int | str | None = None):
        """Initialize the file system logger.

        Args:
            save_dir (str | Path): The directory where the logs will be saved.
            name (str, optional): The name of the experiment. Defaults to "anomalib_logs".
            version (int | str | None, optional): The version of the experiment. Defaults to None.
        """
        super().__init__(save_dir=save_dir, name=name, version=version)

    def add_image(self, image: np.ndarray | Figure, name: str | None = None, **kwargs: Any) -> None:
        assert name is not None, "Name of the image cannot be None."
        file_path = Path(self.log_dir) / name
        if isinstance(image, Figure):
            image.savefig(str(file_path))
        else:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(file_path), image)
