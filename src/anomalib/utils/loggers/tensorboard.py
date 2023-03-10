"""Tensorboard logger with add image interface."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.figure import Figure

try:
    from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
except ModuleNotFoundError:
    print("To use tensorboard logger install it using `pip install tensorboard`")
from pytorch_lightning.utilities import rank_zero_only

from .base import ImageLoggerBase


class AnomalibTensorBoardLogger(ImageLoggerBase, TensorBoardLogger):
    """Logger for tensorboard.

    Adds interface for `add_image` in the logger rather than calling the experiment object.

    Note:
        Same as the Tensorboard Logger provided by PyTorch Lightning and the doc string is reproduced below.

    Logs are saved to
    ``os.path.join(save_dir, name, version)``. This is the default logger in Lightning, it comes
    preinstalled.

    Example:
        >>> from pytorch_lightning import Trainer
        >>> from anomalib.utils.loggers import AnomalibTensorBoardLogger
        >>> logger = AnomalibTensorBoardLogger("tb_logs", name="my_model")
        >>> trainer = Trainer(logger=logger)

    Args:
        save_dir (str): Save directory
        name (Optional, str): Experiment name. Defaults to ``'default'``. If it is the empty string then no
            per-experiment subdirectory is used.
        version (Optional, int, str): Experiment version. If version is not specified the logger inspects the save
            directory for existing versions, then automatically assigns the next available version.
            If it is a string then it is used as the run-specific subdirectory name,
            otherwise ``'version_${version}'`` is used.
        log_graph (bool): Adds the computational graph to tensorboard. This requires that
            the user has defined the `self.example_input_array` attribute in their
            model.
        default_hp_metric (bool): Enables a placeholder metric with key `hp_metric` when `log_hyperparams` is
            called without a metric (otherwise calls to log_hyperparams without a metric are ignored).
        prefix (str): A string to put at the beginning of metric keys.
        **kwargs: Additional arguments like `comment`, `filename_suffix`, etc. used by
            :class:`SummaryWriter` can be passed as keyword arguments in this logger.
    """

    def __init__(
        self,
        save_dir: str,
        name: str | None = "default",
        version: int | str | None = None,
        log_graph: bool = False,
        default_hp_metric: bool = True,
        prefix: str = "",
        **kwargs,
    ):
        super().__init__(
            save_dir,
            name=name,
            version=version,
            log_graph=log_graph,
            default_hp_metric=default_hp_metric,
            prefix=prefix,
            **kwargs,
        )

    @rank_zero_only
    def add_image(self, image: np.ndarray | Figure, name: str | None = None, **kwargs: Any):
        """Interface to add image to tensorboard logger.

        Args:
            image (np.ndarray | Figure): Image to log
            name (str | None): The tag of the image
            kwargs: Accepts only `global_step` (int). The step at which to log the image.
        """
        if "global_step" not in kwargs:
            raise ValueError("`global_step` is required for tensorboard logger")

        # Need to call different functions of `SummaryWriter` for  Figure vs np.ndarray
        if isinstance(image, Figure):
            self.experiment.add_figure(figure=image, tag=name, close=False, **kwargs)
        else:
            self.experiment.add_image(img_tensor=image, tag=name, dataformats="HWC", **kwargs)
