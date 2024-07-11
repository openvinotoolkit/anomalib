"""Tensorboard logger with add image interface."""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

try:
    from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
except ModuleNotFoundError:
    print("To use tensorboard logger install it using `pip install tensorboard`")
from lightning.pytorch.utilities import rank_zero_only

from .base import ImageLoggerBase


class AnomalibTensorBoardLogger(ImageLoggerBase, TensorBoardLogger):
    """Logger for tensorboard.

    Adds interface for `add_image` in the logger rather than calling the experiment object.

    .. note::
        Same as the Tensorboard Logger provided by PyTorch Lightning and the doc string is reproduced below.

    Logs are saved to
    ``os.path.join(save_dir, name, version)``. This is the default logger in Lightning, it comes
    preinstalled.

    Example:
        >>> from anomalib.engine import Engine
        >>> from anomalib.loggers import AnomalibTensorBoardLogger
        ...
        >>> logger = AnomalibTensorBoardLogger("tb_logs", name="my_model")
        >>> engine =  Engine(logger=logger)

    Args:
        save_dir (str): Save directory
        name (str | None): Experiment name. Defaults to ``'default'``.
            If it is the empty string then no per-experiment subdirectory is used.
            Default: ``'default'``.
        version (int | str | None): Experiment version. If version is not
            specified the logger inspects the save directory for existing
            versions, then automatically assigns the next available version.
            If it is a string then it is used as the run-specific subdirectory
            name, otherwise ``'version_${version}'`` is used.
            Defaults to ``None``
        log_graph (bool): Adds the computational graph to tensorboard. This
            requires that the user has defined the `self.example_input_array`
            attribute in their model.
            Defaults to ``False``.
        default_hp_metric (bool): Enables a placeholder metric with key
            ``hp_metric`` when ``log_hyperparams`` is called without a metric
            (otherwise calls to log_hyperparams without a metric are ignored).
            Defaults to ``True``.
        prefix (str): A string to put at the beginning of metric keys.
            Defaults to ``''``.
        **kwargs: Additional arguments like `comment`, `filename_suffix`, etc.
            used by :class:`SummaryWriter` can be passed as keyword arguments in
            this logger.
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
    ) -> None:
        super().__init__(
            save_dir,
            name=name,
            version=version,
            log_graph=log_graph,
            default_hp_metric=default_hp_metric,
            prefix=prefix,
            **kwargs,
        )
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    @rank_zero_only
    def add_image(self, image: np.ndarray | Figure, name: str | None = None, **kwargs) -> None:
        """Interface to add image to tensorboard logger.

        Args:
            image (np.ndarray | Figure): Image to log
            name (str | None): The tag of the image
                Defaults to ``None``.
            kwargs: Accepts only `global_step` (int). The step at which to log the image.
        """
        if "global_step" not in kwargs:
            msg = "`global_step` is required for tensorboard logger"
            raise ValueError(msg)

        # Need to call different functions of `SummaryWriter` for  Figure vs np.ndarray
        if isinstance(image, Figure):
            self.experiment.add_figure(figure=image, tag=name, close=False, **kwargs)
        else:
            self.experiment.add_image(img_tensor=image, tag=name, dataformats="HWC", **kwargs)
