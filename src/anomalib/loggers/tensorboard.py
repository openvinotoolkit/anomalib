"""TensorBoard logger with image logging capabilities.

This module provides a TensorBoard logger implementation that adds an interface for
logging images. It extends both the base image logger and PyTorch Lightning's
TensorBoard logger.

Example:
    >>> from anomalib.loggers import AnomalibTensorBoardLogger
    >>> from anomalib.engine import Engine
    >>> tensorboard_logger = AnomalibTensorBoardLogger("logs")
    >>> engine = Engine(logger=tensorboard_logger)  # doctest: +SKIP

    Log an image:
    >>> import numpy as np
    >>> image = np.random.rand(32, 32, 3)  # doctest: +SKIP
    >>> tensorboard_logger.add_image(
    ...     image=image,
    ...     name="test_image",
    ...     global_step=0
    ... )  # doctest: +SKIP
"""

# Copyright (C) 2022-2025 Intel Corporation
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
    """Logger for TensorBoard with image logging capabilities.

    This logger extends PyTorch Lightning's TensorBoardLogger with an interface
    for logging images. It inherits from both :class:`ImageLoggerBase` and
    :class:`TensorBoardLogger`.

    Args:
        save_dir: Directory path where logs will be saved. The final path will be
            ``os.path.join(save_dir, name, version)``.
        name: Name of the experiment. If it is an empty string, no
            per-experiment subdirectory is used. Defaults to ``"default"``.
        version: Version of the experiment. If not specified, the logger checks
            the save directory for existing versions and assigns the next
            available one. If a string is provided, it is used as the
            run-specific subdirectory name. Otherwise ``"version_${version}"`` is
            used. Defaults to ``None``.
        log_graph: If ``True``, adds the computational graph to TensorBoard. This
            requires that the model has defined the ``example_input_array``
            attribute. Defaults to ``False``.
        default_hp_metric: If ``True``, enables a placeholder metric with key
            ``hp_metric`` when ``log_hyperparams`` is called without a metric.
            Defaults to ``True``.
        prefix: String to prepend to metric keys. Defaults to ``""``.
        **kwargs: Additional arguments like ``comment``, ``filename_suffix``,
            etc. used by :class:`SummaryWriter`.

    Example:
        >>> from anomalib.loggers import AnomalibTensorBoardLogger
        >>> from anomalib.engine import Engine
        >>> logger = AnomalibTensorBoardLogger(
        ...     save_dir="logs",
        ...     name="my_experiment"
        ... )  # doctest: +SKIP
        >>> engine = Engine(logger=logger)  # doctest: +SKIP

    See Also:
        - `TensorBoard Documentation <https://www.tensorflow.org/tensorboard>`_
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
        """Log images to TensorBoard.

        Args:
            image: Image to log, can be either a numpy array or matplotlib
                Figure.
            name: Name/title of the image. Defaults to ``None``.
            **kwargs: Must contain ``global_step`` (int) indicating the step at
                which to log the image. Additional keyword arguments are passed
                to the TensorBoard logging method.

        Raises:
            ValueError: If ``global_step`` is not provided in ``kwargs``.
        """
        if "global_step" not in kwargs:
            msg = "`global_step` is required for tensorboard logger"
            raise ValueError(msg)

        # Need to call different functions of `SummaryWriter` for  Figure vs np.ndarray
        if isinstance(image, Figure):
            self.experiment.add_figure(figure=image, tag=name, close=False, **kwargs)
        else:
            self.experiment.add_image(img_tensor=image, tag=name, dataformats="HWC", **kwargs)
