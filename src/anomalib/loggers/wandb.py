"""Weights & Biases logger with image logging capabilities.

This module provides a Weights & Biases logger implementation that adds an
interface for logging images. It extends both the base image logger and PyTorch
Lightning's WandbLogger.

Example:
    >>> from anomalib.loggers import AnomalibWandbLogger
    >>> from anomalib.engine import Engine
    >>> wandb_logger = AnomalibWandbLogger()  # doctest: +SKIP
    >>> engine = Engine(logger=wandb_logger)  # doctest: +SKIP

    Log an image:
    >>> import numpy as np
    >>> image = np.random.rand(32, 32, 3)  # doctest: +SKIP
    >>> wandb_logger.add_image(
    ...     image=image,
    ...     name="test_image"
    ... )  # doctest: +SKIP
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Literal

import numpy as np
from lightning.fabric.utilities.types import _PATH
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from lightning_utilities.core.imports import module_available
from matplotlib.figure import Figure

from .base import ImageLoggerBase

if module_available("wandb"):
    import wandb

if TYPE_CHECKING:
    from wandb.sdk.lib import RunDisabled
    from wandb.sdk.wandb_run import Run


class AnomalibWandbLogger(ImageLoggerBase, WandbLogger):
    """Logger for Weights & Biases with image logging capabilities.

    This logger extends PyTorch Lightning's WandbLogger with an interface for
    logging images. It inherits from both :class:`ImageLoggerBase` and
    :class:`WandbLogger`.

    Args:
        name: Display name for the run. Defaults to ``None``.
        save_dir: Path where data is saved (wandb dir by default).
            Defaults to ``"."``.
        version: Sets the version, mainly used to resume a previous run.
            Defaults to ``None``.
        offline: Run offline (data can be streamed later to wandb servers).
            Defaults to ``False``.
        dir: Alias for ``save_dir``. Defaults to ``None``.
        id: Sets the version, mainly used to resume a previous run.
            Defaults to ``None``.
        anonymous: Enables or explicitly disables anonymous logging.
            Defaults to ``None``.
        project: The name of the project to which this run will belong.
            Defaults to ``None``.
        log_model: Save checkpoints in wandb dir to upload on W&B servers.
            Defaults to ``False``.
        experiment: WandB experiment object. Automatically set when creating a
            run. Defaults to ``None``.
        prefix: A string to put at the beginning of metric keys.
            Defaults to ``""``.
        checkpoint_name: Name of the checkpoint to save.
            Defaults to ``None``.
        **kwargs: Additional arguments passed to :func:`wandb.init` like
            ``entity``, ``group``, ``tags``, etc.

    Raises:
        ImportError: If required WandB package is not installed.
        MisconfigurationException: If both ``log_model`` and ``offline`` are
            set to ``True``.

    Example:
        >>> from anomalib.loggers import AnomalibWandbLogger
        >>> from anomalib.engine import Engine
        >>> wandb_logger = AnomalibWandbLogger(
        ...     project="my_project",
        ...     name="my_run"
        ... )  # doctest: +SKIP
        >>> engine = Engine(logger=wandb_logger)  # doctest: +SKIP

    Note:
        When logging manually through ``wandb.log`` or
        ``trainer.logger.experiment.log``, make sure to use ``commit=False``
        so the logging step does not increase.

    See Also:
        - `W&B Documentation <https://docs.wandb.ai/integrations/lightning>`_
    """

    def __init__(
        self,
        name: str | None = None,
        save_dir: _PATH = ".",
        version: str | None = None,
        offline: bool = False,
        dir: _PATH | None = None,  # kept to match wandb init # noqa: A002
        id: str | None = None,  # kept to match wandb init # noqa: A002
        anonymous: bool | None = None,
        project: str | None = None,
        log_model: Literal["all"] | bool = False,
        experiment: "Run | RunDisabled | None" = None,
        prefix: str = "",
        checkpoint_name: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            name=name,
            save_dir=save_dir,
            version=version,
            offline=offline,
            dir=dir,
            id=id,
            anonymous=anonymous,
            project=project,
            log_model=log_model,
            experiment=experiment,
            prefix=prefix,
            checkpoint_name=checkpoint_name,
            **kwargs,
        )
        self.image_list: list[wandb.Image] = []  # Cache images

    @rank_zero_only
    def add_image(self, image: np.ndarray | Figure, name: str | None = None, **kwargs) -> None:
        """Log an image to Weights & Biases.

        Args:
            image: Image to log, can be either a numpy array or matplotlib
                Figure.
            name: Name/title of the image. Defaults to ``None``.
            **kwargs: Additional keyword arguments passed to
                :class:`wandb.Image`. Currently unused.
        """
        del kwargs  # Unused argument.

        image = wandb.Image(image, caption=name)
        self.image_list.append(image)

    @rank_zero_only
    def save(self) -> None:
        """Upload images to Weights & Biases server.

        Note:
            There is a limit on the number of images that can be logged together
            to the W&B server.
        """
        super().save()
        if len(self.image_list) > 1:
            wandb.log({"Predictions": self.image_list})
            self.image_list = []
            self.image_list = []
