"""wandb logger with add image interface."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.figure import Figure
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.utilities import rank_zero_only

try:
    import wandb
except ModuleNotFoundError:
    print("To use wandb logger install it using `pip install wandb`")

from .base import ImageLoggerBase


class AnomalibWandbLogger(ImageLoggerBase, WandbLogger):
    """Logger for wandb.

    Adds interface for `add_image` in the logger rather than calling the experiment object.

    Note:
        Same as the wandb Logger provided by PyTorch Lightning and the doc string is reproduced below.

    Log using `Weights and Biases <https://www.wandb.com/>`_.

    Install it with pip:

    .. code-block:: bash

        $ pip install wandb

    Args:
        name: Display name for the run.
        save_dir: Path where data is saved (wandb dir by default).
        offline: Run offline (data can be streamed later to wandb servers).
        id: Sets the version, mainly used to resume a previous run.
        version: Same as id.
        anonymous: Enables or explicitly disables anonymous logging.
        project: The name of the project to which this run will belong.
        log_model: Save checkpoints in wandb dir to upload on W&B servers.
        prefix: A string to put at the beginning of metric keys.
        experiment: WandB experiment object. Automatically set when creating a run.
        **kwargs: Arguments passed to :func:`wandb.init` like `entity`, `group`, `tags`, etc.

    Raises:
        ImportError:
            If required WandB package is not installed on the device.
        MisconfigurationException:
            If both ``log_model`` and ``offline``is set to ``True``.

    Example:
        >>> from anomalib.utils.loggers import AnomalibWandbLogger
        >>> from pytorch_lightning import Trainer
        >>> wandb_logger = AnomalibWandbLogger()
        >>> trainer = Trainer(logger=wandb_logger)

    Note: When logging manually through `wandb.log` or `trainer.logger.experiment.log`,
    make sure to use `commit=False` so the logging step does not increase.

    See Also:
        - `Tutorial <https://colab.research.google.com/drive/16d1uctGaw2y9KhGBlINNTsWpmlXdJwRW?usp=sharing>`__
          on how to use W&B with PyTorch Lightning
        - `W&B Documentation <https://docs.wandb.ai/integrations/lightning>`__

    """

    def __init__(
        self,
        name: str | None = None,
        save_dir: str | None = None,
        offline: bool | None = False,
        id: str | None = None,  # kept to match wandb init pylint: disable=redefined-builtin
        anonymous: bool | None = None,
        version: str | None = None,
        project: str | None = None,
        log_model: str | bool = False,
        experiment=None,
        prefix: str | None = "",
        **kwargs,
    ) -> None:
        super().__init__(
            name=name,
            save_dir=save_dir,
            offline=offline,
            id=id,
            anonymous=anonymous,
            version=version,
            project=project,
            log_model=log_model,
            experiment=experiment,
            prefix=prefix,
            **kwargs,
        )
        self.image_list: list[wandb.Image] = []  # Cache images

    @rank_zero_only
    def add_image(self, image: np.ndarray | Figure, name: str | None = None, **kwargs: Any):
        """Interface to add image to wandb logger.

        Args:
            image (np.ndarray | Figure): Image to log
            name (str | None): The tag of the image
        """
        image = wandb.Image(image, caption=name)
        self.image_list.append(image)

    @rank_zero_only
    def save(self) -> None:
        """Upload images to wandb server.

        Note:
            There is a limit on the number of images that can be logged together to the `wandb` server.
        """
        super().save()
        if len(self.image_list) > 1:
            wandb.log({"Predictions": self.image_list})
            self.image_list = []
