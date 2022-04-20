"""wandb logger with add image interface."""

# Copyright (C) 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

from typing import Any, List, Optional, Union

import numpy as np
from matplotlib.figure import Figure
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.utilities import rank_zero_only

import wandb

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
        name: Optional[str] = None,
        save_dir: Optional[str] = None,
        offline: Optional[bool] = False,
        id: Optional[str] = None,  # kept to match wandb init pylint: disable=redefined-builtin
        anonymous: Optional[bool] = None,
        version: Optional[str] = None,
        project: Optional[str] = None,
        log_model: Union[str, bool] = False,
        experiment=None,
        prefix: Optional[str] = "",
        **kwargs
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
            **kwargs
        )
        self.image_list: List[wandb.Image] = []  # Cache images

    @rank_zero_only
    def add_image(self, image: Union[np.ndarray, Figure], name: Optional[str] = None, **kwargs: Any):
        """Interface to add image to wandb logger.

        Args:
            image (Union[np.ndarray, Figure]): Image to log
            name (Optional[str]): The tag of the image
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
