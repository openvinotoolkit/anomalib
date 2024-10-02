"""Visualizer Callback.

This is assigned by Anomalib Engine internally.
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from pathlib import Path
from typing import Any, cast

from lightning.pytorch import Callback, Trainer
from lightning.pytorch.utilities.types import STEP_OUTPUT

from anomalib.data.utils.image import save_image, show_image
from anomalib.loggers import AnomalibWandbLogger
from anomalib.loggers.base import ImageLoggerBase
from anomalib.models import AnomalyModule
from anomalib.utils.visualization import (
    BaseVisualizer,
    GeneratorResult,
    VisualizationStep,
)

logger = logging.getLogger(__name__)


class _VisualizationCallback(Callback):
    """Callback for visualization that is used internally by the Engine.

    Args:
        visualizers (BaseVisualizer | list[BaseVisualizer]):
            Visualizer objects that are used for computing the visualizations. Defaults to None.
        save (bool, optional): Save the image. Defaults to False.
        root (Path | None, optional): The path to save the images. Defaults to None.
        log (bool, optional): Log the images into the loggers. Defaults to False.
        show (bool, optional): Show the images. Defaults to False.

    Example:
        >>> visualizers = [ImageVisualizer(), MetricsVisualizer()]
        >>> visualization_callback = _VisualizationCallback(
        ... visualizers=visualizers,
        ...   save=True,
        ...   root="results/images"
        ... )

        CLI
        $ anomalib train --model Padim --data MVTec \
            --visualization.visualizers ImageVisualizer \
            --visualization.visualizers+=MetricsVisualizer
        or
        $ anomalib train --model Padim --data MVTec \
            --visualization.visualizers '[ImageVisualizer, MetricsVisualizer]'

    Raises:
        ValueError: Incase `root` is None and `save` is True.
    """

    def __init__(
        self,
        visualizers: BaseVisualizer | list[BaseVisualizer],
        save: bool = False,
        root: Path | None = None,
        log: bool = False,
        show: bool = False,
    ) -> None:
        self.save = save
        if save and root is None:
            msg = "`root` must be provided if save is True"
            raise ValueError(msg)
        self.root: Path = root if root is not None else Path()  # need this check for mypy
        self.log = log
        self.show = show
        self.generators = visualizers if isinstance(visualizers, list) else [visualizers]

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: AnomalyModule,
        outputs: STEP_OUTPUT | None,
        batch: Any,  # noqa: ANN401
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        for generator in self.generators:
            if generator.visualize_on == VisualizationStep.BATCH:
                for result in generator(
                    trainer=trainer,
                    pl_module=pl_module,
                    outputs=outputs,
                    batch=batch,
                    batch_idx=batch_idx,
                    dataloader_idx=dataloader_idx,
                ):
                    if self.save:
                        if result.file_name is None:
                            msg = "``save`` is set to ``True`` but file name is ``None``"
                            raise ValueError(msg)

                        # Get the filename to save the image.
                        # Filename is split based on the datamodule name and category.
                        # For example, if the filename is `MVTec/bottle/000.png`, then the
                        # filename is split based on `MVTec/bottle` and `000.png` is saved.
                        if trainer.datamodule is not None:
                            filename = str(result.file_name).split(
                                sep=f"{trainer.datamodule.name}/{trainer.datamodule.category}",
                            )[-1]
                        else:
                            filename = Path(result.file_name).name
                        save_image(image=result.image, root=self.root, filename=filename)
                    if self.show:
                        show_image(image=result.image, title=str(result.file_name))
                    if self.log:
                        self._add_to_logger(result, pl_module, trainer)

    def on_test_end(self, trainer: Trainer, pl_module: AnomalyModule) -> None:
        for generator in self.generators:
            if generator.visualize_on == VisualizationStep.STAGE_END:
                for result in generator(trainer=trainer, pl_module=pl_module):
                    if self.save:
                        if result.file_name is None:
                            msg = "``save`` is set to ``True`` but file name is ``None``"
                            raise ValueError(msg)
                        save_image(image=result.image, root=self.root, filename=result.file_name)
                    if self.show:
                        show_image(image=result.image, title=str(result.file_name))
                    if self.log:
                        self._add_to_logger(result, pl_module, trainer)

        for logger in trainer.loggers:
            if isinstance(logger, AnomalibWandbLogger):
                logger.save()

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: AnomalyModule,
        outputs: STEP_OUTPUT | None,
        batch: Any,  # noqa: ANN401
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        return self.on_test_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    def on_predict_end(self, trainer: Trainer, pl_module: AnomalyModule) -> None:
        return self.on_test_end(trainer, pl_module)

    @staticmethod
    def _add_to_logger(
        result: GeneratorResult,
        module: AnomalyModule,
        trainer: Trainer,
    ) -> None:
        """Add image to logger.

        Args:
            result (GeneratorResult): Output from the generators.
            module (AnomalyModule): LightningModule from which the global step is extracted.
            trainer (Trainer): Trainer object.
        """
        # Store names of logger and the logger in a dict
        available_loggers = {
            type(logger).__name__.lower().replace("logger", "").replace("anomalib", ""): logger
            for logger in trainer.loggers
        }
        # save image to respective logger
        if result.file_name is None:
            msg = "File name is None"
            raise ValueError(msg)
        filename = result.file_name
        image = result.image
        for log_to in available_loggers:
            # check if logger object is same as the requested object
            if isinstance(available_loggers[log_to], ImageLoggerBase):
                logger: ImageLoggerBase = cast(ImageLoggerBase, available_loggers[log_to])  # placate mypy
                _name = filename.parent.name + "_" + filename.name if isinstance(filename, Path) else filename
                logger.add_image(
                    image=image,
                    name=_name,
                    global_step=module.global_step,
                )
