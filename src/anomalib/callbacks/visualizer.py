"""Visualizer callback.

This module provides the :class:`_VisualizationCallback` for generating and managing visualizations
in Anomalib. This callback is assigned by the Anomalib Engine internally.

The callback handles:
- Generating visualizations during model testing and prediction
- Saving visualizations to disk
- Showing visualizations interactively
- Logging visualizations to various logging backends

Example:
    Create visualization callback with multiple visualizers::

        >>> from anomalib.utils.visualization import ImageVisualizer, MetricsVisualizer
        >>> visualizers = [ImageVisualizer(), MetricsVisualizer()]
        >>> visualization_callback = _VisualizationCallback(
        ...     visualizers=visualizers,
        ...     save=True,
        ...     root="results/images"
        ... )

Note:
    This callback is used internally by the Anomalib Engine and should not be
    instantiated directly by users.
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
from anomalib.models import AnomalibModule
from anomalib.utils.visualization import BaseVisualizer, GeneratorResult, VisualizationStep

logger = logging.getLogger(__name__)


class _VisualizationCallback(Callback):
    """Callback for visualization that is used internally by the Engine.

    This callback handles the generation and management of visualizations during model
    testing and prediction. It supports saving, showing, and logging visualizations
    to various backends.

    Args:
        visualizers (BaseVisualizer | list[BaseVisualizer]): Visualizer objects that
            are used for computing the visualizations.
        save (bool, optional): Save the visualizations. Defaults to ``False``.
        root (Path | None, optional): The path to save the visualizations. Defaults to ``None``.
        log (bool, optional): Log the visualizations to the loggers. Defaults to ``False``.
        show (bool, optional): Show the visualizations. Defaults to ``False``.

    Examples:
        Create visualization callback with multiple visualizers::

            >>> from anomalib.utils.visualization import ImageVisualizer, MetricsVisualizer
            >>> visualizers = [ImageVisualizer(), MetricsVisualizer()]
            >>> visualization_callback = _VisualizationCallback(
            ...     visualizers=visualizers,
            ...     save=True,
            ...     root="results/images"
            ... )

    Note:
        This callback is used internally by the Anomalib Engine and should not be
        instantiated directly by users.

    Raises:
        ValueError: If ``root`` is ``None`` and ``save`` is ``True``.
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
        pl_module: AnomalibModule,
        outputs: STEP_OUTPUT | None,
        batch: Any,  # noqa: ANN401
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Generate visualizations at the end of a test batch.

        Args:
            trainer (Trainer): PyTorch Lightning trainer instance.
            pl_module (AnomalibModule): The current module being tested.
            outputs (STEP_OUTPUT | None): Outputs from the test step.
            batch (Any): Current batch of data.
            batch_idx (int): Index of the current batch.
            dataloader_idx (int, optional): Index of the dataloader. Defaults to 0.

        Example:
            Generate visualizations for a test batch::

                >>> from anomalib.utils.visualization import ImageVisualizer
                >>> callback = _VisualizationCallback(
                ...     visualizers=ImageVisualizer(),
                ...     save=True,
                ...     root="results/images"
                ... )
                >>> callback.on_test_batch_end(trainer, model, outputs, batch, 0)

        Raises:
            ValueError: If ``save`` is ``True`` but ``file_name`` is ``None``.
        """
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

    def on_test_end(self, trainer: Trainer, pl_module: AnomalibModule) -> None:
        """Generate visualizations at the end of testing.

        Args:
            trainer (Trainer): PyTorch Lightning trainer instance.
            pl_module (AnomalibModule): The module that was tested.

        Example:
            Generate visualizations at the end of testing::

                >>> from anomalib.utils.visualization import MetricsVisualizer
                >>> callback = _VisualizationCallback(
                ...     visualizers=MetricsVisualizer(),
                ...     save=True,
                ...     root="results/metrics"
                ... )
                >>> callback.on_test_end(trainer, model)

        Raises:
            ValueError: If ``save`` is ``True`` but ``file_name`` is ``None``.
        """
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
        pl_module: AnomalibModule,
        outputs: STEP_OUTPUT | None,
        batch: Any,  # noqa: ANN401
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Generate visualizations at the end of a prediction batch.

        Args:
            trainer (Trainer): PyTorch Lightning trainer instance.
            pl_module (AnomalibModule): The module being used for prediction.
            outputs (STEP_OUTPUT | None): Outputs from the prediction step.
            batch (Any): Current batch of data.
            batch_idx (int): Index of the current batch.
            dataloader_idx (int, optional): Index of the dataloader. Defaults to 0.

        Note:
            This method calls :meth:`on_test_batch_end` internally.
        """
        return self.on_test_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    def on_predict_end(self, trainer: Trainer, pl_module: AnomalibModule) -> None:
        """Generate visualizations at the end of prediction.

        Args:
            trainer (Trainer): PyTorch Lightning trainer instance.
            pl_module (AnomalibModule): The module that was used for prediction.

        Note:
            This method calls :meth:`on_test_end` internally.
        """
        return self.on_test_end(trainer, pl_module)

    @staticmethod
    def _add_to_logger(
        result: GeneratorResult,
        module: AnomalibModule,
        trainer: Trainer,
    ) -> None:
        """Add visualization to logger.

        Args:
            result (GeneratorResult): Output from the visualization generators.
            module (AnomalibModule): LightningModule from which the global step is extracted.
            trainer (Trainer): Trainer object containing the loggers.

        Example:
            Add visualization to logger::

                >>> result = generator.generate(...)  # Generate visualization
                >>> _VisualizationCallback._add_to_logger(result, model, trainer)

        Raises:
            ValueError: If ``file_name`` is ``None`` when attempting to log.
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
