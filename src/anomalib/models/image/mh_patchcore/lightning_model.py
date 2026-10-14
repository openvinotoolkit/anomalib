# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""MH-PatchCore Lightning model."""

from collections.abc import Sequence
from typing import Any

import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torchvision.transforms.v2 import CenterCrop, Compose, InterpolationMode, Normalize, Resize

from anomalib import LearningType
from anomalib.data import Batch
from anomalib.metrics import Evaluator
from anomalib.models.components import AnomalibModule, MemoryBankMixin
from anomalib.post_processing import PostProcessor
from anomalib.pre_processing import PreProcessor
from anomalib.visualization import Visualizer

from .torch_model import MHPatchcoreModel

_PCA_STAGE = 0
_COVARIANCE_STAGE = 1
_MEMORY_BANK_STAGE = 2
_FITTED_STAGE = 3
_STAGE_NAMES = ("PCA", "covariance", "memory bank")


class MHPatchcore(MemoryBankMixin, AnomalibModule):
    """MH-PatchCore Lightning module for anomaly detection.

    MH-PatchCore fits its statistical state in three ordered passes over the
    training data: PCA, covariance whitening, and memory-bank construction.

    Args:
        backbone (str): Name of the timm backbone. Defaults to
            ``"wide_resnet50_2.tv2_in1k"``.
        layers (Sequence[str]): Ordered backbone layers used for feature
            extraction. Defaults to ``("layer2", "layer3")``.
        pre_trained (bool): Whether to load pretrained backbone weights.
            Defaults to ``True``.
        pca_variance_ratio (float): Fraction of explained variance retained by
            incremental PCA. Defaults to ``0.99``.
        covariance_shrinkage (float): Fixed covariance shrinkage coefficient.
            Defaults to ``0.07``.
        memory_bank_size (int): Maximum number of vectors in the finalized
            memory bank. Defaults to ``1000``.
        local_coreset_size (int): Maximum number of vectors retained in each
            merge-reduce block. Defaults to ``256``.
        num_neighbors (int): Number of memory-bank neighbors used for image
            score reweighting. Defaults to ``9``.
        pre_processor (PreProcessor | bool): Preprocessor instance or flag to
            use the default. Defaults to ``True``.
        post_processor (PostProcessor | bool): Postprocessor instance or flag to
            use the default. Defaults to ``True``.
        evaluator (Evaluator | bool): Evaluator instance or flag to use the
            default. Defaults to ``True``.
        visualizer (Visualizer | bool): Visualizer instance or flag to use the
            default. Defaults to ``True``.
    """

    def __init__(
        self,
        backbone: str = "wide_resnet50_2.tv2_in1k",
        layers: Sequence[str] = ("layer2", "layer3"),
        pre_trained: bool = True,
        pca_variance_ratio: float = 0.99,
        covariance_shrinkage: float = 0.07,
        memory_bank_size: int = 1000,
        local_coreset_size: int = 256,
        num_neighbors: int = 9,
        pre_processor: PreProcessor | bool = True,
        post_processor: PostProcessor | bool = True,
        evaluator: Evaluator | bool = True,
        visualizer: Visualizer | bool = True,
    ) -> None:
        super().__init__(
            pre_processor=pre_processor,
            post_processor=post_processor,
            evaluator=evaluator,
            visualizer=visualizer,
        )
        self.model = MHPatchcoreModel(
            backbone=backbone,
            layers=layers,
            pre_trained=pre_trained,
            pca_variance_ratio=pca_variance_ratio,
            covariance_shrinkage=covariance_shrinkage,
            memory_bank_size=memory_bank_size,
            local_coreset_size=local_coreset_size,
            num_neighbors=num_neighbors,
        )
        self.register_buffer("_fitting_stage", torch.tensor(_PCA_STAGE, dtype=torch.int64))
        self.register_buffer("_stage_batch_count", torch.tensor(0, dtype=torch.int64))
        self._fitting_stage: torch.Tensor
        self._stage_batch_count: torch.Tensor

    @staticmethod
    def configure_pre_processor(image_size: tuple[int, int] | None = None) -> PreProcessor:
        """Configure the canonical MH-PatchCore preprocessor.

        Args:
            image_size (tuple[int, int] | None): Resize dimensions before the
                fixed ``224 x 224`` center crop. Defaults to ``(256, 256)``.

        Returns:
            PreProcessor: Resize, center-crop, and ImageNet normalization
                transforms used by MH-PatchCore.

        Raises:
            ValueError: If either resize dimension is smaller than the fixed
                center-crop size.
        """
        image_size = image_size or (256, 256)
        if image_size[0] < 224 or image_size[1] < 224:
            msg = f"Image size {image_size} cannot be smaller than center crop size (224, 224)."
            raise ValueError(msg)
        return PreProcessor(
            transform=Compose([
                Resize(image_size, interpolation=InterpolationMode.BILINEAR, antialias=True),
                CenterCrop((224, 224)),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]),
        )

    @staticmethod
    def configure_optimizers() -> None:
        """Return no optimizer because MH-PatchCore performs statistical fitting."""
        return

    def training_step(self, batch: Batch, *args, **kwargs) -> torch.Tensor:
        """Update the statistical component for the active fitting pass.

        Args:
            batch (Batch): Batch containing training images.
            *args: Additional positional arguments, which are unused.
            **kwargs: Additional keyword arguments, which are unused.

        Returns:
            torch.Tensor: Dummy scalar loss required by Lightning.

        Raises:
            RuntimeError: If training continues after all fitting passes.
            TypeError: If the Torch model does not return training embeddings.
        """
        del args, kwargs
        embeddings = self.model(batch.image)
        if not isinstance(embeddings, torch.Tensor):
            msg = "MHPatchcoreModel must return embeddings during fitting."
            raise TypeError(msg)

        stage = int(self._fitting_stage.item())
        if stage == _PCA_STAGE:
            self.model.pca.update(embeddings)
        elif stage == _COVARIANCE_STAGE:
            self.model.covariance.update(self.model.pca(embeddings))
        elif stage == _MEMORY_BANK_STAGE:
            projected_embeddings = self.model.pca(embeddings)
            self.model.memory_bank.update(self.model.covariance(projected_embeddings))
        else:
            msg = "MHPatchcore is already fully fitted and cannot start another fitting pass."
            raise RuntimeError(msg)

        self._stage_batch_count.add_(1)
        return torch.tensor(0.0, requires_grad=True, device=self.device)

    def fit(self) -> None:
        """Finalize the active fitting pass.

        Raises:
            RuntimeError: If the active pass received no training batches or the
                persisted fitting stage is invalid.
        """
        stage = int(self._fitting_stage.item())
        if stage == _FITTED_STAGE:
            return
        if stage not in {_PCA_STAGE, _COVARIANCE_STAGE, _MEMORY_BANK_STAGE}:
            msg = f"MHPatchcore has invalid fitting stage {stage}."
            raise RuntimeError(msg)
        if int(self._stage_batch_count.item()) == 0:
            msg = f"Cannot finalize the {_STAGE_NAMES[stage]} pass because it received no training batches."
            raise RuntimeError(msg)

        if stage == _PCA_STAGE:
            self.model.pca.finalize()
        elif stage == _COVARIANCE_STAGE:
            self.model.covariance.finalize()
        else:
            self.model.memory_bank.finalize()

        self._fitting_stage.add_(1)
        self._stage_batch_count.zero_()
        if stage == _MEMORY_BANK_STAGE:
            self._is_fitted.fill_(1)

    def on_validation_start(self) -> None:
        """Finalize the memory bank immediately before final-epoch validation.

        Raises:
            RuntimeError: If validation starts before the memory-bank pass.
        """
        if bool(self._is_fitted.item()):
            return
        stage = int(self._fitting_stage.item())
        if stage != _MEMORY_BANK_STAGE:
            stage_name = _STAGE_NAMES[stage] if stage in range(len(_STAGE_NAMES)) else str(stage)
            msg = f"MHPatchcore validation requires all three fitting passes, but the active pass is {stage_name}."
            raise RuntimeError(msg)
        self.fit()

    def on_train_epoch_end(self) -> None:
        """Finalize the active pass after its training epoch."""
        if not bool(self._is_fitted.item()):
            self.fit()

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Reject mid-pass checkpoints, then restore shared Anomalib state.

        Transient PCA, covariance, and merge-reduce accumulators are not
        serialized. Epoch-boundary and fully fitted checkpoints have a zero
        stage batch count and can be restored safely. Calling ``super`` keeps
        component enable/disable flags and safe post-processor config restore
        from :class:`~anomalib.models.components.AnomalibModule`.

        Args:
            checkpoint (dict[str, Any]): Lightning checkpoint to validate.

        Raises:
            RuntimeError: If the checkpoint was saved after processing one or
                more batches of an unfinished fitting pass.
        """
        stage_batch_count = checkpoint["state_dict"].get("_stage_batch_count")
        if stage_batch_count is not None and int(stage_batch_count.item()) != 0:
            msg = (
                f"Cannot restore a {self.__class__.__name__} checkpoint saved in the middle of a fitting pass. "
                "Resume from an epoch-boundary or fully fitted checkpoint."
            )
            raise RuntimeError(msg)
        super().on_load_checkpoint(checkpoint)

    def validation_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        """Generate anomaly predictions for a validation batch.

        Args:
            batch (Batch): Batch containing validation images and metadata.
            *args: Additional positional arguments, which are unused.
            **kwargs: Additional keyword arguments, which are unused.

        Returns:
            STEP_OUTPUT: Batch updated with image scores and anomaly maps.
        """
        del args, kwargs
        predictions = self.model(batch.image)
        return batch.update(**predictions._asdict())

    @property
    def trainer_arguments(self) -> dict[str, int]:
        """Return trainer arguments required by the three fitting passes.

        Returns:
            dict[str, int]: Single-device, three-epoch trainer configuration.
        """
        return {
            "gradient_clip_val": 0,
            "max_epochs": 3,
            "check_val_every_n_epoch": 3,
            "num_sanity_val_steps": 0,
            "devices": 1,
        }

    @property
    def learning_type(self) -> LearningType:
        """Return the one-class learning type.

        Returns:
            LearningType: ``LearningType.ONE_CLASS``.
        """
        return LearningType.ONE_CLASS
