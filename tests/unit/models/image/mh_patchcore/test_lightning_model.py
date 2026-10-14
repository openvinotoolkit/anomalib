# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the MH-PatchCore Lightning model."""

from collections.abc import Sequence
from pathlib import Path

import pytest
import torch
from _pytest.monkeypatch import MonkeyPatch
from lightning.pytorch import Callback, LightningModule, Trainer
from torch import nn
from torch.utils.data import DataLoader

from anomalib import LearningType
from anomalib.data import ImageBatch, ImageItem
from anomalib.models import MHPatchcore, get_model, list_models
from anomalib.models.image.mh_patchcore import torch_model


class MockFeatureExtractor(nn.Module):
    """Return small deterministic feature maps without loading a backbone."""

    def __init__(self, backbone: str, layers: Sequence[str], pre_trained: bool) -> None:
        super().__init__()
        self.backbone = backbone
        self.layers = tuple(layers)
        self.pre_trained = pre_trained
        self.out_dims = (1, 1)

    def forward(self, input_tensor: torch.Tensor) -> dict[str, torch.Tensor]:  # noqa: PLR6301
        """Generate deterministic features for each input image."""
        batch_size = input_tensor.shape[0]
        layer2 = torch.arange(batch_size * 12, dtype=input_tensor.dtype, device=input_tensor.device)
        layer3 = torch.arange(batch_size * 4, dtype=input_tensor.dtype, device=input_tensor.device)
        return {
            "layer2": layer2.reshape(batch_size, 1, 3, 4),
            "layer3": layer3.reshape(batch_size, 1, 2, 2),
        }


class ValidationStateRecorder(Callback):
    """Record whether validation batches see a finalized memory bank."""

    def __init__(self) -> None:
        self.fitted_states: list[bool] = []

    def on_validation_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: ImageBatch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Record the fitted state immediately before validation inference."""
        del trainer, batch, batch_idx, dataloader_idx
        if not isinstance(pl_module, MHPatchcore):
            msg = "ValidationStateRecorder requires an MHPatchcore module."
            raise TypeError(msg)
        self.fitted_states.append(pl_module.model.memory_bank.is_fitted)


def make_model(monkeypatch: MonkeyPatch) -> MHPatchcore:
    """Construct a lightweight MH-PatchCore model for lifecycle tests."""
    monkeypatch.setattr(torch_model, "TimmFeatureExtractor", MockFeatureExtractor)
    return MHPatchcore(
        memory_bank_size=4,
        local_coreset_size=4,
        num_neighbors=2,
        pre_processor=False,
        post_processor=False,
        evaluator=False,
        visualizer=False,
    )


def make_loader() -> DataLoader:
    """Create a one-batch typed image loader."""
    items = [ImageItem(image=torch.zeros(3, 17, 19))]
    return DataLoader(items, batch_size=1, collate_fn=ImageBatch.collate, pin_memory=True)


def make_trainer(max_epochs: int, callbacks: list[Callback] | None = None) -> Trainer:
    """Create a quiet deterministic CPU trainer."""
    return Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=max_epochs,
        check_val_every_n_epoch=3,
        num_sanity_val_steps=0,
        callbacks=callbacks,
        deterministic=True,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
    )


@pytest.mark.parametrize("name", ["MHPatchcore", "mh_patchcore", "m_h_patchcore"])
def test_model_discovery(monkeypatch: MonkeyPatch, name: str) -> None:
    """Public model aliases should resolve to MH-PatchCore."""
    monkeypatch.setattr(torch_model, "TimmFeatureExtractor", MockFeatureExtractor)

    assert isinstance(get_model(name, pre_trained=False), MHPatchcore)


def test_model_listing() -> None:
    """Public model listings should include MH-PatchCore in each format."""
    assert "m_h_patchcore" in list_models(case="snake")
    assert "MHPatchcore" in list_models(case="pascal")
    assert "Mh Patchcore" in list_models(case="title")


def test_cli_construction(monkeypatch: MonkeyPatch) -> None:
    """The CLI should construct MH-PatchCore with typed overrides."""
    from anomalib.cli import AnomalibCLI

    monkeypatch.setattr(torch_model, "TimmFeatureExtractor", MockFeatureExtractor)
    cli = AnomalibCLI(
        [
            "fit",
            "--model",
            "MHPatchcore",
            "--model.backbone",
            "resnet18",
            "--model.pre_trained",
            "false",
            "--model.pca_variance_ratio",
            "0.1",
            "--model.memory_bank_size",
            "8",
            "--model.local_coreset_size",
            "4",
        ],
        run=False,
    )

    assert isinstance(cli.model, MHPatchcore)
    assert cli.model.model.backbone == "resnet18"
    assert cli.model.model.feature_extractor.pre_trained is False
    assert cli.model.model.pca.variance_ratio == 0.1
    assert cli.model.model.memory_bank.memory_bank_size == 8
    assert cli.model.model.memory_bank.local_coreset_size == 4


@pytest.mark.parametrize(
    ("argument", "value", "message"),
    [
        ("layers", (), "layers must contain"),
        ("pca_variance_ratio", True, "variance_ratio must be"),
        ("pca_variance_ratio", 0.0, "variance_ratio must be"),
        ("covariance_shrinkage", 1.1, "shrinkage must be"),
        ("memory_bank_size", 0, "memory_bank_size must be"),
        ("local_coreset_size", True, "local_coreset_size must be"),
        ("num_neighbors", -1, "num_neighbors must be"),
    ],
)
def test_constructor_validation(
    monkeypatch: MonkeyPatch,
    argument: str,
    value: object,
    message: str,
) -> None:
    """The public constructor should reject invalid fitting parameters."""
    monkeypatch.setattr(torch_model, "TimmFeatureExtractor", MockFeatureExtractor)

    with pytest.raises(ValueError, match=message):
        MHPatchcore(**{argument: value})


def test_model_configuration(monkeypatch: MonkeyPatch) -> None:
    """The wrapper should expose canonical trainer and preprocessing settings."""
    model = make_model(monkeypatch)
    processed = MHPatchcore.configure_pre_processor()(torch.zeros(1, 3, 300, 300))

    assert model.learning_type == LearningType.ONE_CLASS
    assert model.configure_optimizers() is None
    assert model.trainer_arguments == {
        "gradient_clip_val": 0,
        "max_epochs": 3,
        "check_val_every_n_epoch": 3,
        "num_sanity_val_steps": 0,
        "devices": 1,
    }
    assert processed.shape == (1, 3, 224, 224)
    with pytest.raises(ValueError, match="cannot be smaller than center crop"):
        MHPatchcore.configure_pre_processor((223, 256))


def test_three_pass_fit_finalizes_before_validation(monkeypatch: MonkeyPatch) -> None:
    """Final-epoch validation should run only after all fitting passes."""
    model = make_model(monkeypatch)
    loader = make_loader()
    recorder = ValidationStateRecorder()

    make_trainer(max_epochs=3, callbacks=[recorder]).fit(
        model,
        train_dataloaders=loader,
        val_dataloaders=loader,
    )

    assert recorder.fitted_states == [True]
    assert bool(model._is_fitted.item())  # noqa: SLF001
    assert model._fitting_stage.item() == 3  # noqa: SLF001
    assert model._stage_batch_count.item() == 0  # noqa: SLF001
    assert model.model.pca.is_fitted
    assert model.model.covariance.is_fitted
    assert model.model.memory_bank.is_fitted


def test_final_pass_falls_back_to_train_epoch_end(monkeypatch: MonkeyPatch) -> None:
    """A fit without validation should finalize the memory bank after epoch three."""
    model = make_model(monkeypatch)

    make_trainer(max_epochs=3).fit(model, train_dataloaders=make_loader())

    assert bool(model._is_fitted.item())  # noqa: SLF001
    assert model._fitting_stage.item() == 3  # noqa: SLF001
    assert model.model.memory_bank.is_fitted


def test_invalid_lifecycle_transitions_fail_clearly(monkeypatch: MonkeyPatch) -> None:
    """Empty passes and premature validation should not advance fitting state."""
    model = make_model(monkeypatch)

    with pytest.raises(RuntimeError, match=r"PCA pass.*no training batches"):
        model.fit()
    with pytest.raises(RuntimeError, match="validation requires all three fitting passes"):
        model.on_validation_start()

    assert model._fitting_stage.item() == 0  # noqa: SLF001
    assert not bool(model._is_fitted.item())  # noqa: SLF001


def test_fitted_checkpoint_roundtrip(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    """A fully fitted checkpoint should reproduce scores and anomaly maps."""
    model = make_model(monkeypatch)
    loader = make_loader()
    trainer = make_trainer(max_epochs=3)
    trainer.fit(model, train_dataloaders=loader)
    batch = next(iter(loader))
    model.eval()
    with torch.no_grad():
        expected = model.model(batch.image)

    checkpoint_path = tmp_path / "fitted.ckpt"
    trainer.save_checkpoint(checkpoint_path)
    restored = MHPatchcore.load_from_checkpoint(checkpoint_path, weights_only=True)
    restored.eval()
    with torch.no_grad():
        actual = restored.model(batch.image)

    assert restored.pre_processor is None
    assert restored.post_processor is None
    assert restored.evaluator is None
    assert restored.visualizer is None
    assert bool(restored._is_fitted.item())  # noqa: SLF001
    assert restored._fitting_stage.item() == 3  # noqa: SLF001
    assert restored._stage_batch_count.item() == 0  # noqa: SLF001
    torch.testing.assert_close(actual.pred_score, expected.pred_score, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(actual.anomaly_map, expected.anomaly_map, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("completed_epochs", [1, 2])
def test_epoch_boundary_checkpoint_resume(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    completed_epochs: int,
) -> None:
    """Checkpoints after PCA or covariance should resume the remaining passes."""
    initial = make_model(monkeypatch)
    initial_trainer = make_trainer(max_epochs=completed_epochs)
    initial_trainer.fit(initial, train_dataloaders=make_loader())
    assert initial._fitting_stage.item() == completed_epochs  # noqa: SLF001
    assert initial._stage_batch_count.item() == 0  # noqa: SLF001

    checkpoint_path = tmp_path / f"epoch-{completed_epochs}.ckpt"
    initial_trainer.save_checkpoint(checkpoint_path)
    resumed = make_model(monkeypatch)
    make_trainer(max_epochs=3).fit(
        resumed,
        train_dataloaders=make_loader(),
        val_dataloaders=make_loader(),
        ckpt_path=checkpoint_path,
    )

    assert bool(resumed._is_fitted.item())  # noqa: SLF001
    assert resumed._fitting_stage.item() == 3  # noqa: SLF001
    assert resumed._stage_batch_count.item() == 0  # noqa: SLF001
    assert resumed.model.pca.is_fitted
    assert resumed.model.covariance.is_fitted
    assert resumed.model.memory_bank.is_fitted


def test_mid_pass_checkpoint_is_rejected(monkeypatch: MonkeyPatch) -> None:
    """Loading a checkpoint with transient fitting progress should fail."""
    model = make_model(monkeypatch)
    model.train()
    model.training_step(next(iter(make_loader())))
    checkpoint = {"state_dict": model.state_dict()}
    restored = make_model(monkeypatch)

    with pytest.raises(RuntimeError, match="middle of a fitting pass"):
        restored.on_load_checkpoint(checkpoint)
