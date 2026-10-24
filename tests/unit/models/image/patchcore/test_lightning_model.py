# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Lightning-level tests for the PatchCore write-through embedding store.

These tests drive a real Lightning ``Trainer`` with a mocked backbone so the
engine-driven training flow - ``training_step`` per batch, ``fit`` at epoch end
- is exercised end to end, including the exact-capacity reservation logic.
"""

import torch
from _pytest.monkeypatch import MonkeyPatch
from lightning.pytorch import Trainer
from torch import nn
from torch.utils.data import DataLoader

from anomalib.data import ImageBatch, ImageItem
from anomalib.models.image.patchcore import lightning_model, torch_model


class MockFeatureExtractor(nn.Module):
    """Deterministic input-dependent feature extractor (no backbone download)."""

    def __init__(self, backbone: str, layers: tuple[str, ...], pre_trained: bool) -> None:
        super().__init__()
        self.backbone = backbone
        self.layers = tuple(layers)
        self.pre_trained = pre_trained

    def forward(self, input_tensor: torch.Tensor) -> dict[str, torch.Tensor]:  # noqa: PLR6301
        """Return input-scaled deterministic feature maps for both layers."""
        batch_size = input_tensor.shape[0]
        scale = input_tensor.flatten(1).mean(dim=1, keepdim=True)
        layer2 = scale * torch.arange(12, dtype=input_tensor.dtype, device=input_tensor.device)
        layer2 = layer2 + torch.arange(batch_size, dtype=input_tensor.dtype, device=input_tensor.device).unsqueeze(1)
        layer3 = scale * torch.arange(4, dtype=input_tensor.dtype, device=input_tensor.device)
        layer3 = layer3 + torch.arange(batch_size, dtype=input_tensor.dtype, device=input_tensor.device).unsqueeze(1)
        return {
            "layer2": layer2.reshape(batch_size, 1, 3, 4),
            "layer3": layer3.reshape(batch_size, 1, 2, 2),
        }


def make_model(monkeypatch: MonkeyPatch) -> "lightning_model.Patchcore":
    """Create a Patchcore module without constructing a real backbone."""
    monkeypatch.setattr(torch_model, "TimmFeatureExtractor", MockFeatureExtractor)
    return lightning_model.Patchcore(
        layers=["layer2", "layer3"],
        pre_trained=False,
        coreset_sampling_ratio=1.0,
        pre_processor=False,
        post_processor=False,
        evaluator=False,
        visualizer=False,
    )


def make_loader(num_images: int, batch_size: int = 2) -> DataLoader:
    """Create a deterministic image dataloader with per-image values."""
    items = [ImageItem(image=torch.full((3, 17, 19), float(idx + 1))) for idx in range(num_images)]
    return DataLoader(items, batch_size=batch_size, collate_fn=ImageBatch.collate, pin_memory=True)


def make_trainer() -> Trainer:
    """Create a quiet deterministic CPU trainer."""
    return Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        num_sanity_val_steps=0,
        deterministic=True,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
    )


class TrainingFlowSpy:
    """Records when the embedding store upgrades, relative to training batches."""

    def __init__(self, model: "lightning_model.Patchcore") -> None:
        self.batch_counter = 0
        self.upgraded_at_batch: int | None = None
        self.original_step = model.training_step
        self.original_upgrade = model.model.embedding_store.upgrade
        store = model.model.embedding_store
        spy = self

        def counting_step(batch: object, *args: object, **kwargs: object) -> object:
            result = spy.original_step(batch, *args, **kwargs)
            spy.batch_counter += 1
            return result

        def recording_upgrade(*args: object, **kwargs: object) -> bool:
            result = spy.original_upgrade(*args, **kwargs)
            if result and spy.upgraded_at_batch is None:
                spy.upgraded_at_batch = spy.batch_counter
            return result

        model.training_step = counting_step  # type: ignore[method-assign]
        store.upgrade = recording_upgrade  # type: ignore[method-assign]


def sort_rows_lexicographically(bank: torch.Tensor) -> torch.Tensor:
    """Return the bank rows sorted by column 0, ties broken by column 1."""
    import numpy as np

    order = np.lexsort((bank[:, 1].numpy(), bank[:, 0].numpy()))
    return bank[torch.as_tensor(order)]


class TestEngineDrivenWriteThrough:
    """The engine.fit flow must stage embeddings exactly and correctly."""

    @staticmethod
    def test_fit_upgrades_to_exact_staging(monkeypatch: MonkeyPatch) -> None:
        """After the first training batch, the store must be in staging mode.

        The reservation is sized from the dataloader length and the measured
        rows-per-image of the first batch. This fails on the historical
        implementation, where the store is a plain list that never upgrades.
        """
        model = make_model(monkeypatch)
        spy = TrainingFlowSpy(model)
        trainer = make_trainer()

        trainer.fit(model, train_dataloaders=make_loader(6))

        # The counter increments after each step, so an upgrade during the
        # first training_step is recorded at batch 0. fit() then consolidates
        # and clears the store, so staging state is gone by the end; the
        # upgrade record and the bank are what prove the write-through path.
        assert spy.upgraded_at_batch == 0, "upgrade must happen during the first batch"
        assert model.model.memory_bank.shape[0] == 72  # 6 images x 12 rows, ratio 1.0

    @staticmethod
    def test_engine_fit_bank_matches_list_mode_reference(monkeypatch: MonkeyPatch) -> None:
        """Staged fit and historical list fit must produce the same bank (as a set)."""
        model = make_model(monkeypatch)
        trainer = make_trainer()
        trainer.fit(model, train_dataloaders=make_loader(6))
        staged_bank = model.model.memory_bank.clone()

        # Reference: the same data through the historical list path.
        reference_model = make_model(monkeypatch)

        def _refuse_upgrade(*_args: object, **_kwargs: object) -> bool:
            return False

        reference_model.model.embedding_store.upgrade = _refuse_upgrade  # type: ignore[method-assign]
        reference_trainer = make_trainer()
        reference_trainer.fit(reference_model, train_dataloaders=make_loader(6))

        assert reference_model.model.embedding_store.is_list_mode
        assert staged_bank.shape == reference_model.model.memory_bank.shape
        # KCenterGreedy walks in random order, so compare the banks as sorted
        # multisets of rows (lexicographic by both columns - column 0 alone
        # has ties).
        staged_sorted = sort_rows_lexicographically(staged_bank)
        reference_sorted = sort_rows_lexicographically(reference_model.model.memory_bank)
        assert torch.equal(staged_sorted, reference_sorted)

    @staticmethod
    def test_fit_without_trainer_hint_stays_correct(monkeypatch: MonkeyPatch) -> None:
        """A reservation that cannot happen must not break the fit.

        This pins the fallback: reservation failures only cost the memory
        optimization, never correctness.
        """
        model = make_model(monkeypatch)

        def _no_reserve(*_args: object, **_kwargs: object) -> None:
            return None

        model._reserve_embedding_store = _no_reserve  # type: ignore[method-assign]  # noqa: SLF001
        trainer = make_trainer()

        trainer.fit(model, train_dataloaders=make_loader(6))

        assert model.model.memory_bank.shape[0] == 72
        assert model.model.embedding_store.is_list_mode

    @staticmethod
    def test_prediction_after_fit_uses_staged_bank(monkeypatch: MonkeyPatch) -> None:
        """After engine fit, eval-mode forward must produce predictions."""
        model = make_model(monkeypatch)
        trainer = make_trainer()
        trainer.fit(model, train_dataloaders=make_loader(6))

        model.eval()
        output = model.model(torch.full((2, 3, 17, 19), 3.0))

        assert output.pred_score.shape == (2,)
        assert output.anomaly_map.shape == (2, 1, 17, 19)
