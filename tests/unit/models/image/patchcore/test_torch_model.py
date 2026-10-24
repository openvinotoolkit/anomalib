# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the PatchCore torch model and its write-through embedding store."""

import pytest
import torch
from _pytest.monkeypatch import MonkeyPatch
from torch import nn

from anomalib.models.image.patchcore import torch_model
from anomalib.models.image.patchcore.torch_model import PatchcoreModel


class MockFeatureExtractor(nn.Module):
    """Deterministic feature extractor that avoids downloading a backbone."""

    def __init__(self, backbone: str, layers: tuple[str, ...], pre_trained: bool) -> None:
        super().__init__()
        self.backbone = backbone
        self.layers = tuple(layers)
        self.pre_trained = pre_trained
        self.out_dims = (1, 1)

    def forward(self, input_tensor: torch.Tensor) -> dict[str, torch.Tensor]:  # noqa: PLR6301
        """Return input-dependent deterministic feature maps for both layers.

        The values scale with the input so different images produce
        distinguishable, per-image embeddings.
        """
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


@pytest.fixture
def model(monkeypatch: MonkeyPatch) -> PatchcoreModel:
    """Create a PatchcoreModel without constructing a real backbone."""
    monkeypatch.setattr(torch_model, "TimmFeatureExtractor", MockFeatureExtractor)
    return PatchcoreModel(layers=["layer2", "layer3"], pre_trained=False)


def collect_training_embeddings(model: PatchcoreModel, batches: list[torch.Tensor]) -> None:
    """Drive the model through training-mode forwards and end-of-training fit."""
    model.train()
    for images in batches:
        model(images)
    model.subsample_embedding(sampling_ratio=1.0)
    model.eval()


class TestSubsampleErrors:
    """The historical fit-time error contract must be preserved."""

    @staticmethod
    def test_subsample_without_embeddings_raises() -> None:
        """Fitting without any training embeddings must fail with the historical message."""
        from anomalib.models.image.patchcore.torch_model import PatchcoreModel

        model = PatchcoreModel(layers=["layer2"], pre_trained=False)
        with pytest.raises(ValueError, match="Embedding store is empty"):
            model.subsample_embedding(sampling_ratio=0.1)


class TestSubsampleEquivalence:
    """The write-through store must produce a byte-identical memory bank."""

    @staticmethod
    def test_memory_bank_contains_exactly_the_training_embeddings(model: PatchcoreModel) -> None:
        """With ratio 1.0 the bank must be exactly the staged rows (as a set).

        KCenterGreedy starts from a random index, so with ``sampling_ratio=1``
        every row is selected but in greedy order. Sort both sides on their
        content to compare them order-independently.
        """
        batches = [torch.randn(2, 3, 17, 19) for _ in range(3)]
        model.train()
        embeddings = [model(images).clone() for images in batches]
        expected = torch.vstack(embeddings)

        torch.manual_seed(0)
        model.subsample_embedding(sampling_ratio=1.0)

        assert model.memory_bank.shape == expected.shape
        order_expected = expected[:, 0].argsort()
        order_actual = model.memory_bank[:, 0].argsort()
        assert torch.equal(model.memory_bank[order_actual], expected[order_expected])

    @staticmethod
    def test_staged_rows_keep_batch_order(model: PatchcoreModel) -> None:
        """Consolidated rows must be in batch order, exactly like ``torch.vstack``.

        KCenterGreedy deliberately reorders rows during coreset selection, so
        the order contract is asserted on the consolidated staging tensor -
        the input the sampler receives - not on the subsampled bank.
        """
        batches = [torch.full((1, 3, 17, 19), float(idx + 1)) for idx in range(3)]
        model.train()
        embeddings = [model(images).clone() for images in batches]

        consolidated = model.embedding_store.consolidate()

        assert torch.equal(consolidated, torch.vstack(embeddings))
        # The mock produces a 3x4 patch grid, so each image contributes 12
        # rows; the per-image value scales with the input, so rows of later
        # batches must dominate earlier ones, blockwise.
        rows_per_image = 12
        first_block = consolidated[:rows_per_image]
        second_block = consolidated[rows_per_image : 2 * rows_per_image]
        third_block = consolidated[2 * rows_per_image :]
        assert torch.all(first_block > 0)
        assert torch.all(second_block > first_block)
        assert torch.all(third_block > second_block)

    @staticmethod
    def test_fit_preserves_every_training_embedding(model: PatchcoreModel) -> None:
        """A ratio-1.0 fit must keep every staged row, none duplicated.

        KCenterGreedy picks a random starting index, so the bank is ordered by
        the greedy walk rather than by batch; what must hold is that the fit
        loses nothing: the bank is exactly the multiset of staged rows.
        """
        batches = [torch.randn(2, 3, 17, 19) for _ in range(3)]
        model.train()
        embeddings = [model(images).clone() for images in batches]
        staged = torch.vstack(embeddings)

        torch.manual_seed(0)
        model.subsample_embedding(sampling_ratio=1.0)

        assert model.memory_bank.shape == staged.shape
        expected_order = staged[:, 0].argsort()
        actual_order = model.memory_bank[:, 0].argsort()
        assert torch.equal(model.memory_bank[actual_order], staged[expected_order])


class TestWriteThroughMemory:
    """The memory regression this change fixes: no second full-size copy at fit."""

    @staticmethod
    def test_fit_never_vstacks_the_full_corpus(model: PatchcoreModel) -> None:
        """Consolidation must not materialize a second full-size embedding copy.

        The historical implementation held the per-batch list and allocated the
        full stacked matrix at fit time, doubling peak memory exactly when the
        bank is largest (gh-3815). This test fails on that implementation: it
        spies on ``torch.vstack`` during the whole fit and fails if any single
        call carries the entire corpus.
        """
        batches = [torch.randn(4, 3, 17, 19) for _ in range(4)]
        model.train()
        first = model(batches[0]).clone()
        model.embedding_store.upgrade(
            num_rows=first.shape[0] * len(batches),
            num_features=first.shape[1],
            device=first.device,
            dtype=first.dtype,
        )
        rest = [model(images).clone() for images in batches[1:]]
        embeddings = [first, *rest]
        staged_bytes = torch.vstack(embeddings).numel() * embeddings[0].element_size()

        calls: list[int] = []
        real_vstack = torch.vstack

        def spy(*args: object, **kwargs: object) -> torch.Tensor:
            tensors = args[0] if args else kwargs.get("tensors")
            if isinstance(tensors, (list, tuple)) and tensors:
                live = sum(t.numel() * t.element_size() for t in tensors)
                calls.append(live)
            return real_vstack(*args, **kwargs)

        torch.vstack = spy  # type: ignore[assignment]
        try:
            model.subsample_embedding(sampling_ratio=1.0)
        finally:
            torch.vstack = real_vstack  # type: ignore[assignment]

        expected_rows = sum(emb.shape[0] for emb in embeddings)
        assert model.memory_bank.shape[0] == expected_rows
        # The staged corpus itself is never handed to a single vstack call.
        assert not any(live >= staged_bytes for live in calls), (
            f"fit materialized a full-size copy via vstack: {calls} vs staged {staged_bytes} bytes"
        )

    @staticmethod
    def test_exact_reservation_is_write_through(model: PatchcoreModel) -> None:
        """A reserved store must write batches in place, not keep a list of them.

        On the historical list implementation this fails outright: there is no
        ``reserve``/``upgrade`` at all, and the per-batch chunks stay alive
        until fit.
        """
        batches = [torch.randn(2, 3, 17, 19) for _ in range(3)]
        model.train()
        first = model(batches[0])

        store = model.embedding_store
        store.upgrade(
            num_rows=first.shape[0] * len(batches),
            num_features=first.shape[1],
            device=first.device,
            dtype=first.dtype,
        )
        _ = model(batches[1])
        _ = model(batches[2])

        assert store.is_staged
        # The per-batch embedding returned by forward is a fresh tensor each
        # time; the store must not hold references to any of them.
        staged_chunks = getattr(store, "_chunks", None)
        assert staged_chunks == []
        # Consolidation hands over the exactly-filled staging tensor itself.
        consolidated = store.consolidate()
        assert consolidated.data_ptr() == store._staging.data_ptr()  # noqa: SLF001

    @staticmethod
    def test_memory_bank_is_not_the_staging_tensor(model: PatchcoreModel) -> None:
        """After fit, the memory bank must own its storage.

        The coreset sampler returns a new (smaller) tensor, so the bank is
        naturally detached from the staging allocation. This pins that no
        future regression hands the oversized staging storage to the bank.
        """
        batches = [torch.randn(2, 3, 17, 19) for _ in range(3)]
        model.train()
        for images in batches:
            _ = model(images)

        torch.manual_seed(0)
        model.subsample_embedding(sampling_ratio=0.5)

        assert model.memory_bank.shape[0] < 3 * 2 * 12
        store = model.embedding_store
        assert store.is_list_mode or store._staging is None  # noqa: SLF001
        assert store.num_rows == 0
