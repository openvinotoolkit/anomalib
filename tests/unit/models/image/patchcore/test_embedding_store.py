# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the PatchCore write-through embedding store."""

import pytest
import torch

from anomalib.models.image.patchcore.embedding_store import EmbeddingStore


class TestListMode:
    """The fallback path must behave exactly like the historical list."""

    @staticmethod
    def test_order_matches_vstack() -> None:
        """Rows must come out in the order batches were pushed."""
        store = EmbeddingStore()
        first = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        second = torch.arange(40, 76, dtype=torch.float32).reshape(9, 4)

        store.push(first)
        store.push(second)

        assert torch.equal(store.consolidate(), torch.vstack([first, second]))

    @staticmethod
    def test_counters() -> None:
        """Row and batch counters must be exact in list mode."""
        store = EmbeddingStore()
        store.push(torch.zeros(3, 4))
        store.push(torch.zeros(9, 4))

        assert store.num_rows == 12
        assert store.num_batches == 2
        assert len(store) == 2
        assert store.is_list_mode
        assert not store.is_staged

    @staticmethod
    def test_consolidate_empty_raises() -> None:
        """Consolidating an empty store must fail like the historical code."""
        store = EmbeddingStore()
        with pytest.raises(ValueError, match="No embeddings collected"):
            store.consolidate()

    @staticmethod
    def test_push_non_2d_raises() -> None:
        """Only 2D embedding batches may be staged."""
        store = EmbeddingStore()
        with pytest.raises(ValueError, match="must be 2D"):
            store.push(torch.zeros(2, 4, 4))

    @staticmethod
    def test_clear_resets() -> None:
        """clear() must return the store to a pristine list-mode state."""
        store = EmbeddingStore()
        store.push(torch.zeros(3, 4))
        store.clear()

        assert store.num_rows == 0
        assert store.num_batches == 0
        assert store.is_list_mode
        assert store.first_chunk is None


class TestStagingMode:
    """The optimized path: exact reservation, write-through, zero-copy handover."""

    @staticmethod
    def test_exact_fill_is_zero_copy() -> None:
        """A fully filled staging tensor must be handed over without a copy."""
        store = EmbeddingStore()
        store.reserve(num_rows=12, num_features=4, device=torch.device("cpu"), dtype=torch.float32)
        first = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        second = torch.arange(40, 76, dtype=torch.float32).reshape(9, 4)
        store.push(first)
        store.push(second)

        consolidated = store.consolidate()

        assert torch.equal(consolidated, torch.vstack([first, second]))
        assert consolidated.shape == (12, 4)
        assert not store.is_list_mode

    @staticmethod
    def test_partial_fill_returns_view() -> None:
        """A short fill must return a view, never a full-size clone."""
        store = EmbeddingStore()
        store.reserve(num_rows=100, num_features=4, device=torch.device("cpu"), dtype=torch.float32)
        store.push(torch.arange(12, dtype=torch.float32).reshape(3, 4))

        consolidated = store.consolidate()

        assert consolidated.shape == (3, 4)
        assert torch.equal(consolidated, torch.arange(12, dtype=torch.float32).reshape(3, 4))

    @staticmethod
    def test_reserve_zero_rows_stays_list_mode() -> None:
        """A non-positive reservation must be a no-op, not a crash."""
        store = EmbeddingStore()
        store.reserve(num_rows=0, num_features=4, device=torch.device("cpu"), dtype=torch.float32)

        assert store.is_list_mode

    @staticmethod
    def test_reserve_negative_rows_raises() -> None:
        """Negative reservations are configuration errors."""
        store = EmbeddingStore()
        with pytest.raises(ValueError, match="negative"):
            store.reserve(num_rows=-1, num_features=4, device=torch.device("cpu"), dtype=torch.float32)

    @staticmethod
    def test_reserve_after_push_raises() -> None:
        """Reserving after training started must fail loudly, not corrupt."""
        store = EmbeddingStore()
        store.push(torch.zeros(3, 4))
        with pytest.raises(RuntimeError, match="after batches have been pushed"):
            store.reserve(num_rows=10, num_features=4, device=torch.device("cpu"), dtype=torch.float32)

    @staticmethod
    def test_re_reserve_same_capacity_is_idempotent() -> None:
        """Reserving the same capacity twice must be safe."""
        store = EmbeddingStore()
        store.reserve(num_rows=10, num_features=4, device=torch.device("cpu"), dtype=torch.float32)
        store.reserve(num_rows=10, num_features=4, device=torch.device("cpu"), dtype=torch.float32)

        assert store.is_staged
        assert store._reserved_rows == 10  # noqa: SLF001

    @staticmethod
    def test_reserve_different_capacity_raises() -> None:
        """A second, different reservation must be refused."""
        store = EmbeddingStore()
        store.reserve(num_rows=10, num_features=4, device=torch.device("cpu"), dtype=torch.float32)
        with pytest.raises(RuntimeError, match="different capacity"):
            store.reserve(num_rows=20, num_features=4, device=torch.device("cpu"), dtype=torch.float32)

    @staticmethod
    def test_width_mismatch_raises() -> None:
        """A batch with a different embedding width must be refused."""
        store = EmbeddingStore()
        store.reserve(num_rows=10, num_features=4, device=torch.device("cpu"), dtype=torch.float32)
        with pytest.raises(RuntimeError, match="width changed"):
            store.push(torch.zeros(3, 5))

    @staticmethod
    def test_dtype_mismatch_raises() -> None:
        """A batch with a different dtype must be refused."""
        store = EmbeddingStore()
        store.reserve(num_rows=10, num_features=4, device=torch.device("cpu"), dtype=torch.float32)
        with pytest.raises(RuntimeError, match="dtype changed"):
            store.push(torch.zeros(3, 4, dtype=torch.float64))


class TestUpgrade:
    """Upgrading a started list-mode store to exact staging."""

    @staticmethod
    def test_upgrade_moves_history_and_stages_rest() -> None:
        """History pushed before the upgrade must keep its position."""
        store = EmbeddingStore()
        first = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        store.push(first)
        assert store.upgrade(num_rows=12, num_features=4, device=torch.device("cpu"), dtype=torch.float32)

        second = torch.arange(40, 76, dtype=torch.float32).reshape(9, 4)
        store.push(second)

        assert torch.equal(store.consolidate(), torch.vstack([first, second]))

    @staticmethod
    def test_upgrade_refuses_undersized() -> None:
        """An upgrade smaller than what is already staged must be refused."""
        store = EmbeddingStore()
        store.push(torch.zeros(3, 4))
        assert not store.upgrade(num_rows=2, num_features=4, device=torch.device("cpu"), dtype=torch.float32)
        assert store.is_list_mode

    @staticmethod
    def test_upgrade_refuses_wrong_geometry() -> None:
        """An upgrade with the wrong width must be refused."""
        store = EmbeddingStore()
        store.push(torch.zeros(3, 4))
        assert not store.upgrade(num_rows=12, num_features=5, device=torch.device("cpu"), dtype=torch.float32)

    @staticmethod
    def test_upgrade_empty_store_reserves() -> None:
        """Upgrading an empty store is equivalent to reserving."""
        store = EmbeddingStore()
        assert store.upgrade(num_rows=6, num_features=4, device=torch.device("cpu"), dtype=torch.float32)
        assert store.is_staged


class TestOverflow:
    """Correctness when a reservation is too small for what arrives."""

    @staticmethod
    def test_overflow_falls_back_to_list_mode() -> None:
        """More rows than reserved must degrade to the correct fallback, not crash."""
        store = EmbeddingStore()
        store.reserve(num_rows=5, num_features=4, device=torch.device("cpu"), dtype=torch.float32)
        first = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        second = torch.arange(40, 76, dtype=torch.float32).reshape(9, 4)
        store.push(first)
        store.push(second)  # 3 + 9 = 12 > 5 reserved

        consolidated = store.consolidate()

        assert torch.equal(consolidated, torch.vstack([first, second]))
        assert store.num_rows == 12
        assert not store.is_staged

    @staticmethod
    def test_overflow_drops_staging_storage() -> None:
        """The oversized reservation must be released on overflow."""
        store = EmbeddingStore()
        store.reserve(num_rows=5, num_features=4, device=torch.device("cpu"), dtype=torch.float32)
        store.push(torch.arange(12, dtype=torch.float32).reshape(3, 4))

        staging_before = store._staging  # noqa: SLF001
        store.push(torch.arange(40, 76, dtype=torch.float32).reshape(9, 4))

        assert staging_before is not None
        assert store._staging is None  # noqa: SLF001
