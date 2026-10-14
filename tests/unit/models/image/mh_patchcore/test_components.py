# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for MH-PatchCore statistical components."""

import numpy as np
import pytest
import torch
from sklearn.decomposition import IncrementalPCA

from anomalib.models.image.mh_patchcore import components
from anomalib.models.image.mh_patchcore.components import CovarianceWhitening, MergeReduceMemoryBank, StreamingPCA


@pytest.mark.parametrize("variance_ratio", [0.0, -0.1, 1.1, True])
def test_variance_ratio_validation(variance_ratio: float) -> None:
    """Variance retention must be numeric and within the supported interval."""
    with pytest.raises(ValueError, match="variance_ratio must be in"):
        StreamingPCA(variance_ratio=variance_ratio)


def test_streaming_fit_preserves_undersized_boundary_batches() -> None:
    """Undersized first and final batches should contribute exactly once."""
    generator = torch.Generator().manual_seed(7)
    embeddings = torch.randn(12, 4, generator=generator)
    batches = list(embeddings.split([2, 4, 5, 1]))
    pca = StreamingPCA(variance_ratio=0.9)

    for batch in batches:
        pca.update(batch)
    pca.finalize()

    reference = IncrementalPCA(n_components=None)
    reference.partial_fit(torch.cat(batches[:2]).numpy())
    reference.partial_fit(torch.cat(batches[2:]).numpy())
    cumulative_variance = np.cumsum(reference.explained_variance_ratio_)
    expected_components = int(np.flatnonzero(cumulative_variance >= 0.9)[0] + 1)
    expected = reference.transform(embeddings.numpy())[:, :expected_components].astype(np.float32)

    assert pca.num_components.item() == expected_components
    assert pca.components.shape == (expected_components, embeddings.shape[1])
    assert pca.components.dtype == torch.float32
    assert pca.mean.shape == (embeddings.shape[1],)
    assert pca.mean.dtype == torch.float64
    assert pca.projected_mean.shape == (expected_components,)
    assert pca.projected_mean.dtype == torch.float32
    torch.testing.assert_close(pca.mean, embeddings.double().mean(dim=0))
    torch.testing.assert_close(pca(embeddings), torch.from_numpy(expected), rtol=1e-5, atol=1e-6)
    assert torch.isfinite(pca(embeddings)).all()


def test_finalize_all_undersized_stream() -> None:
    """A complete stream smaller than its feature dimension should still fit."""
    values = np.random.default_rng(2).normal(size=(3, 8)).astype(np.float32)
    embeddings = torch.from_numpy(values)
    pca = StreamingPCA(variance_ratio=1.0)
    pca.update(embeddings[:1])
    pca.update(embeddings[1:])

    pca.finalize()

    assert pca.num_components.item() == len(embeddings)
    assert pca.components.shape == (len(embeddings), embeddings.shape[1])
    assert pca(embeddings).shape == (len(embeddings), len(embeddings))
    assert torch.isfinite(pca(embeddings)).all()


def test_persistent_state_restores_torch_transform() -> None:
    """A state-dict roundtrip should not require the fitting estimator."""
    embeddings = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    pca = StreamingPCA(variance_ratio=0.95)
    pca.update(embeddings[:4])
    pca.update(embeddings[4:])
    pca.finalize()
    expected = pca(embeddings)

    restored = StreamingPCA(variance_ratio=0.95)
    restored.load_state_dict(pca.state_dict())

    assert restored.is_fitted
    assert restored._estimator is None  # noqa: SLF001
    torch.testing.assert_close(restored(embeddings), expected)


def test_fitted_state_errors() -> None:
    """Invalid fitting-state transitions should fail clearly."""
    pca = StreamingPCA()
    embeddings = torch.randn(6, 4)

    with pytest.raises(RuntimeError, match="finalized before transform"):
        pca.transform(embeddings)
    with pytest.raises(RuntimeError, match="received no embedding batches"):
        pca.finalize()

    pca.update(embeddings)
    pca.finalize()
    with pytest.raises(RuntimeError, match="cannot be updated after finalization"):
        pca.update(embeddings)
    with pytest.raises(RuntimeError, match="already finalized"):
        pca.finalize()


@pytest.mark.parametrize("shrinkage", [-0.1, 1.1, float("nan"), True])
def test_covariance_shrinkage_validation(shrinkage: float) -> None:
    """Covariance shrinkage must be finite, numeric, and within its interval."""
    with pytest.raises(ValueError, match="shrinkage must be"):
        CovarianceWhitening(shrinkage=shrinkage)


def test_streaming_covariance_matches_batch_estimate() -> None:
    """Stream splitting should not change the unbiased covariance estimate."""
    embeddings = torch.tensor(
        [
            [0.5, 1.0, -1.0],
            [1.0, 2.0, 0.0],
            [2.0, 1.5, 1.0],
            [3.5, 4.0, 2.0],
            [5.0, 3.0, 4.0],
            [8.0, 6.0, 5.0],
        ],
        dtype=torch.float64,
    )
    single_batch = CovarianceWhitening()
    split_batches = CovarianceWhitening()

    single_batch.update(embeddings)
    for batch in embeddings.split([1, 2, 3]):
        split_batches.update(batch)

    expected = torch.cov(embeddings.T)
    torch.testing.assert_close(single_batch.covariance(), expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(split_batches.covariance(), expected, rtol=1e-12, atol=1e-12)

    single_batch.finalize()
    split_batches.finalize()
    torch.testing.assert_close(split_batches.mean, single_batch.mean, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(
        split_batches.whitening_matrix,
        single_batch.whitening_matrix,
        rtol=1e-12,
        atol=1e-12,
    )


def test_near_singular_whitening_preserves_mahalanobis_geometry() -> None:
    """Regularized whitening should remain finite for correlated features."""
    base = torch.arange(1, 9, dtype=torch.float64)
    embeddings = torch.stack((base, 2 * base, base + 1e-10 * base.square()), dim=1)
    whitening = CovarianceWhitening(shrinkage=0.0)
    whitening.update(embeddings[:3])
    whitening.update(embeddings[3:])
    covariance = whitening.covariance()
    regularized = components._regularize_covariance(covariance, whitening.shrinkage)  # noqa: SLF001

    whitening.finalize()
    transformed = whitening(embeddings)
    delta = embeddings[0] - embeddings[-1]
    expected_distance = delta @ torch.linalg.solve(regularized, delta)
    actual_distance = (transformed[0] - transformed[-1]).square().sum()

    assert whitening.mean.dtype == torch.float64
    assert whitening.whitening_matrix.dtype == torch.float64
    assert torch.isfinite(transformed).all()
    torch.testing.assert_close(actual_distance, expected_distance, rtol=1e-5, atol=1e-6)


def test_covariance_persistent_state_restores_transform() -> None:
    """A state-dict roundtrip should restore only finalized whitening state."""
    embeddings = torch.tensor(
        [[0.0, 1.0], [1.0, 3.0], [2.0, 2.0], [4.0, 5.0]],
        dtype=torch.float64,
    )
    whitening = CovarianceWhitening()
    whitening.update(embeddings)
    whitening.finalize()
    expected = whitening(embeddings)

    restored = CovarianceWhitening()
    restored.load_state_dict(whitening.state_dict())

    assert restored.is_fitted
    assert restored._running_mean is None  # noqa: SLF001
    assert restored._m2 is None  # noqa: SLF001
    torch.testing.assert_close(restored(embeddings), expected)


def test_covariance_state_errors() -> None:
    """Invalid covariance fitting transitions should fail clearly."""
    whitening = CovarianceWhitening()
    embeddings = torch.tensor([[0.0, 1.0], [1.0, 2.0]])

    with pytest.raises(RuntimeError, match="finalized before transform"):
        whitening.transform(embeddings)
    with pytest.raises(RuntimeError, match="received no embedding batches"):
        whitening.finalize()

    whitening.update(embeddings[:1])
    with pytest.raises(RuntimeError, match="at least two samples"):
        whitening.finalize()
    with pytest.raises(ValueError, match="previous batches have dimension"):
        whitening.update(torch.ones(1, 3))

    whitening.update(embeddings[1:])
    whitening.finalize()
    with pytest.raises(RuntimeError, match="cannot be updated after finalization"):
        whitening.update(embeddings)
    with pytest.raises(RuntimeError, match="already finalized"):
        whitening.finalize()


@pytest.mark.parametrize(
    "embeddings",
    [
        torch.empty(0, 2),
        torch.empty(2, 0),
        torch.ones(2),
        torch.tensor([[0.0, float("inf")]]),
    ],
)
def test_covariance_input_validation(embeddings: torch.Tensor) -> None:
    """Malformed covariance batches should be rejected without changing state."""
    whitening = CovarianceWhitening()

    with pytest.raises(ValueError, match="embeddings must"):
        whitening.update(embeddings)


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("memory_bank_size", 0),
        ("memory_bank_size", -1),
        ("memory_bank_size", True),
        ("memory_bank_size", 1.0),
        ("local_coreset_size", 0),
        ("local_coreset_size", -1),
        ("local_coreset_size", True),
        ("local_coreset_size", 1.0),
    ],
)
def test_merge_reduce_budget_validation(argument: str, value: object) -> None:
    """Merge-reduce budgets must be positive integers and reject booleans."""
    with pytest.raises(ValueError, match=rf"{argument} must be a positive integer"):
        MergeReduceMemoryBank(**{argument: value})


def test_merge_reduce_preserves_order_when_budget_covers_stream() -> None:
    """A bank covering every candidate should retain the encountered order."""
    embeddings = torch.tensor([[3.0, 0.0], [1.0, 2.0], [2.0, 1.0]])
    memory_bank = MergeReduceMemoryBank(memory_bank_size=8, local_coreset_size=8)

    memory_bank.update(embeddings)
    memory_bank.finalize()

    assert memory_bank.bank.dtype == torch.float32
    torch.testing.assert_close(memory_bank.bank, embeddings)


def test_merge_reduce_levels_are_deterministic_and_persistent() -> None:
    """Binary merges should preserve canonical ordering and checkpoint state."""
    batches = [
        torch.tensor(values, dtype=torch.float64).reshape(-1, 1)
        for values in ([0, 1, 4], [10, 11, 15], [20, 22, 25], [30, 34, 35], [40, 41, 47], [50, 56, 57], [60, 61, 69])
    ]
    memory_bank = MergeReduceMemoryBank(memory_bank_size=3, local_coreset_size=2)

    for batch in batches:
        memory_bank.update(batch)

    assert sorted(memory_bank._levels) == [0, 1, 2]  # noqa: SLF001
    torch.testing.assert_close(
        memory_bank._levels[0].flatten(),  # noqa: SLF001
        torch.tensor([69.0, 60.0], dtype=torch.float64),
    )
    torch.testing.assert_close(
        memory_bank._levels[1].flatten(),  # noqa: SLF001
        torch.tensor([40.0, 57.0], dtype=torch.float64),
    )
    torch.testing.assert_close(
        memory_bank._levels[2].flatten(),  # noqa: SLF001
        torch.tensor([0.0, 35.0], dtype=torch.float64),
    )

    memory_bank.finalize()

    expected = torch.tensor([[0.0], [69.0], [35.0]])
    torch.testing.assert_close(memory_bank.bank, expected)
    source = torch.cat(batches).to(dtype=torch.float32)
    assert all(bool((source == vector).all(dim=1).any()) for vector in memory_bank.bank)

    repeated = MergeReduceMemoryBank(memory_bank_size=3, local_coreset_size=2)
    for batch in batches:
        repeated.update(batch)
    repeated.finalize()
    torch.testing.assert_close(repeated.bank, expected)

    restored = MergeReduceMemoryBank(memory_bank_size=3, local_coreset_size=2)
    restored.load_state_dict(memory_bank.state_dict())
    assert restored.is_fitted
    torch.testing.assert_close(restored.bank, expected)


def test_merge_reduce_ties_and_state_errors() -> None:
    """Ties should select the first index and invalid state transitions should fail."""
    memory_bank = MergeReduceMemoryBank(memory_bank_size=2, local_coreset_size=4)
    with pytest.raises(RuntimeError, match="received no embedding batches"):
        memory_bank.finalize()
    with pytest.raises(ValueError, match="non-empty two-dimensional"):
        memory_bank.update(torch.empty(0, 2))
    with pytest.raises(ValueError, match="only finite values"):
        memory_bank.update(torch.tensor([[0.0, float("inf")]]))

    embeddings = torch.tensor([[-2.0, 0.0], [2.0, 0.0], [0.0, -1.0], [0.0, 1.0]])
    memory_bank.update(embeddings)
    with pytest.raises(ValueError, match="previous batches have dimension"):
        memory_bank.update(torch.ones(1, 3))
    memory_bank.finalize()

    torch.testing.assert_close(memory_bank.bank, embeddings[:2])
    with pytest.raises(RuntimeError, match="cannot be updated after finalization"):
        memory_bank.update(embeddings)
    with pytest.raises(RuntimeError, match="already finalized"):
        memory_bank.finalize()
