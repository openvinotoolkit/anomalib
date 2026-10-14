# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for anomalib.utils.serialization."""

from enum import Enum
from pathlib import Path

import numpy as np
import torch
from torch.serialization import get_unsafe_globals_in_checkpoint

from anomalib import PrecisionType
from anomalib.utils.serialization import ANOMALIB_SAFE_GLOBALS, NUMPY_SAFE_GLOBALS, anomalib_safe_globals


class _ExtraEnum(Enum):
    """Local enum used only to exercise the ``extra=`` allowlist path."""

    VALUE = "value"


def test_precision_type_is_allowlisted() -> None:
    """PrecisionType is the first-party enum persisted in Patchcore hyperparameters."""
    assert PrecisionType in ANOMALIB_SAFE_GLOBALS


def test_numpy_safe_globals_are_non_empty() -> None:
    """Shared numpy leaf types are allowlisted for scheduler/optimizer state."""
    assert len(NUMPY_SAFE_GLOBALS) > 0
    assert np.ndarray in NUMPY_SAFE_GLOBALS
    assert np.dtype in NUMPY_SAFE_GLOBALS
    assert np.dtypes.Float64DType in NUMPY_SAFE_GLOBALS


def test_numpy_safe_globals_do_not_import_private_numpy_modules() -> None:
    """NumPy reducers are obtained through public objects."""
    reducer_globals = [entry for entry in NUMPY_SAFE_GLOBALS if isinstance(entry, tuple)]

    assert reducer_globals
    assert {entry[0].__name__ for entry in reducer_globals} == {"scalar", "_reconstruct"}


def test_safe_globals_context_allows_precision_type(tmp_path: Path) -> None:
    """weights_only load of PrecisionType succeeds only inside the allowlist context."""
    path = tmp_path / "hparams.pt"
    # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch
    torch.save({"precision": PrecisionType.FLOAT32}, path)

    assert "anomalib.PrecisionType" in get_unsafe_globals_in_checkpoint(path)

    with anomalib_safe_globals():
        # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch
        loaded = torch.load(path, weights_only=True)

    assert loaded["precision"] == PrecisionType.FLOAT32


def test_safe_globals_context_allows_numpy_scheduler_leaves(tmp_path: Path) -> None:
    """weights_only load succeeds for numpy scalars/arrays used in LR schedules."""
    path = tmp_path / "scheduler.pt"
    # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch
    torch.save(
        {
            "lr": np.float64(1e-3),
            "schedule": np.linspace(0.0, 1.0, 8),
        },
        path,
    )

    unsafe = get_unsafe_globals_in_checkpoint(path)
    assert any("numpy" in name for name in unsafe)

    with anomalib_safe_globals():
        # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch
        loaded = torch.load(path, weights_only=True)

    assert loaded["lr"] == np.float64(1e-3)
    assert np.allclose(loaded["schedule"], np.linspace(0.0, 1.0, 8))


def test_safe_globals_extra_merges_with_shared_allowlist(tmp_path: Path) -> None:
    """``extra`` types are allowlisted alongside shared ``ANOMALIB_SAFE_GLOBALS``."""
    path = tmp_path / "hparams.pt"
    # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch
    torch.save({"precision": PrecisionType.FLOAT32, "extra": _ExtraEnum.VALUE}, path)

    with anomalib_safe_globals(extra=[_ExtraEnum]):
        # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch
        loaded = torch.load(path, weights_only=True)

    assert loaded["precision"] == PrecisionType.FLOAT32
    assert loaded["extra"] == _ExtraEnum.VALUE
