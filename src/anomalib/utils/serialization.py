# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Helpers for loading anomalib checkpoints with ``weights_only=True``."""

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any, Protocol

import numpy as np
import torch

from anomalib import PrecisionType


class _PickleReducible(Protocol):
    """This class is used only for typing."""

    def __reduce__(self) -> tuple[Callable[..., object], tuple[object, ...]]: ...


def _numpy_reduce_function(value: _PickleReducible) -> Callable[..., object]:
    """Get NumPy's pickle reducer without importing private NumPy modules."""
    return value.__reduce__()[0]


# NumPy stores these reducer globals under version-dependent module paths. Get
# the callable from public NumPy objects, then register both known pickle names.
_numpy_scalar = _numpy_reduce_function(np.float64(0))
_numpy_reconstruct = _numpy_reduce_function(np.empty(0))

# First-party types that appear in checkpoint ``hyper_parameters``. Keep this to
# shared anomalib enums. Model-local types go through ``extra=`` /
# ``AnomalibModule.checkpoint_safe_globals()``.
ANOMALIB_SAFE_GLOBALS: list[Any] = [PrecisionType]

# Numpy leaf types that land in Lightning optimizer / LR-scheduler state
# (e.g. Dinomaly ``WarmCosineScheduler.schedule``). Both pickle module paths are
# allowlisted so checkpoints saved under either name can load.
NUMPY_SAFE_GLOBALS: list[Any] = [
    (_numpy_scalar, "numpy.core.multiarray.scalar"),
    (_numpy_scalar, "numpy._core.multiarray.scalar"),
    (_numpy_reconstruct, "numpy.core.multiarray._reconstruct"),
    (_numpy_reconstruct, "numpy._core.multiarray._reconstruct"),
    np.dtype,
    np.ndarray,
    # Instantiated during ndarray/dtype unpickling even when not listed by
    # get_unsafe_globals_in_checkpoint (NumPy 2.x).
    np.dtypes.Float64DType,
]


@contextmanager
def anomalib_safe_globals(extra: Sequence[Any] | None = None) -> Iterator[None]:
    """Temporarily allowlist anomalib types for ``torch.load(..., weights_only=True)``.

    Merges shared anomalib enums, numpy leaf types from optimizer/scheduler
    state, and optional model-specific types from ``extra``.

    Use this at both load seams:

    - ``AnomalibModule.load_from_checkpoint`` (direct API / ``Engine.export``)
    - ``AnomalibCheckpointIO.load_checkpoint`` (Trainer ``ckpt_path`` restores)
    """
    allowlist = [*ANOMALIB_SAFE_GLOBALS, *NUMPY_SAFE_GLOBALS, *(extra or ())]
    with torch.serialization.safe_globals(allowlist):
        yield
