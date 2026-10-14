# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Lazy dummy-dataset fixtures for the test suite."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from anomalib.data import ImageDataFormat, VideoDataFormat
from tests.helpers.data import DummyImageDatasetGenerator, DummyVideoDatasetGenerator

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

# Keep shared fixture datasets small; generator API defaults remain unchanged.
_DEFAULT_IMAGE_SHAPE = (64, 64)
_DEPTH_IMAGE_SHAPE = (256, 256)
_DEPTH_IMAGE_FORMATS = {"adam_3d", "mvtec_3d"}
_DEFAULT_NUM_TRAIN = 5
_DEFAULT_NUM_TEST = 5
_DEFAULT_NUM_FRAMES = 8
_DEFAULT_FRAME_SHAPE = (64, 64)

_IMAGE_FORMATS = {fmt.value for fmt in ImageDataFormat}
_VIDEO_FORMATS = {fmt.value for fmt in VideoDataFormat}


@pytest.fixture(scope="session")
def dataset_path(project_path: Path) -> Path:
    """Return the dummy-datasets root without generating any formats."""
    path = project_path / "datasets"
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_make_dummy_dataset(dataset_path: Path) -> Callable[[str], Path]:
    """Build a factory that generates a single dummy format on first use."""
    created: set[str] = set()

    def _make(data_format: str) -> Path:
        key = data_format
        # Regenerate if a prior wipe left the format marked created but missing.
        if key in created and (dataset_path / key).exists():
            return dataset_path
        created.discard(key)

        if key in _VIDEO_FORMATS or key in {"ucsdped", "avenue", "shanghaitech"}:
            DummyVideoDatasetGenerator(
                data_format=key,
                root=dataset_path,
                num_train=_DEFAULT_NUM_TRAIN,
                num_test=_DEFAULT_NUM_TEST,
                num_frames=_DEFAULT_NUM_FRAMES,
                frame_shape=_DEFAULT_FRAME_SHAPE,
            ).generate_dataset()
        elif key == "realiad" or key in _IMAGE_FORMATS:
            # Folder/tabular tests build from MVTec AD or their own layouts.
            if key.startswith(("folder", "tabular")):
                msg = f"Format {key!r} is not auto-generated; use mvtecad/folder helpers in the test."
                raise ValueError(msg)
            DummyImageDatasetGenerator(
                data_format=key,
                root=dataset_path,
                num_train=_DEFAULT_NUM_TRAIN,
                num_test=_DEFAULT_NUM_TEST,
                image_shape=_DEPTH_IMAGE_SHAPE if key in _DEPTH_IMAGE_FORMATS else _DEFAULT_IMAGE_SHAPE,
            ).generate_dataset()
        else:
            msg = f"Unknown dummy data format: {key!r}"
            raise ValueError(msg)

        created.add(key)
        return dataset_path

    return _make


@pytest.fixture(scope="session")
def make_dummy_dataset(dataset_path: Path) -> Callable[[str], Path]:
    """Return a factory that generates a single dummy format on first use."""
    return build_make_dummy_dataset(dataset_path)


@pytest.fixture(scope="session")
def mvtecad_path(make_dummy_dataset: Callable[[str], Path], dataset_path: Path) -> Path:
    """Return the MVTec AD dummy category root."""
    make_dummy_dataset("mvtecad")
    return dataset_path / "mvtecad"
