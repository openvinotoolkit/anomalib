# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import pytest

from pydantic_models.sink import FolderSinkConfig, SinkType
from services.dispatch_service import DispatchService


@pytest.fixture
def folder_sink_config(tmp_path: Path) -> FolderSinkConfig:
    return FolderSinkConfig(
        id=uuid4(),
        project_id=uuid4(),
        name="Local Folder",
        sink_type=SinkType.FOLDER,
        folder_path=str(tmp_path),
        output_formats=[],
        rate_limit=0.2,
    )


def test_get_destinations_filters_failed_dispatchers(folder_sink_config: FolderSinkConfig) -> None:
    """Test that a dispatcher failing initialization is skipped and logged."""
    # We patch the specific registry entry for FOLDER to raise an exception.

    def failing_factory(config):
        raise RuntimeError("Initialization error")

    with (
        patch.dict(DispatchService._dispatcher_registry, {SinkType.FOLDER: failing_factory}),
        patch("services.dispatch_service.logger") as mock_logger,
    ):
        destinations = DispatchService.get_destinations([folder_sink_config])

    assert destinations == []

    mock_logger.opt.assert_called_once_with(exception=True)
    mock_logger.opt.return_value.warning.assert_called_once_with(
        f"Failed to initialize dispatcher for sink type: {SinkType.FOLDER}"
    )


def test_get_destinations_filters_unrecognized_sink() -> None:
    """Test that an unrecognized sink type is filtered out and logs a warning without an exception."""

    # Create a mock config with a dummy sink type.
    class DummySinkConfig:
        sink_type = "unrecognized_type"

    dummy_config = DummySinkConfig()

    with patch("services.dispatch_service.logger") as mock_logger:
        # get_destinations types expect Sequence[Sink] but for test we bypass with the dummy.
        destinations = DispatchService.get_destinations([dummy_config])  # type: ignore

        assert len(destinations) == 0
        mock_logger.warning.assert_called_once_with("Unrecognized sink type: unrecognized_type")


def test_get_destinations_returns_valid_dispatcher(folder_sink_config: FolderSinkConfig) -> None:
    """Test that a valid config returns a dispatcher correctly."""
    destinations = DispatchService.get_destinations([folder_sink_config])

    assert len(destinations) == 1
    # We expect a FolderDispatcher instance. Check by class name.
    assert type(destinations[0]).__name__ == "FolderDispatcher"
