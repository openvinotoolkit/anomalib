# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from fastapi import status

from api.dependencies import get_media_service
from main import app
from pydantic_models import Media, MediaList
from pydantic_models.base import Pagination
from services import MediaService, ResourceNotFoundError
from services.exceptions import ResourceType


@pytest.fixture
def fxt_media(fxt_project):
    return Media(
        id=uuid4(),
        project_id=fxt_project.id,
        filename="test_image.jpg",
        size=64,
        is_anomalous=False,
        width=800,
        height=600,
    )


@pytest.fixture
def fxt_media_service() -> MagicMock:
    media_service = MagicMock(spec=MediaService)
    app.dependency_overrides[get_media_service] = lambda: media_service
    return media_service


def test_get_media_list_empty(fxt_client, fxt_media_service):
    project_id = uuid4()
    fxt_media_service.get_media_list.return_value = MediaList(
        media=[],
        pagination=Pagination(offset=0, limit=20, count=0, total=0),
    )

    response = fxt_client.get(f"/api/projects/{project_id}/images")
    assert response.status_code == status.HTTP_200_OK
    assert response.json()["media"] == []
    fxt_media_service.get_media_list.assert_called_once_with(project_id=project_id, limit=20, offset=0)


def test_get_media_list(fxt_client, fxt_media_service, fxt_media):
    project_id = uuid4()
    fxt_media_service.get_media_list.return_value = MediaList(
        media=[fxt_media],
        pagination=Pagination(offset=0, limit=20, count=1, total=1),
    )

    response = fxt_client.get(f"/api/projects/{project_id}/images")
    assert response.status_code == status.HTTP_200_OK
    assert len(response.json()["media"]) == 1
    fxt_media_service.get_media_list.assert_called_once_with(project_id=project_id, limit=20, offset=0)


def test_get_media_thumbnail_success(fxt_client, fxt_media_service, tmp_path):
    """Test successful thumbnail retrieval."""
    project_id = uuid4()
    media_id = uuid4()
    thumbnail_path = tmp_path / f"thumb_{media_id}.png"
    thumbnail_path.write_bytes(b"fake thumbnail content")

    fxt_media_service.get_thumbnail_file_path.return_value = str(thumbnail_path)

    response = fxt_client.get(f"/api/projects/{project_id}/images/{media_id}/thumbnail")

    assert response.status_code == status.HTTP_200_OK
    assert response.content == b"fake thumbnail content"
    fxt_media_service.get_thumbnail_file_path.assert_called_once_with(project_id=project_id, media_id=media_id)


def test_get_media_thumbnail_not_found(fxt_client, fxt_media_service):
    """Test thumbnail retrieval raises FileNotFoundError when media not found."""
    project_id = uuid4()
    media_id = uuid4()

    fxt_media_service.get_thumbnail_file_path.side_effect = FileNotFoundError("Media not found")

    response = fxt_client.get(f"/api/projects/{project_id}/images/{media_id}/thumbnail")

    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert response.json()["detail"] == f"Media with ID {media_id} not found"
    fxt_media_service.get_thumbnail_file_path.assert_called_once_with(project_id=project_id, media_id=media_id)


def test_delete_media_success(fxt_client, fxt_media_service, fxt_media):
    """Test successful media deletion."""
    project_id = fxt_media.project_id
    media_id = fxt_media.id

    fxt_media_service.delete_media.return_value = None

    response = fxt_client.delete(f"/api/projects/{project_id}/images/{media_id}")

    assert response.status_code == status.HTTP_204_NO_CONTENT
    assert response.content == b""
    fxt_media_service.delete_media.assert_called_once_with(media_id=media_id, project_id=project_id)


def test_delete_media_not_found(fxt_client, fxt_media_service):
    """Test media deletion when media not found."""
    project_id = uuid4()
    media_id = uuid4()

    fxt_media_service.delete_media.side_effect = ResourceNotFoundError(
        resource_type=ResourceType.MEDIA,
        resource_id=str(media_id),
    )

    response = fxt_client.delete(f"/api/projects/{project_id}/images/{media_id}")

    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert response.json()["detail"] == f"Media with ID {media_id} not found."
    fxt_media_service.delete_media.assert_called_once_with(media_id=media_id, project_id=project_id)


def test_delete_media_invalid_id(fxt_client, fxt_media_service):
    """Test media deletion with invalid media ID format."""
    project_id = uuid4()

    response = fxt_client.delete(f"/api/projects/{project_id}/images/invalid-id")

    assert response.status_code == status.HTTP_400_BAD_REQUEST
    fxt_media_service.delete_media.assert_not_called()
