# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from fastapi import status

from api.dependencies import get_model_service
from main import app
from pydantic_models import Model, ModelList
from pydantic_models.base import Pagination
from services import ModelService, ResourceNotFoundError
from services.exceptions import ResourceType


@pytest.fixture
def fxt_model(fxt_project):
    id = uuid4()
    return Model(
        id=id,
        name=f"test_model ({str(id).split('-')[0]})",
        architecture="padim",
        project_id=fxt_project.id,
        export_path="/path/to/model",
        train_job_id=uuid4(),
        dataset_snapshot_id=uuid4(),
        size=1024,
    )


@pytest.fixture
def fxt_model_service() -> MagicMock:
    model_service = MagicMock(spec=ModelService)
    app.dependency_overrides[get_model_service] = lambda: model_service
    return model_service


def test_get_models_empty(fxt_client, fxt_model_service, fxt_project):
    project_id = fxt_project.id
    fxt_model_service.get_model_list.return_value = ModelList(
        models=[],
        pagination=Pagination(offset=0, limit=20, count=0, total=0),
    )

    response = fxt_client.get(f"/api/projects/{project_id}/models")
    assert response.status_code == status.HTTP_200_OK
    assert response.json()["models"] == []
    fxt_model_service.get_model_list.assert_called_once_with(project_id=project_id, limit=20, offset=0)


def test_get_models(fxt_client, fxt_model_service, fxt_model, fxt_project):
    project_id = fxt_project.id
    fxt_model_service.get_model_list.return_value = ModelList(
        models=[fxt_model],
        pagination=Pagination(offset=0, limit=20, count=1, total=1),
    )

    response = fxt_client.get(f"/api/projects/{project_id}/models")
    assert response.status_code == status.HTTP_200_OK
    assert len(response.json()["models"]) == 1
    assert response.json()["models"][0]["size"] == 1024
    fxt_model_service.get_model_list.assert_called_once_with(project_id=project_id, limit=20, offset=0)


def test_delete_model_success(fxt_client, fxt_model_service, fxt_project):
    project_id = fxt_project.id
    model_id = uuid4()

    response = fxt_client.delete(f"/api/projects/{project_id}/models/{model_id}")

    assert response.status_code == status.HTTP_204_NO_CONTENT
    fxt_model_service.delete_model.assert_called_once_with(project_id=project_id, model_id=model_id)


def test_delete_model_not_found(fxt_client, fxt_model_service, fxt_project):
    project_id = fxt_project.id
    model_id = uuid4()
    fxt_model_service.delete_model.side_effect = ResourceNotFoundError(ResourceType.MODEL, str(model_id))

    response = fxt_client.delete(f"/api/projects/{project_id}/models/{model_id}")

    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert "not found" in response.json()["detail"].lower()
