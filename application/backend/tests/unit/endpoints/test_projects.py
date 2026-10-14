# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from fastapi import status

from api.dependencies import get_job_service, get_pipeline_service, get_project_service
from main import app
from pydantic_models import Pipeline, ProjectList
from pydantic_models.base import Pagination
from services import JobService, PipelineService, ProjectService


@pytest.fixture
def fxt_project_service() -> MagicMock:
    project_service = MagicMock(spec=ProjectService)
    app.dependency_overrides[get_project_service] = lambda: project_service
    return project_service


@pytest.fixture
def fxt_job_service() -> MagicMock:
    job_service = MagicMock(spec=JobService)
    app.dependency_overrides[get_job_service] = lambda: job_service
    return job_service


@pytest.fixture
def fxt_pipeline_service() -> MagicMock:
    pipeline_service = MagicMock(spec=PipelineService)
    app.dependency_overrides[get_pipeline_service] = lambda: pipeline_service
    return pipeline_service


def test_get_projects(fxt_client, fxt_project_service, fxt_project):
    fxt_project_service.get_project_list.return_value = ProjectList(
        projects=[fxt_project],
        pagination=Pagination(offset=0, limit=20, count=1, total=1),
    )

    response = fxt_client.get("/api/projects")
    assert response.status_code == status.HTTP_200_OK
    assert len(response.json()["projects"]) == 1
    fxt_project_service.get_project_list.assert_called_once_with(limit=20, offset=0)


def test_get_project_not_found(fxt_client, fxt_project_service):
    project_id = uuid4()
    fxt_project_service.get_project_by_id.return_value = None

    response = fxt_client.get(f"/api/projects/{project_id}")
    assert response.status_code == status.HTTP_404_NOT_FOUND
    fxt_project_service.get_project_by_id.assert_called_once_with(project_id)


def test_delete_project(fxt_client, fxt_project_service, fxt_job_service, fxt_pipeline_service, fxt_project):
    fxt_project_service.get_project_by_id.return_value = fxt_project
    fxt_pipeline_service.get_active_pipeline.return_value = None
    fxt_job_service.has_running_jobs.return_value = False

    response = fxt_client.delete(f"/api/projects/{fxt_project.id}")
    assert response.status_code == status.HTTP_200_OK
    fxt_project_service.delete_project.assert_called_once_with(fxt_project.id)


def test_delete_project_not_found(fxt_client, fxt_project_service, fxt_job_service, fxt_pipeline_service):
    project_id = uuid4()
    fxt_project_service.get_project_by_id.return_value = None

    response = fxt_client.delete(f"/api/projects/{project_id}")
    assert response.status_code == status.HTTP_404_NOT_FOUND
    fxt_project_service.delete_project.assert_not_called()


def test_delete_project_active_pipeline(
    fxt_client,
    fxt_project_service,
    fxt_job_service,
    fxt_pipeline_service,
    fxt_project,
):
    fxt_project_service.get_project_by_id.return_value = fxt_project
    active_pipeline = MagicMock(spec=Pipeline)
    active_pipeline.project_id = fxt_project.id
    active_pipeline.is_running = True
    fxt_pipeline_service.get_active_pipeline.return_value = active_pipeline

    response = fxt_client.delete(f"/api/projects/{fxt_project.id}")
    assert response.status_code == status.HTTP_409_CONFLICT
    assert "active pipeline" in response.json()["detail"]
    fxt_project_service.delete_project.assert_not_called()


def test_delete_project_running_jobs(
    fxt_client,
    fxt_project_service,
    fxt_job_service,
    fxt_pipeline_service,
    fxt_project,
):
    fxt_project_service.get_project_by_id.return_value = fxt_project
    fxt_pipeline_service.get_active_pipeline.return_value = None
    fxt_job_service.has_running_jobs.return_value = True

    response = fxt_client.delete(f"/api/projects/{fxt_project.id}")
    assert response.status_code == status.HTTP_409_CONFLICT
    assert "running jobs" in response.json()["detail"]
    fxt_project_service.delete_project.assert_not_called()


def test_delete_project_failure(fxt_client, fxt_project_service, fxt_job_service, fxt_pipeline_service, fxt_project):
    fxt_project_service.get_project_by_id.return_value = fxt_project
    fxt_pipeline_service.get_active_pipeline.return_value = None
    fxt_job_service.has_running_jobs.return_value = False
    fxt_project_service.delete_project.side_effect = RuntimeError("Deletion failed")

    response = fxt_client.delete(f"/api/projects/{fxt_project.id}")
    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert "Deletion failed" in response.json()["detail"]
    fxt_project_service.delete_project.assert_called_once_with(fxt_project.id)
