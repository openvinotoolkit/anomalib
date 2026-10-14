# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from fastapi import UploadFile
from PIL import Image
from sqlalchemy.ext.asyncio.session import AsyncSession

from pydantic_models import (
    Job,
    JobList,
    JobStatus,
    JobType,
    Media,
    MediaList,
    Model,
    ModelList,
    PredictionLabel,
    PredictionResponse,
    Project,
    ProjectList,
)
from pydantic_models.base import Pagination
from pydantic_models.job import TrainJobPayload


@pytest.fixture
def fxt_project():
    """Fixture for a test project."""
    return Project(
        id=uuid4(),
        name="Test Project",
    )


@pytest.fixture
def fxt_job(fxt_project):
    """Fixture for a test job."""
    return Job(
        project_id=fxt_project.id,
        type=JobType.TRAINING,
        payload={"model_name": "padim"},
        message="Test job created",
        status=JobStatus.PENDING,
    )


@pytest.fixture
def fxt_job_payload(fxt_project):
    """Fixture for a test job payload."""
    return TrainJobPayload(
        project_id=fxt_project.id,
        model_name="padim",
        device=None,
    )


@pytest.fixture
def fxt_media(fxt_project):
    """Fixture for a test media."""
    return Media(
        id=uuid4(),
        project_id=fxt_project.id,
        filename="test_image.jpg",
        size=1024,
        is_anomalous=False,
        width=800,
        height=600,
    )


@pytest.fixture
def fxt_model(fxt_project):
    """Fixture for a test model."""
    id = uuid4()
    return Model(
        id=id,
        project_id=fxt_project.id,
        name=f"padim ({str(id).split('-')[0]})",
        architecture="padim",
        format="openvino",
        is_ready=True,
        export_path="/path/to/model",
        train_job_id=uuid4(),
        dataset_snapshot_id=uuid4(),
        size=0,
    )


@pytest.fixture
def fxt_prediction_response():
    """Fixture for a test prediction response."""
    return PredictionResponse(
        anomaly_map="base64_encoded_image",
        label=PredictionLabel.NORMAL,
        score=0.1,
    )


@pytest.fixture
def fxt_upload_file():
    """Fixture for a test upload file."""
    file_content = b"fake image content"
    upload_file = MagicMock(spec=UploadFile)
    upload_file.filename = "test_image.jpg"
    upload_file.size = len(file_content)
    upload_file.read = AsyncMock(return_value=file_content)
    return upload_file


@pytest.fixture
def fxt_db_session():
    """Fixture for a mock database session."""
    return AsyncMock(spec=AsyncSession)


@pytest.fixture
def fxt_project_list(fxt_project):
    """Fixture for a test project list."""
    return ProjectList(
        projects=[fxt_project],
        pagination=Pagination(offset=0, limit=20, count=1, total=1),
    )


@pytest.fixture
def fxt_job_list(fxt_job):
    """Fixture for a test job list."""
    return JobList(
        jobs=[fxt_job],
        pagination=Pagination(offset=0, limit=20, count=1, total=1),
    )


@pytest.fixture
def fxt_media_list(fxt_media):
    """Fixture for a test media list."""
    return MediaList(
        media=[fxt_media],
        pagination=Pagination(offset=0, limit=20, count=1, total=1),
    )


@pytest.fixture
def fxt_model_list(fxt_model):
    """Fixture for a test model list."""
    return ModelList(
        models=[fxt_model],
        pagination=Pagination(offset=0, limit=20, count=1, total=1),
    )


@pytest.fixture
def fxt_image_bytes():
    """Fixture for test image bytes."""
    # Create a valid image for testing
    img = Image.new("RGB", (800, 600), color="blue")
    with BytesIO() as buf:
        img.save(buf, format="JPEG")
        return buf.getvalue()


@pytest.fixture
def fxt_openvino_inferencer():
    """Fixture for a mock OpenVINO inferencer."""
    inferencer = MagicMock()
    prediction = MagicMock()
    prediction.anomaly_map = MagicMock()
    prediction.anomaly_map.squeeze.return_value = [[0.1, 0.2], [0.3, 0.4]]
    prediction.pred_label = MagicMock()
    prediction.pred_label.item.return_value = 0
    prediction.pred_score = MagicMock()
    prediction.pred_score.item.return_value = 0.1
    inferencer.predict.return_value = prediction
    return inferencer
