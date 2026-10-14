# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import io
from enum import StrEnum
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
import yaml
from fastapi import status

from api.dependencies import get_configuration_service
from main import app
from pydantic_models.base import Pagination
from pydantic_models.sink import FolderSinkConfig, MqttSinkConfig, OutputFormat, SinkList, SinkType
from pydantic_models.source import SourceList, SourceType, UsbCameraSourceConfig, VideoFileSourceConfig
from services import ConfigurationService, ResourceAlreadyExistsError, ResourceInUseError, ResourceNotFoundError
from services.exceptions import ResourceType


class ConfigApiPath(StrEnum):
    SINKS = "sinks"
    SOURCES = "sources"


@pytest.fixture
def fxt_folder_sink(fxt_project) -> FolderSinkConfig:
    return FolderSinkConfig(
        id=uuid4(),
        project_id=fxt_project.id,
        sink_type=SinkType.FOLDER,
        name="Test Folder Sink",
        rate_limit=0.1,
        output_formats=[OutputFormat.PREDICTIONS],
        folder_path="/test/path",
    )


@pytest.fixture
def fxt_usb_camera_source(fxt_project) -> UsbCameraSourceConfig:
    return UsbCameraSourceConfig(
        id=uuid4(),
        project_id=fxt_project.id,
        source_type=SourceType.USB_CAMERA,
        name="Test USB Camera Source",
        device_id=1,
    )


@pytest.fixture
def fxt_video_source(fxt_project) -> VideoFileSourceConfig:
    return VideoFileSourceConfig(
        id=uuid4(),
        project_id=fxt_project.id,
        source_type=SourceType.VIDEO_FILE,
        name="Test Folder Source",
        video_path="/test/video/path.mp4",
    )


@pytest.fixture
def fxt_mqtt_sink(fxt_project) -> MqttSinkConfig:
    return MqttSinkConfig(
        id=uuid4(),
        project_id=fxt_project.id,
        sink_type=SinkType.MQTT,
        name="Test MQTT Sink",
        rate_limit=0.2,
        output_formats=[OutputFormat.IMAGE_WITH_PREDICTIONS],
        broker_host="localhost",
        broker_port=1883,
        topic="test_topic",
    )


@pytest.fixture
def fxt_config_service() -> MagicMock:
    config_service = MagicMock(spec=ConfigurationService)
    app.dependency_overrides[get_configuration_service] = lambda: config_service
    return config_service


class TestSourceAndSinkEndpoints:
    @pytest.mark.parametrize(
        "fixture_name, api_path, create_method",
        [
            ("fxt_usb_camera_source", ConfigApiPath.SOURCES, "create_source"),
            ("fxt_folder_sink", ConfigApiPath.SINKS, "create_sink"),
        ],
    )
    def test_create_config_success(
        self,
        fixture_name,
        api_path,
        create_method,
        fxt_config_service,
        fxt_client,
        fxt_project,
        request,
    ):
        fxt_config = request.getfixturevalue(fixture_name)
        config_id = str(fxt_config.id)
        getattr(fxt_config_service, create_method).return_value = fxt_config

        response = fxt_client.post(
            f"/api/projects/{fxt_project.id}/{api_path}",
            json=fxt_config.model_dump(exclude={"id", "project_id"}),
        )

        assert response.status_code == status.HTTP_201_CREATED
        assert response.json()["id"] == config_id
        assert response.json()["name"] == fxt_config.name
        getattr(fxt_config_service, create_method).assert_called_once()

    @pytest.mark.parametrize(
        "api_path, create_method",
        [
            (ConfigApiPath.SOURCES, "create_source"),
            (ConfigApiPath.SINKS, "create_sink"),
        ],
    )
    def test_create_sink_validation_error(self, api_path, create_method, fxt_config_service, fxt_client, fxt_project):
        response = fxt_client.post(f"/api/projects/{fxt_project.id}/{api_path}", json={"name": ""})

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        getattr(fxt_config_service, create_method).assert_not_called()

    @pytest.mark.parametrize(
        "resource_type, api_path, fixture_name, create_method",
        [
            (ResourceType.SOURCE, ConfigApiPath.SOURCES, "fxt_usb_camera_source", "create_source"),
            (ResourceType.SINK, ConfigApiPath.SINKS, "fxt_folder_sink", "create_sink"),
        ],
    )
    def test_create_config_exists(
        self,
        resource_type,
        api_path,
        fixture_name,
        create_method,
        fxt_config_service,
        fxt_client,
        fxt_project,
        request,
    ):
        fxt_config = request.getfixturevalue(fixture_name)
        getattr(fxt_config_service, create_method).side_effect = ResourceAlreadyExistsError(
            resource_type=resource_type,
            resource_name="New Config",
        )
        response = fxt_client.post(
            f"/api/projects/{fxt_project.id}/{api_path}",
            json=fxt_config.model_dump(exclude={"id", "project_id"}),
        )

        assert response.status_code == status.HTTP_409_CONFLICT
        getattr(fxt_config_service, create_method).assert_called_once()

    @pytest.mark.parametrize(
        "fixtures, api_path, list_method",
        [
            (["fxt_usb_camera_source", "fxt_video_source"], ConfigApiPath.SOURCES, "list_sources"),
            (["fxt_folder_sink", "fxt_mqtt_sink"], ConfigApiPath.SINKS, "list_sinks"),
        ],
    )
    def test_list_configs(self, fixtures, api_path, list_method, fxt_config_service, fxt_client, fxt_project, request):
        fxt_configs = [request.getfixturevalue(fixture) for fixture in fixtures]
        if api_path == ConfigApiPath.SOURCES:
            list_response = SourceList(
                sources=fxt_configs,
                pagination=Pagination(offset=0, limit=20, count=2, total=2),
            )
        else:
            list_response = SinkList(
                sinks=fxt_configs,
                pagination=Pagination(offset=0, limit=20, count=2, total=2),
            )
        getattr(fxt_config_service, list_method).return_value = list_response

        response = fxt_client.get(f"/api/projects/{fxt_project.id}/{api_path}")

        assert response.status_code == status.HTTP_200_OK
        if api_path == ConfigApiPath.SOURCES:
            assert len(response.json()["sources"]) == 2
        else:
            assert len(response.json()["sinks"]) == 2
        getattr(fxt_config_service, list_method).assert_called_once_with(project_id=fxt_project.id, limit=20, offset=0)

    @pytest.mark.parametrize(
        "fixture_name, api_path, get_method",
        [
            ("fxt_usb_camera_source", ConfigApiPath.SOURCES, "get_source_by_id"),
            ("fxt_folder_sink", ConfigApiPath.SINKS, "get_sink_by_id"),
        ],
    )
    def test_get_config_success(
        self,
        fixture_name,
        api_path,
        get_method,
        fxt_config_service,
        fxt_client,
        fxt_project,
        request,
    ):
        fxt_config = request.getfixturevalue(fixture_name)
        config_id = str(fxt_config.id)
        getattr(fxt_config_service, get_method).return_value = fxt_config

        response = fxt_client.get(f"/api/projects/{fxt_project.id}/{api_path}/{config_id}")

        assert response.status_code == status.HTTP_200_OK
        assert response.json()["id"] == config_id
        getattr(fxt_config_service, get_method).assert_called_once_with(fxt_config.id, fxt_project.id)

    @pytest.mark.parametrize(
        "resource_type, api_path, get_method",
        [
            (ResourceType.SOURCE, ConfigApiPath.SOURCES, "get_source_by_id"),
            (ResourceType.SINK, ConfigApiPath.SINKS, "get_sink_by_id"),
        ],
    )
    def test_get_config_not_found(
        self,
        resource_type,
        api_path,
        get_method,
        fxt_config_service,
        fxt_client,
        fxt_project,
    ):
        config_id = uuid4()
        getattr(fxt_config_service, get_method).side_effect = ResourceNotFoundError(resource_type, str(config_id))

        response = fxt_client.get(f"/api/projects/{fxt_project.id}/{api_path}/{config_id!s}")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        getattr(fxt_config_service, get_method).assert_called_once_with(config_id, fxt_project.id)

    @pytest.mark.parametrize("api_path", [ConfigApiPath.SOURCES, ConfigApiPath.SINKS])
    def test_get_config_invalid_uuid(self, api_path, fxt_client, fxt_project):
        response = fxt_client.get(f"/api/projects/{fxt_project.id}/{api_path}/invalid-uuid")
        assert response.status_code == status.HTTP_400_BAD_REQUEST

    @pytest.mark.parametrize(
        "fixture_name, api_path, update_method, update_data",
        [
            ("fxt_usb_camera_source", ConfigApiPath.SOURCES, "update_source", {"device_id": 5}),
            ("fxt_folder_sink", ConfigApiPath.SINKS, "update_sink", {"folder_path": "/new/path"}),
        ],
    )
    def test_update_sink_success(
        self,
        fixture_name,
        api_path,
        update_method,
        update_data,
        fxt_config_service,
        fxt_client,
        fxt_project,
        request,
    ):
        fxt_config = request.getfixturevalue(fixture_name)
        config_id = str(fxt_config.id)
        updated_config = fxt_config.model_copy(update=update_data)
        getattr(fxt_config_service, update_method).return_value = updated_config

        response = fxt_client.patch(f"/api/projects/{fxt_project.id}/{api_path}/{config_id}", json=update_data)

        assert response.status_code == status.HTTP_200_OK
        key, value = next(iter(update_data.items()))
        assert response.json()[key] == value
        getattr(fxt_config_service, update_method).assert_called_once_with(fxt_config.id, fxt_project.id, update_data)

    @pytest.mark.parametrize(
        "api_path, update_method, resource_type",
        [
            (ConfigApiPath.SOURCES, "update_source", ResourceType.SOURCE),
            (ConfigApiPath.SINKS, "update_sink", ResourceType.SINK),
        ],
    )
    def test_update_config_not_found(
        self,
        api_path,
        update_method,
        resource_type,
        fxt_config_service,
        fxt_client,
        fxt_project,
    ):
        config_id = str(uuid4())
        getattr(fxt_config_service, update_method).side_effect = ResourceNotFoundError(resource_type, config_id)

        response = fxt_client.patch(f"/api/projects/{fxt_project.id}/{api_path}/{config_id}", json={"name": "Updated"})

        assert response.status_code == status.HTTP_404_NOT_FOUND

    @pytest.mark.parametrize(
        "fixture_name, api_path, update_method, update_data",
        [
            ("fxt_usb_camera_source", ConfigApiPath.SOURCES, "update_source", {"source_type": "folder"}),
            ("fxt_folder_sink", ConfigApiPath.SINKS, "update_sink", {"sink_type": "mqtt"}),
        ],
    )
    def test_update_config_type_forbidden(
        self,
        fixture_name,
        api_path,
        update_method,
        update_data,
        fxt_config_service,
        fxt_client,
        fxt_project,
        request,
    ):
        fxt_config = request.getfixturevalue(fixture_name)
        config_id = str(fxt_config.id)

        response = fxt_client.patch(f"/api/projects/{fxt_project.id}/{api_path}/{config_id}", json=update_data)

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "_type" in response.json()["detail"]
        getattr(fxt_config_service, update_method).assert_not_called()

    @pytest.mark.parametrize(
        "fixture_name, api_path, delete_method",
        [
            ("fxt_usb_camera_source", ConfigApiPath.SOURCES, "delete_source_by_id"),
            ("fxt_folder_sink", ConfigApiPath.SINKS, "delete_sink_by_id"),
        ],
    )
    def test_delete_config_success(
        self,
        fixture_name,
        api_path,
        delete_method,
        fxt_config_service,
        fxt_client,
        fxt_project,
        request,
    ):
        fxt_config = request.getfixturevalue(fixture_name)
        config_id = str(fxt_config.id)
        getattr(fxt_config_service, delete_method).side_effect = None

        response = fxt_client.delete(f"/api/projects/{fxt_project.id}/{api_path}/{config_id}")

        assert response.status_code == status.HTTP_204_NO_CONTENT
        getattr(fxt_config_service, delete_method).assert_called_once_with(fxt_config.id, fxt_project.id)

    @pytest.mark.parametrize(
        "api_path, delete_method",
        [
            (ConfigApiPath.SOURCES, "delete_source_by_id"),
            (ConfigApiPath.SINKS, "delete_sink_by_id"),
        ],
    )
    def test_delete_config_invalid_id(self, api_path, delete_method, fxt_config_service, fxt_client, fxt_project):
        getattr(fxt_config_service, delete_method).side_effect = None

        response = fxt_client.delete(f"/api/projects/{fxt_project.id}/{api_path}/invalid-id")

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        getattr(fxt_config_service, delete_method).assert_not_called()

    @pytest.mark.parametrize(
        "api_path, delete_method, resource_type",
        [
            (ConfigApiPath.SOURCES, "delete_source_by_id", ResourceType.SOURCE),
            (ConfigApiPath.SINKS, "delete_sink_by_id", ResourceType.SINK),
        ],
    )
    def test_delete_config_not_found(
        self,
        api_path,
        delete_method,
        resource_type,
        fxt_config_service,
        fxt_client,
        fxt_project,
    ):
        config_id = str(uuid4())
        getattr(fxt_config_service, delete_method).side_effect = ResourceNotFoundError(resource_type, config_id)

        response = fxt_client.delete(f"/api/projects/{fxt_project.id}/{api_path}/{config_id}")

        assert response.status_code == status.HTTP_404_NOT_FOUND

    @pytest.mark.parametrize(
        "fixture_name, api_path, delete_method, resource_type",
        [
            ("fxt_usb_camera_source", ConfigApiPath.SOURCES, "delete_source_by_id", ResourceType.SOURCE),
            ("fxt_folder_sink", ConfigApiPath.SINKS, "delete_sink_by_id", ResourceType.SINK),
        ],
    )
    def test_delete_config_in_use(
        self,
        fixture_name,
        api_path,
        delete_method,
        resource_type,
        fxt_config_service,
        fxt_client,
        fxt_project,
        request,
    ):
        fxt_config = request.getfixturevalue(fixture_name)
        config_id = str(fxt_config.id)
        err = ResourceInUseError(resource_type, config_id)
        getattr(fxt_config_service, delete_method).side_effect = err

        response = fxt_client.delete(f"/api/projects/{fxt_project.id}/{api_path}/{config_id}")

        assert response.status_code == status.HTTP_409_CONFLICT
        assert str(err) == response.json()["detail"]

    @pytest.mark.parametrize(
        "fixture_name,api_path,get_method, expected_yaml",
        [
            (
                "fxt_usb_camera_source",
                ConfigApiPath.SOURCES,
                "get_source_by_id",
                "device_id: 1\nname: Test USB Camera Source\nsource_type: usb_camera\n",
            ),
            (
                "fxt_folder_sink",
                ConfigApiPath.SINKS,
                "get_sink_by_id",
                "folder_path: /test/path\nname: Test Folder Sink\noutput_formats:\n- predictions"
                "\nrate_limit: 0.1\nsink_type: folder\n",
            ),
        ],
    )
    def test_export_config_success(
        self,
        fixture_name,
        api_path,
        get_method,
        expected_yaml,
        fxt_config_service,
        fxt_client,
        fxt_project,
        request,
    ):
        fxt_config = request.getfixturevalue(fixture_name)
        config_id = str(fxt_config.id)
        getattr(fxt_config_service, get_method).return_value = fxt_config

        response = fxt_client.post(f"/api/projects/{fxt_project.id}/{api_path}/{config_id}:export")

        assert response.status_code == status.HTTP_200_OK
        assert response.headers["content-type"] == "application/x-yaml"
        assert f"{api_path[:-1]}_{config_id}.yaml" in response.headers["content-disposition"]
        assert response.text == expected_yaml
        getattr(fxt_config_service, get_method).assert_called_once_with(fxt_config.id, fxt_project.id)

    @pytest.mark.parametrize(
        "fixture_name,api_path, create_method",
        [
            ("fxt_usb_camera_source", ConfigApiPath.SOURCES, "create_source"),
            ("fxt_folder_sink", ConfigApiPath.SINKS, "create_sink"),
        ],
    )
    def test_import_config_success(
        self,
        fixture_name,
        api_path,
        create_method,
        fxt_config_service,
        fxt_client,
        fxt_project,
        request,
    ):
        fxt_config = request.getfixturevalue(fixture_name)
        sink_data = fxt_config.model_dump(exclude={"id", "project_id"}, mode="json")
        yaml_content = yaml.safe_dump(sink_data)
        getattr(fxt_config_service, create_method).return_value = fxt_config

        files = {"yaml_file": ("test.yaml", io.BytesIO(yaml_content.encode()), "application/x-yaml")}
        response = fxt_client.post(f"/api/projects/{fxt_project.id}/{api_path}:import", files=files)

        assert response.status_code == status.HTTP_201_CREATED
        assert response.json()["id"] == str(fxt_config.id)
        getattr(fxt_config_service, create_method).assert_called_once()

    @pytest.mark.parametrize(
        "api_path",
        [ConfigApiPath.SOURCES, ConfigApiPath.SINKS],
    )
    def test_import_config_invalid_yaml(self, api_path, fxt_client, fxt_project):
        invalid_yaml = "invalid: yaml: content: ["
        files = {"yaml_file": ("test.yaml", io.BytesIO(invalid_yaml.encode()), "application/x-yaml")}

        response = fxt_client.post(f"/api/projects/{fxt_project.id}/{api_path}:import", files=files)

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "Invalid YAML format" in response.json()["detail"]

    @pytest.mark.parametrize(
        "api_path, config_type",
        [
            (ConfigApiPath.SOURCES, "source_type"),
            (ConfigApiPath.SINKS, "sink_type"),
        ],
    )
    def test_import_disconnected_config_fails(self, api_path, config_type, fxt_client, fxt_project):
        config_data = {config_type: "disconnected", "name": "Test"}
        yaml_content = yaml.safe_dump(config_data)
        files = {"yaml_file": ("test.yaml", io.BytesIO(yaml_content.encode()), "application/x-yaml")}

        response = fxt_client.post(f"/api/projects/{fxt_project.id}/{api_path}:import", files=files)

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "DISCONNECTED cannot be imported" in response.json()["detail"]
