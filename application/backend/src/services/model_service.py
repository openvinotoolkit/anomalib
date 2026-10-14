# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import asyncio
import base64
import io
import os
import shutil
import tempfile
from dataclasses import dataclass
from multiprocessing.synchronize import Event as EventClass
from pathlib import Path
from uuid import UUID, uuid4

import anyio
import cv2
import numpy as np
import openvino.properties.hint as ov_hints
from anomalib.data import AnomalibDataModule, Folder
from anomalib.deploy import CompressionType, ExportType, OpenVINOInferencer
from anomalib.engine import Engine
from anomalib.models import get_model
from loguru import logger
from PIL import Image
from sqlalchemy.ext.asyncio.session import AsyncSession

from db import get_async_db_session_ctx
from pydantic_models import Model, ModelList, PredictionLabel, PredictionResponse
from pydantic_models.base import Pagination
from pydantic_models.model import ExportParameters
from repositories import JobRepository, ModelRepository
from repositories.binary_repo import ModelBinaryRepository, ModelExportBinaryRepository
from services import ResourceNotFoundError
from services.dataset_snapshot_service import DatasetSnapshotService
from services.exceptions import DeviceNotFoundError, ResourceType
from services.system_service import SystemService

DEFAULT_DEVICE = "CPU"


@dataclass
class LoadedModel:
    name: str
    id: UUID
    model: Model
    device: str | None = None

    def __post_init__(self):
        if not self.device:
            logger.warning(f"No inference device set for model '{self.name}'; falling back to {DEFAULT_DEVICE}")
            self.device = DEFAULT_DEVICE


class ModelService:
    """Service for managing models and inference operations.

    Handles model CRUD operations, loading inference models, and running
    predictions on images. Uses asyncio.to_thread for CPU-intensive operations
    to maintain event loop responsiveness.
    """

    def __init__(self, mp_model_reload_event: EventClass | None = None) -> None:
        self._mp_model_reload_event = mp_model_reload_event

    def activate_model(self) -> None:
        """Notify workers to (re)load the active model.

        Sets the shared multiprocessing event so the inference worker reloads
        the current active model lazily and only once.
        """
        try:
            if self._mp_model_reload_event is not None:
                self._mp_model_reload_event.set()
        except Exception as e:
            # Best-effort signaling; avoid bubbling up to API layer
            logger.debug(f"Failed to signal model reload event: {e}")

    @staticmethod
    async def create_model(model: Model) -> Model:
        async with get_async_db_session_ctx() as session:
            repo = ModelRepository(session, project_id=model.project_id)
            return await repo.save(model)

    @staticmethod
    async def get_model_list(project_id: UUID, limit: int, offset: int) -> ModelList:
        async with get_async_db_session_ctx() as session:
            repo = ModelRepository(session, project_id=project_id)
            total = await repo.get_all_count()
            items = await repo.get_all_pagination(limit=limit, offset=offset)
        return ModelList(
            models=items,
            pagination=Pagination(
                limit=limit,
                offset=offset,
                count=len(items),
                total=total,
            ),
        )

    @staticmethod
    async def get_model_by_id(project_id: UUID, model_id: UUID) -> Model | None:
        async with get_async_db_session_ctx() as session:
            repo = ModelRepository(session, project_id=project_id)
            return await repo.get_by_id(model_id)

    @classmethod
    async def delete_model(cls, project_id: UUID, model_id: UUID) -> None:
        model = await cls.get_model_by_id(project_id, model_id)
        if not model:
            raise ResourceNotFoundError(resource_id=str(model_id), resource_type=ResourceType.MODEL)

        model_binary_repo = ModelBinaryRepository(project_id=project_id, model_id=model_id)
        try:
            await model_binary_repo.delete_model_folder()
        except FileNotFoundError:
            logger.warning(f"Model artifacts already absent on disk for model {model_id} in project {project_id}")
        ds_snapshot_id = model.dataset_snapshot_id
        await DatasetSnapshotService.delete_snapshot_if_unused(snapshot_id=ds_snapshot_id, project_id=project_id)

        train_job_id = model.train_job_id

        async with get_async_db_session_ctx() as session:
            repo = ModelRepository(session, project_id=project_id)
            await repo.delete_by_id(model_id)

            if train_job_id:
                job_repo = JobRepository(session)
                await job_repo.delete_by_id(train_job_id)

    @classmethod
    async def delete_project_models_db(cls, session: AsyncSession, project_id: UUID, commit: bool = False) -> None:
        """Delete all models associated with a project from the database."""
        # We still need to handle side effects like snapshot reference counting if possible,
        # but since we are deleting the project, all snapshots will be deleted anyway.
        # So we can just delete the models.
        repo = ModelRepository(session, project_id=project_id)
        await repo.delete_all(commit=commit)

    @classmethod
    async def cleanup_project_model_files(cls, project_id: UUID) -> None:
        """Cleanup model files for a project."""
        try:
            # Cleanup project folder (removes all model folders at once)
            # Note: using dummy model_id since we are deleting the entire project folder
            model_binary_repo = ModelBinaryRepository(project_id=project_id, model_id=uuid4())
            await model_binary_repo.delete_project_folder()
            logger.info(f"Cleaned up model files for project {project_id}")

            model_export_bin_repo = ModelExportBinaryRepository(project_id=project_id, model_id=uuid4())
            await model_export_bin_repo.delete_project_folder()
            logger.info(f"Cleaned up model export files for project {project_id}")
        except Exception as e:
            logger.warning(f"Failed to cleanup model files for project {project_id}: {e}")

    async def export_model(self, project_id: UUID, model_id: UUID, export_parameters: ExportParameters) -> Path:
        """Export a trained model to a zip file.

        Args:
            project_id: ID of the project
            model_id: ID of the model
            export_parameters: Parameters for export (format, compression)

        Returns:
            Path: Path to the exported zip file
        """
        model = await self.get_model_by_id(project_id, model_id)
        if model is None:
            raise ResourceNotFoundError(resource_type=ResourceType.MODEL, resource_id=str(model_id))

        bin_repo = ModelExportBinaryRepository(project_id=project_id, model_id=model_id)
        export_zip_path = anyio.Path(
            bin_repo.get_model_export_path(model_name=model.name, export_params=export_parameters),
        )

        # Cache check
        if await export_zip_path.exists():
            return Path(export_zip_path)

        # Locate checkpoint
        model_binary_repo = ModelBinaryRepository(project_id=project_id, model_id=model_id)
        ckpt_path = Path(model_binary_repo.model_folder_path) / "checkpoint" / "model.ckpt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found at {ckpt_path}")

        if export_parameters.compression in {CompressionType.INT8_PTQ, CompressionType.INT8_ACQ}:
            # We need reference images for INT8_PTQ and INT8_ACQ quantization.
            # Use the dataset snapshot to create a temporary datamodule.
            datamodule_name = "export-datamodule"
            async with DatasetSnapshotService.use_snapshot_as_folder(
                snapshot_id=model.dataset_snapshot_id,
                project_id=project_id,
            ) as dataset_path:
                datamodule = Folder(
                    name=datamodule_name,
                    normal_dir=os.path.join(dataset_path, "normal"),
                )
                datamodule.setup()

                return await asyncio.to_thread(
                    self._run_export,
                    architecture=model.architecture,
                    ckpt_path=ckpt_path,
                    export_parameters=export_parameters,
                    export_zip_path=Path(export_zip_path),
                    datamodule=datamodule,
                )

        # No datamodule needed for other compression types
        return await asyncio.to_thread(
            self._run_export,
            architecture=model.architecture,
            ckpt_path=ckpt_path,
            export_parameters=export_parameters,
            export_zip_path=Path(export_zip_path),
            datamodule=None,
        )

    @staticmethod
    def _run_export(
        architecture: str,
        ckpt_path: Path,
        export_parameters: ExportParameters,
        export_zip_path: Path,
        datamodule: AnomalibDataModule | None = None,
    ) -> Path:
        """Run the export process in a separate thread."""
        # Setup engine
        engine = Engine()
        model_module = get_model(architecture)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Export
            engine.export(
                model=model_module,
                export_type=export_parameters.format,
                export_root=temp_path,
                ckpt_path=str(ckpt_path),
                compression_type=export_parameters.compression,
                datamodule=datamodule,
            )

            # Create zip archive
            # shutil.make_archive expects base_name without extension for zip
            shutil.make_archive(str(export_zip_path.with_suffix("")), "zip", temp_path)

        return export_zip_path

    @classmethod
    async def load_inference_model(cls, model: Model, device: str | None = None) -> OpenVINOInferencer:
        """Load a model for inference using the anomalib OpenVINO inferencer.

        Args:
            model: The model to load
            device: Device to use for inference. If None, defaults to "AUTO"
        """
        if model.format is not ExportType.OPENVINO:
            raise NotImplementedError(f"Model format {model.format} is not supported for inference at this moment.")

        model_bin_repo = ModelBinaryRepository(project_id=model.project_id, model_id=model.id)
        model_path = model_bin_repo.get_weights_file_path(name="model.xml")
        device_name = device or DEFAULT_DEVICE
        try:
            return await asyncio.to_thread(
                OpenVINOInferencer,
                path=model_path,
                device=device_name.upper(),  # OV always expects uppercase device names
                config={ov_hints.performance_mode: ov_hints.PerformanceMode.LATENCY},
            )
        except Exception as e:
            if device and not SystemService.is_device_supported_for_inference(device):
                raise DeviceNotFoundError(device_name=device_name) from e
            raise e

    @classmethod
    async def predict_image(
        cls,
        model: Model,
        image_bytes: bytes,
        cached_models: dict[UUID, OpenVINOInferencer] | None = None,
        device: str | None = None,
    ) -> PredictionResponse:
        """Run prediction on an image using the specified model.

        Uses asyncio.to_thread to run the entire CPU-intensive prediction pipeline
        in a single thread, maintaining event loop responsiveness.

        Args:
            model: The model to use for prediction
            image_bytes: Raw image bytes from uploaded file
            cached_models: Optional dict to cache loaded models (for performance)
            device: Optional string indicating the device to use for inference

        Returns:
            PredictionResponse: Structured prediction results
        """
        # Determine if we can use cached model (must exist and device must match if specified)
        use_cached = (
            cached_models is not None
            and model.id in cached_models
            and (device is None or cached_models[model.id].device == device)
        )

        if use_cached and cached_models is not None:
            inference_model = cached_models[model.id]
        else:
            logger.info(f"Loading model with device: {device or DEFAULT_DEVICE}")
            inference_model = await cls.load_inference_model(model, device=device)
            if cached_models is not None:
                cached_models[model.id] = inference_model

        # Run entire prediction pipeline in a single thread
        # This includes image processing, model inference, and result processing
        response_data = await asyncio.to_thread(cls._run_prediction_pipeline, inference_model, image_bytes)

        return PredictionResponse(**response_data)

    @staticmethod
    def _run_prediction_pipeline(inference_model: OpenVINOInferencer, image_bytes: bytes) -> dict:
        """Run the complete prediction pipeline in a single thread."""
        # Process image
        npd = np.frombuffer(image_bytes, np.uint8)
        numpy_image = cv2.imdecode(npd, -1)
        if numpy_image is None:
            raise ValueError("Failed to decode image")

        # Remove alpha channel
        if len(numpy_image.shape) == 3 and numpy_image.shape[-1] == 4:
            numpy_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGBA2RGB)

        # Run prediction
        pred = inference_model.predict(numpy_image)

        # Process anomaly map
        if pred.anomaly_map is None:
            raise ValueError("Prediction returned no anomaly map")

        arr = pred.anomaly_map.squeeze()  # Remove dimensions of size 1
        arr_scaled = (arr * 255).astype(np.uint8)  # Scale to 0-255 and convert to uint8
        # convert to color map
        heatmap = cv2.applyColorMap(arr_scaled, cv2.COLORMAP_JET)
        # Add alpha channel with opacity weighted according to the anomaly score
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGBA)
        heatmap[:, :, 3] = arr_scaled

        im = Image.fromarray(heatmap)  # Automatically detects RGBA mode

        # Convert to base64
        with io.BytesIO() as buf:
            im.save(buf, format="PNG")
            im_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        # Create response data
        if pred.pred_label is None or pred.pred_score is None:
            raise ValueError("Prediction returned no label or score")

        label = PredictionLabel.ANOMALOUS if pred.pred_label.item() else PredictionLabel.NORMAL
        score = float(pred.pred_score.item())

        return {"anomaly_map": im_base64, "label": label, "score": score}
