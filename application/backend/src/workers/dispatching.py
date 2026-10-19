# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import copy
import multiprocessing as mp
import queue
from typing import TYPE_CHECKING

from loguru import logger

from pydantic_models import Sink, SinkType
from services import ActivePipelineService, DispatchService
from utils.jpeg_encoder import JPEGEncoder
from workers.base import BaseThreadWorker

if TYPE_CHECKING:
    from multiprocessing.synchronize import Condition as ConditionClass
    from multiprocessing.synchronize import Event as EventClass

    import numpy as np

    from core.mjpeg_broadcaster import MJPEGBroadcaster
    from entities.stream_data import StreamData
    from services.dispatchers import Dispatcher


class DispatchingWorker(BaseThreadWorker):
    """A thread that pulls predictions from the queue and dispatches them to the configured outputs
    and visualization streams.
    """

    ROLE = "Dispatching"

    def __init__(
        self,
        pred_queue: mp.Queue,
        rtc_stream_queue: queue.Queue,
        mjpeg_broadcaster: MJPEGBroadcaster,
        stop_event: EventClass,
        active_config_changed_condition: ConditionClass,
    ) -> None:
        super().__init__(stop_event=stop_event)
        self._pred_queue = pred_queue
        self._rtc_stream_queue = rtc_stream_queue
        self._mjpeg_broadcaster = mjpeg_broadcaster

        self._active_config_changed_condition = active_config_changed_condition
        self._active_pipeline_service: ActivePipelineService | None = None

        self._prev_sink_config: Sink | None = None
        self._destinations: list[Dispatcher] = []
        self._jpeg_encoder = JPEGEncoder()

    def _reset_sink_if_needed(self, sink_config: Sink) -> None:
        if sink_config.sink_type is SinkType.DISCONNECTED:
            self._destinations = []
            self._prev_sink_config = copy.deepcopy(sink_config)
        elif not self._prev_sink_config or sink_config != self._prev_sink_config:
            logger.info(f"Sink config changed from {self._prev_sink_config} to {sink_config}")
            self._destinations = DispatchService.get_destinations(output_configs=[sink_config])
            self._prev_sink_config = copy.deepcopy(sink_config)

    def _publish_mjpeg(self, frame: np.ndarray) -> None:
        """Encode and broadcast MJPEG frame bytes.

        Args:
            frame (np.ndarray): RGB frame to encode and publish.
        """
        jpeg_bytes = self._jpeg_encoder.encode(frame)
        if jpeg_bytes is None:
            return
        self._mjpeg_broadcaster.publish_threadsafe(jpeg_bytes)

    @logger.catch()
    async def run_loop(self) -> None:
        self._active_pipeline_service = await ActivePipelineService.create(
            config_changed_condition=self._active_config_changed_condition,
            start_daemon=True,
        )

        while not self.should_stop():
            # Exit if parent process died (if ever run as a process)
            parent_process = mp.parent_process()
            if parent_process is not None and not parent_process.is_alive():
                break

            # Read from the queue
            try:
                stream_data: StreamData = self._pred_queue.get(timeout=1)
            except queue.Empty:
                logger.debug("Nothing to dispatch yet... retrying in 1 second")
                await asyncio.sleep(1)
                continue

            passthrough_mode = not self._active_pipeline_service.is_running
            if passthrough_mode:
                logger.trace("Passthrough mode; only dispatching to visualization streams")
                # Dispatch to MJPEG and WebRTC streams
                self._publish_mjpeg(stream_data.frame_data)
                try:
                    self._rtc_stream_queue.put(stream_data.frame_data, block=False)
                except queue.Full:
                    logger.trace("Visualization queue is full; skipping")
                continue

            sink_config = self._active_pipeline_service.sink_config
            if sink_config.sink_type == SinkType.DISCONNECTED:
                logger.debug("No sink available due to ephemeral inference")

            self._reset_sink_if_needed(sink_config)

            if stream_data.inference_data is None:
                logger.error("Missing inference data in stream_data; skipping dispatch")
                continue

            inference_data = stream_data.inference_data
            if inference_data is None:
                logger.error("No inference data available")
                continue

            image_with_visualization = inference_data.visualized_prediction
            prediction = inference_data.prediction
            # Postprocess and dispatch results
            # Dispatch to visualization streams
            self._publish_mjpeg(image_with_visualization)
            try:
                self._rtc_stream_queue.put(image_with_visualization, block=False)
            except queue.Full:
                logger.trace("Visualization queue is full; skipping")

            # Dispatch to other destinations
            try:
                async with asyncio.TaskGroup() as task_group:
                    for destination in self._destinations:
                        task_group.create_task(
                            destination.dispatch(
                                original_image=stream_data.frame_data,
                                image_with_visualization=image_with_visualization,
                                predictions=prediction,
                            ),
                        )
            except Exception as e:
                logger.error(f"One or more errors occurred during dispatch: {e}")
