# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import statistics
from datetime import UTC, datetime, timedelta
from uuid import UUID

from exceptions import ResourceNotFoundException
from pydantic_models import PipelineStatus
from pydantic_models.metrics import InferenceMetrics, LatencyMetrics, PipelineMetrics, ThroughputMetrics, TimeWindow

from .metrics_service import MetricsService
from .pipeline_service import PipelineService


class PipelineMetricsService:
    def __init__(
        self,
        pipeline_service: PipelineService,
        metrics_service: MetricsService,
    ) -> None:
        self._pipeline_service: PipelineService = pipeline_service
        self._metrics_service: MetricsService = metrics_service

    async def get_pipeline_metrics(self, pipeline_id: UUID, time_window: int = 60) -> PipelineMetrics:
        """Calculate metrics for a pipeline over a specified time window."""
        # First check if pipeline exists
        pipeline = await self._pipeline_service.get_pipeline_by_id(pipeline_id)
        if pipeline is None:
            raise ResourceNotFoundException(resource_id=pipeline_id, resource_name="pipeline")
        if pipeline.status != PipelineStatus.RUNNING:
            raise ValueError("Cannot get metrics for a pipeline that is not running.")

        # Calculate time window
        end_time = datetime.now(UTC)
        start_time = end_time - timedelta(seconds=time_window)

        # Get actual latency measurements from the metrics service
        latency_samples = self._metrics_service.get_latency_measurements(
            model_id=pipeline.model_id,  # type: ignore[arg-type] # model is always there for running pipeline
            time_window=time_window,
        )

        # Calculate latency metrics
        if latency_samples:
            latency_metrics = LatencyMetrics(
                avg_ms=statistics.mean(latency_samples),
                min_ms=min(latency_samples),
                max_ms=max(latency_samples),
                p95_ms=self._calculate_percentile(latency_samples, 95),
                latest_ms=latency_samples[-1],
            )
        else:
            # No data available
            latency_metrics = LatencyMetrics(avg_ms=None, min_ms=None, max_ms=None, p95_ms=None, latest_ms=None)

        # Get throughput measurements from the metrics service
        total_requests, throughput_data = self._metrics_service.get_throughput_measurements(
            model_id=pipeline.model_id,  # type: ignore[arg-type]
            time_window=time_window,
        )
        if total_requests:
            throughput_metrics = ThroughputMetrics(
                avg_requests_per_second=total_requests / time_window if time_window > 0 else 0.0,
                total_requests=total_requests,
                max_requests_per_second=max((count for _, count in throughput_data), default=0),
            )
        else:
            # No data available
            throughput_metrics = ThroughputMetrics(
                avg_requests_per_second=None,
                total_requests=None,
                max_requests_per_second=None,
            )

        window = TimeWindow(start=start_time, end=end_time, time_window=time_window)
        inference_metrics = InferenceMetrics(latency=latency_metrics, throughput=throughput_metrics)
        return PipelineMetrics(time_window=window, inference=inference_metrics)

    @staticmethod
    def _calculate_percentile(data: list[float], percentile: int) -> float:
        """Calculate the specified percentile of the data."""
        if not data:
            return 0.0

        sorted_data = sorted(data)
        k = (len(sorted_data) - 1) * (percentile / 100.0)
        floor_k = int(k)
        ceil_k = floor_k + 1

        if ceil_k >= len(sorted_data):
            return sorted_data[-1]

        # Linear interpolation
        fraction = k - floor_k
        return sorted_data[floor_k] + fraction * (sorted_data[ceil_k] - sorted_data[floor_k])
