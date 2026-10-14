# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import time
from collections import defaultdict
from datetime import UTC, datetime
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.synchronize import Lock
from typing import NamedTuple
from uuid import UUID

import numpy as np
from loguru import logger

MAX_MEASUREMENTS = 1024  # max number of measurements to keep
DTYPE = np.dtype([
    ("model_id", "U36"),  # 36 * 4 = 144 bytes for UUID string
    ("latency_ms", np.dtype(float)),  # 8 bytes for latency in milliseconds
    ("timestamp", np.dtype(float)),  # 8 bytes for timestamp (epoch time in seconds)
])  # 144 + 8 + 8 = 160 bytes per latency measurement
SIZE = DTYPE.itemsize * MAX_MEASUREMENTS  # 160 * 1024 = 163840 bytes (160KB) allocated


class LatencyMeasurement(NamedTuple):
    """Individual latency measurement"""

    model_id: str  # UUID as 36 character string e.g. "7fdcae10-c3e3-427c-9a88-d346bb219d0f"
    latency_ms: float
    timestamp: float


class MetricsService:
    """Process-safe metrics service using shared memory for model metric data"""

    def __init__(self, shm_name: str, lock: Lock, max_age_seconds: int = 60):
        self._max_age_seconds = max_age_seconds
        self._lock = lock
        self._shm = SharedMemory(name=shm_name)
        self._array: np.ndarray = np.ndarray((MAX_MEASUREMENTS,), dtype=DTYPE, buffer=self._shm.buf)
        self._head = 0  # index for next write

    def update_max_age(self, max_age_seconds: int) -> None:
        with self._lock:
            self._max_age_seconds = max_age_seconds

    @staticmethod
    def record_inference_start() -> float:
        return time.perf_counter()

    def record_inference_end(self, model_id: UUID, start_time: float) -> None:
        """
        Record the end of an inference and store the latency measurement.

        Args:
            model_id: UUID of the model to record measurement for
            start_time: Start time from record_inference_start()
        """
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000.0
        timestamp = datetime.now(UTC).timestamp()

        measurement = LatencyMeasurement(str(model_id), latency_ms, timestamp)
        with self._lock:
            idx = self._head % MAX_MEASUREMENTS
            self._array[idx] = (measurement.model_id, measurement.latency_ms, measurement.timestamp)
            self._head += 1
            logger.debug(f"Latency measurement recorded for model {model_id}: {latency_ms:.2f} ms")

    def get_latency_measurements(self, model_id: UUID, time_window: int = 60) -> list[float]:
        """
        Retrieve latency measurements for a specific model within the given time window.

        Args:
            model_id: UUID of the model to filter measurements
            time_window: Time window in seconds to look back for measurements (default 60s)

        Returns: List of latency measurements in milliseconds
        """
        str_model_id = str(model_id)
        cutoff_time = datetime.now(UTC).timestamp() - time_window
        with self._lock:
            arr = self._array.copy()
        result = []
        for entry in arr:
            if entry["timestamp"] < cutoff_time or entry["timestamp"] == 0.0:
                continue
            if entry["model_id"] == str_model_id:
                result.append(entry["latency_ms"])
        return result

    def get_throughput_measurements(self, model_id: UUID, time_window: int = 60) -> tuple[int, list[tuple[float, int]]]:
        """
        Retrieve throughput measurements for a specific model within the given time window.

        Args:
            model_id: UUID of the model to filter measurements
            time_window: Time window in seconds to look back for measurements (default 60s)

        Returns: Tuple of (total_requests, list of (timestamp, inference_count) per second)
        """
        str_model_id = str(model_id)
        cutoff_time = datetime.now(UTC).timestamp() - time_window
        with self._lock:
            arr = self._array.copy()

        # Count inferences per second
        inferences_per_second: dict[int, int] = defaultdict(int)
        total_requests = 0

        for entry in arr:
            if entry["timestamp"] < cutoff_time or entry["timestamp"] == 0.0:
                continue
            if entry["model_id"] == str_model_id:
                # Round timestamp to the nearest second
                second_timestamp = int(entry["timestamp"])
                inferences_per_second[second_timestamp] += 1
                total_requests += 1

        # Convert to list of (timestamp, count) tuples
        throughput_data = [(float(ts), count) for ts, count in inferences_per_second.items()]
        throughput_data.sort()  # Sort by timestamp

        return total_requests, throughput_data

    def reset(self) -> None:
        with self._lock:
            self._max_age_seconds = 60
            self._array[:] = "00000000-0000-0000-0000-000000000000", 0.0, 0.0
            self._head = 0

    def __del__(self):
        self._shm.close()
