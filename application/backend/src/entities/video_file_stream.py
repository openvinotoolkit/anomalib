# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import time

import cv2
import numpy as np

from entities.base_opencv_stream import BaseOpenCVStream
from entities.stream_data import StreamData
from pydantic_models import SourceType


class VideoFileStream(BaseOpenCVStream):
    """Video stream implementation using video file via OpenCV."""

    def __init__(self, video_path: str) -> None:
        """Initialize video file stream."""
        super().__init__(source=video_path, source_type=SourceType.VIDEO_FILE, video_path=video_path)
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_time = 1.0 / fps if fps > 0 else 1.0 / 30.0
        self.last_frame_time = time.time()

    def get_data(self) -> StreamData:
        """Get the latest frame from the video stream."""
        current_time = time.time()
        elapsed_time = current_time - self.last_frame_time
        if elapsed_time < self.frame_time:
            time.sleep(self.frame_time - elapsed_time)
        self.last_frame_time = time.time()
        return super().get_data()

    def _handle_read_failure(self) -> np.ndarray:
        """Reset video to beginning when it ends and try again."""
        if self.cap is None:
            raise RuntimeError("Video capture not initialized")

        # Reset video to beginning when it ends
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to capture frame from video file")
        return frame

    def is_real_time(self) -> bool:
        return False
