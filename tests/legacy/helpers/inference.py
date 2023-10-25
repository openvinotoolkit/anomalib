"""Utilities to help tests inferencers"""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from typing import Iterable, List

import numpy as np


class MockImageLoader:
    """Create mock images for inference on CPU based on the specifics of the original torch test dataset.
    Uses yield so as to avoid storing everything in the memory.
    Args:
        image_size (List[int]): Size of input image
        total_count (int): Total images in the test dataset
    """

    def __init__(self, image_size: List[int], total_count: int):
        self.total_count = total_count
        self.image_size = image_size
        self.image = np.ones((*self.image_size, 3)).astype(np.uint8)

    def __len__(self):
        """Get total count of images."""
        return self.total_count

    def __call__(self) -> Iterable[np.ndarray]:
        """Yield batch of generated images.
        Args:
            idx (int): Unused
        """
        for _ in range(self.total_count):
            yield self.image
