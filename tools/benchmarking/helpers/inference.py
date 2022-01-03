"""Utils to help compute inference statistics."""

# Copyright (C) 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import time
from pathlib import Path
from typing import Dict, Iterable, List, Union

import numpy as np
from omegaconf import DictConfig, ListConfig
from torch.utils.data import DataLoader

from anomalib.core.model import AnomalyModule
from anomalib.core.model.inference import OpenVINOInferencer, TorchInferencer


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


def get_torch_throughput(
    config: Union[DictConfig, ListConfig], model: AnomalyModule, test_dataset: DataLoader, meta_data: Dict
) -> float:
    """Tests the model on dummy data. Images are passed sequentially to make the comparision with OpenVINO model fair.

    Args:
        config (Union[DictConfig, ListConfig]): Model config.
        model (Path): Model on which inference is called.
        test_dataset (DataLoader): The test dataset used as a reference for the mock dataset.
        meta_data (Dict): Metadata used for normalization.

    Returns:
        float: Inference throughput
    """
    model.eval()
    inferencer = TorchInferencer(config, model)
    torch_dataloader = MockImageLoader(config.dataset.image_size, len(test_dataset))
    start_time = time.time()
    # Since we don't care about performance metrics and just the throughput, use mock data.
    for image in torch_dataloader():
        inferencer.predict(image, superimpose=False, meta_data=meta_data)

    # get throughput
    inference_time = time.time() - start_time
    throughput = len(test_dataset) / inference_time

    return throughput


def get_openvino_throughput(
    config: Union[DictConfig, ListConfig], model_path: Path, test_dataset: DataLoader, meta_data: Dict
) -> float:
    """Runs the generated OpenVINO model on a dummy dataset to get throughput.

    Args:
        config (Union[DictConfig, ListConfig]): Model config.
        model_path (Path): Path to folder containing the OpenVINO models. It then searches `model.xml` in the folder.
        test_dataset (DataLoader): The test dataset used as a reference for the mock dataset.
        meta_data (Dict): Metadata used for normalization.

    Returns:
        float: Inference throughput
    """
    inferencer = OpenVINOInferencer(config, model_path / "model.xml")
    openvino_dataloader = MockImageLoader(config.dataset.image_size, total_count=len(test_dataset))
    start_time = time.time()
    # Create test images on CPU. Since we don't care about performance metrics and just the throughput, use mock data.
    for image in openvino_dataloader():
        inferencer.predict(image, superimpose=False, meta_data=meta_data)

    # get throughput
    inference_time = time.time() - start_time
    throughput = len(test_dataset) / inference_time

    return throughput
