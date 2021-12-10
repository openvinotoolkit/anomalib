"""Benchmark all the algorithms in the repo."""

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


import os
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, Iterable, List, Union

import albumentations as A
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, ListConfig
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from torch.utils.data import DataLoader
from tqdm import tqdm

from anomalib.config import get_configurable_parameters
from anomalib.core.callbacks import get_callbacks
from anomalib.core.model.inference import OpenVINOInferencer
from anomalib.data import get_datamodule
from anomalib.models import get_model

MODEL_LIST = ["padim", "dfkde", "dfm", "patchcore", "stfpm"]
SEED = 42

# Modify category list according to dataset
CATEGORY_LIST = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]

IMAGE_SIZE_LIST = ["128", "256", "512"]


class OpenVINOMockDataLoader:
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


def get_openvino_throughput(config: Union[DictConfig, ListConfig], model_path: Path, test_dataset: DataLoader) -> float:
    """Runs the generated OpenVINO model on a dummy dataset to get throughput.

    Args:
        config (Union[DictConfig, ListConfig]): Model config.
        model_path (Path): Path to folder containint the OpenVINO models. It then searches `model.xml` in the folder.
        test_dataset (DataLoader): The test dataset used as a reference for the mock dataset.

    Returns:
        float: Inference throughput
    """
    # transform might not always be in config
    if "transform" not in config.keys():
        # save transforms in temporary location
        A.save(A.Compose([A.Resize(*config.dataset.image_size)]), "/tmp/transforms.yaml", data_format="yaml")
        config.transform = "/tmp/transforms.yaml"
    inferencer = OpenVINOInferencer(config, model_path / "model.xml")
    openvino_dataloader = OpenVINOMockDataLoader(config.dataset.image_size, total_count=len(test_dataset))
    start_time = time.time()
    # Create test images on CPU. Since we don't care about performance metrics and just the throughput, use mock data.
    for image in openvino_dataloader():
        inferencer(image)

    # get throughput
    inference_time = time.time() - start_time
    throughput = len(test_dataset) / inference_time

    os.unlink("/tmp/transforms.yaml")
    return throughput


def convert_to_openvino(model: pl.LightningModule, export_path: Union[Path, str], input_size: List[int]):
    """Convert the trained model to OpenVINO."""
    export_path = export_path if isinstance(export_path, Path) else Path(export_path)
    onnx_path = export_path / "model.onnx"
    height, width = input_size
    torch.onnx.export(
        model,
        torch.zeros((1, 3, height, width)).to(model.device),
        onnx_path,
        opset_version=11,
        input_names=["input"],
        output_names=["output"],
    )
    optimize_command = "mo --input_model " + str(onnx_path) + " --output_dir " + str(export_path)
    os.system(optimize_command)


def update_callbacks(config: Union[DictConfig, ListConfig]) -> List[Callback]:
    """Disable callbacks in config.

    Args:
        config (Union[DictConfig, ListConfig]): Model config loaded from anomalib
    """
    config.project.log_images_to = []  # disable visualizer callback

    # disable openvino optimization
    if "optimization" in config.keys():
        config.pop("optimization")

    # disable load model callback
    if "weight_file" in config.model.keys():
        config.model.pop("weight_file")

    # disable save_to_csv. Metrics will be captured from stdout
    if "save_to_csv" in config.project.keys():
        config.project.pop("save_to_csv")

    callbacks = get_callbacks(config)

    # remove ModelCheckpoint callback
    for index, callback in enumerate(callbacks):
        if isinstance(callback, ModelCheckpoint):
            callbacks.pop(index)
            break

    return callbacks


def get_single_model_metrics(model_name: str, gpu_count: int, category: str, image_size: int) -> Dict:
    """Collects metrics for `model_name` and returns a dict of results.

    Args:
        model_name (str): Name of the model
        gpu_count (int): Number of gpus. Use `gpu_count=0` for cpu
        category (str): Category of the dataset

    Returns:
        Dict: Collection of all the metrics such as time taken, throughput and performance scores.
    """
    config = get_configurable_parameters(model_name=model_name)
    # Seed for reproducibility
    seed_everything(42)

    # TODO run gpu and cpu training in parallel as they don't share resources. issue #18
    config.trainer.gpus = gpu_count
    config.dataset.category = category
    config.dataset.image_size = [image_size, image_size]
    config.model.input_size = config.dataset.image_size

    with TemporaryDirectory() as project_path:
        config.project.path = project_path
        datamodule = get_datamodule(config)
        model = get_model(config)

        callbacks = update_callbacks(config)

        trainer = Trainer(**config.trainer, logger=None, callbacks=callbacks)

        start_time = time.time()
        trainer.fit(model=model, datamodule=datamodule)

        # get start time
        training_time = time.time() - start_time

        # Creating new variable is faster according to https://stackoverflow.com/a/4330829
        start_time = time.time()
        # get test results
        test_results = trainer.test(model=model, datamodule=datamodule)

        # get testing time
        testing_time = time.time() - start_time

        throughput = len(datamodule.test_dataloader().dataset) / testing_time

        # Get OpenVINO metrics
        openvino_throughput = float("nan")
        if gpu_count > 0:  # Train only once if both CPU and GPU training are called
            # Create dirs for openvino model export
            openvino_export_path = Path("./exported_models") / model_name / category / str(image_size)
            openvino_export_path.mkdir(parents=True, exist_ok=True)
            convert_to_openvino(model, openvino_export_path, config.model.input_size)
            openvino_throughput = get_openvino_throughput(
                config, openvino_export_path, datamodule.test_dataloader().dataset
            )

        # arrange the data
        data = {
            "Training Time (s)": training_time,
            "Testing Time (s)": testing_time,
            "Inference Throughput (fps)": throughput,
            "OpenVINO Inference Throughput (fps)": openvino_throughput,
            "Image Size": image_size,
        }
        for key, val in test_results[0].items():
            data[key] = float(val)

    return data


def sweep():
    """Go over all models, categories, and devices and collect metrics."""
    # TODO add image resolution; maybe a recursive grid search. issue #18
    for model_name in MODEL_LIST:
        metrics_list = []
        for image_size in IMAGE_SIZE_LIST:
            for category in tqdm(CATEGORY_LIST, desc=f"{model_name}|{image_size}"):
                for gpu_count in range(1, 2):
                    model_metrics = get_single_model_metrics(model_name, gpu_count, category, int(image_size))
                    model_metrics["Device"] = "CPU" if gpu_count == 0 else "GPU"
                    model_metrics["Category"] = category
                    metrics_list.append(model_metrics)
        metrics_df = pd.DataFrame(metrics_list)
        result_path = Path(f"results/{model_name}.csv")
        os.makedirs(result_path.parent, exist_ok=True)
        metrics_df.to_csv(result_path)


if __name__ == "__main__":
    sweep()
