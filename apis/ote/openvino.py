"""
Openvino Anomaly Task
"""

# Copyright (C) 2021 Intel Corporation
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

import logging
import os
import tempfile
from typing import Optional, Union

from addict import Dict as ADDict
from compression.api import DataLoader
from compression.engines.ie_engine import IEEngine
from compression.graph import load_model, save_model
from compression.graph.model_utils import compress_model_weights, get_nodes_by_type
from compression.pipeline.initializer import create_pipeline
from omegaconf import ListConfig
from omegaconf.dictconfig import DictConfig
from ote_sdk.entities.annotation import Annotation
from ote_sdk.entities.datasets import DatasetEntity
from ote_sdk.entities.inference_parameters import InferenceParameters
from ote_sdk.entities.model import ModelEntity, ModelStatus
from ote_sdk.entities.optimization_parameters import OptimizationParameters
from ote_sdk.entities.resultset import ResultSetEntity
from ote_sdk.entities.scored_label import ScoredLabel
from ote_sdk.entities.shapes.rectangle import Rectangle
from ote_sdk.entities.task_environment import TaskEnvironment
from ote_sdk.usecases.evaluation.metrics_helper import MetricsHelper
from ote_sdk.usecases.tasks.interfaces.evaluate_interface import IEvaluationTask
from ote_sdk.usecases.tasks.interfaces.inference_interface import IInferenceTask
from ote_sdk.usecases.tasks.interfaces.optimization_interface import (
    IOptimizationTask,
    OptimizationType,
)

from anomalib.core.model.inference import OpenVinoInferencer

logger = logging.getLogger(__name__)


class OTEOpenVINOAnomalyDataloader(DataLoader):
    """
    Dataloader for loading SC dataset into OTE OpenVINO Inferencer

    Args:
        dataset (DatasetEntity): SC dataset entity
        inferencer (OpenVinoInferencer): Openvino Inferencer
    """

    def __init__(self, config: Union[DictConfig, ListConfig], dataset: DatasetEntity, inferencer: OpenVinoInferencer):
        super().__init__(config=config)
        self.dataset = dataset
        self.inferencer = inferencer

    def __getitem__(self, index):
        image = self.dataset[index].numpy
        annotation = self.dataset[index].annotation_scene
        inputs = self.inferencer.pre_process(image)

        return (index, annotation), inputs

    def __len__(self):
        return len(self.dataset)


class OpenVINOAnomalyClassificationTask(IInferenceTask, IEvaluationTask, IOptimizationTask):
    """
    OpenVINO inference task

    Args:
        task_environment (TaskEnvironment): task environment of the trained anomaly model
        config (Union[DictConfig, ListConfig]): configuration file
    """

    def __init__(
        self,
        task_environment: TaskEnvironment,
        config: Union[DictConfig, ListConfig],
    ) -> None:
        self.task_environment = task_environment
        self.config = config
        self.inferencer = self.load_inferencer()
        labels = task_environment.get_labels()
        self.normal_label = [label for label in labels if label.name == "normal"][0]
        self.anomalous_label = [label for label in labels if label.name == "anomalous"][0]

    def infer(self, dataset: DatasetEntity, inference_parameters: InferenceParameters) -> DatasetEntity:
        for dataset_item in dataset:
            anomaly_map = self.inferencer.predict(dataset_item.numpy, superimpose=False)
            pred_score = anomaly_map.reshape(-1).max()
            # This always assumes that threshold is available in the task environment
            pred_label = pred_score >= self.task_environment.get_hyper_parameters().model.threhold
            assigned_label = self.anomalous_label if pred_label else self.normal_label
            shape = Annotation(
                Rectangle(x1=0, y1=0, x2=1, y2=1), labels=[ScoredLabel(assigned_label, probability=pred_score)]
            )
            dataset_item.append_annotations([shape])

        return dataset

    def evaluate(self, output_resultset: ResultSetEntity, evaluation_metric: Optional[str] = None):
        output_resultset.performance = MetricsHelper.compute_f_measure(output_resultset).get_performance()

    def optimize(
        self,
        optimization_type: OptimizationType,
        dataset: DatasetEntity,
        output_model: ModelEntity,
        optimization_parameters: Optional[OptimizationParameters],
    ):
        if optimization_type is not OptimizationType.POT:
            raise ValueError("POT is the only supported optimization type for OpenVino models")

        data_loader = OTEOpenVINOAnomalyDataloader(config=self.config, dataset=dataset, inferencer=self.inferencer)

        with tempfile.TemporaryDirectory() as tempdir:
            xml_path = os.path.join(tempdir, "model.xml")
            bin_path = os.path.join(tempdir, "model.bin")
            with open(xml_path, "wb") as xml_file:
                xml_file.write(self.task_environment.model.get_data("openvino.xml"))
            with open(bin_path, "wb") as bin_file:
                bin_file.write(self.task_environment.model.get_data("openvino.bin"))

            model_config = ADDict({"model_name": "openvino_model", "model": xml_path, "weights": bin_path})

            model = load_model(model_config)

            if get_nodes_by_type(model, ["FakeQuantize"]):
                logger.warning("Model is already optimized by POT")
                output_model.model_status = ModelStatus.FAILED
                return

        engine_config = ADDict({"device": "CPU"})

        hparams = self.task_environment.get_hyper_parameters()
        stat_subset_size = hparams.pot_parameters.stat_subset_size
        preset = hparams.pot_parameters.preset.name.lower()

        algorithms = [
            {
                "name": "DefaultQuantization",
                "params": {
                    "target_device": "ANY",
                    "preset": preset,
                    "stat_subset_size": min(stat_subset_size, len(data_loader)),
                },
            }
        ]

        engine = IEEngine(config=engine_config, data_loader=data_loader, metric=None)

        pipeline = create_pipeline(algorithms, engine)

        compressed_model = pipeline.run(model)

        compress_model_weights(compressed_model)

        with tempfile.TemporaryDirectory() as tempdir:
            save_model(compressed_model, tempdir, model_name="model")
            with open(os.path.join(tempdir, "model.xml"), "rb") as xml_file:
                output_model.set_data("openvino.xml", xml_file.read())
            with open(os.path.join(tempdir, "model.bin"), "rb") as bin_file:
                output_model.set_data("openvino.bin", bin_file.read())
        output_model.model_status = ModelStatus.SUCCESS

        self.task_environment.model = output_model
        self.inferencer = self.load_inferencer()

    def load_inferencer(self) -> OpenVinoInferencer:
        """
        Create the OpenVINO inverencer object

        Returns:
            OpenVinoInferencer object
        """
        return OpenVinoInferencer(
            config=self.config,
            path=(
                self.task_environment.model.get_data("openvino.xml"),
                self.task_environment.model.get_data("openvino.bin"),
            ),
        )
