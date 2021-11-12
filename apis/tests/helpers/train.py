"""
Anomaly Classification Training Helper
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

import importlib
import logging
import os
import time
from typing import Union

from ote_sdk.configuration.helper import create
from ote_sdk.entities.inference_parameters import InferenceParameters
from ote_sdk.entities.label_schema import LabelSchemaEntity
from ote_sdk.entities.model import (
    ModelEntity,
    ModelOptimizationType,
    ModelPrecision,
    ModelStatus,
    OptimizationMethod,
)
from ote_sdk.entities.model_template import TargetDevice, parse_model_template
from ote_sdk.entities.optimization_parameters import OptimizationParameters
from ote_sdk.entities.resultset import ResultSetEntity
from ote_sdk.entities.subset import Subset
from ote_sdk.entities.task_environment import TaskEnvironment
from ote_sdk.entities.train_parameters import TrainParameters
from ote_sdk.usecases.tasks.interfaces.export_interface import ExportType
from ote_sdk.usecases.tasks.interfaces.optimization_interface import OptimizationType

from apis.ote import AnomalyClassificationTask
from apis.ote.openvino import OpenVINOAnomalyClassificationTask
from apis.tests.helpers.dataset import OTEAnomalyDatasetGenerator
from tests.helpers.dataset import get_dataset_path

logger = logging.getLogger(__name__)


class OTEAnomalyTrainer:
    """
    OTE Trainer is a helper class that creates the required components
    to train, infer and export a model.

    Args:
        model_template_path (str, optional): path to model template.
            Defaults to "./apis/ote/configs/template.yaml".
    """

    def __init__(self, model_template_path: str = "./apis/ote/configs/template.yaml"):
        dataset_path = os.path.join(get_dataset_path(), "bottle")
        self.dataset_generator = OTEAnomalyDatasetGenerator(path=dataset_path)
        self.dataset = self.dataset_generator.generate()

        self.model_template_path = model_template_path
        self.model_template = parse_model_template(model_template_path)
        self.task_environment = self.create_task_environment()
        self.base_task = self.create_task()
        self.openvino_task: OpenVINOAnomalyClassificationTask

        self.output_model = ModelEntity(
            train_dataset=self.dataset,
            configuration=self.task_environment.get_model_configuration(),
            model_status=ModelStatus.NOT_READY,
        )

        self.was_training_run_before: bool = False
        self.was_export_run_before: bool = False
        self.stored_exception_training: Exception
        self.stored_exception_export: Exception

    def create_task_environment(self) -> TaskEnvironment:
        """
        Create Task Environment

        Returns:
            TaskEnvironment: Task Environment
        """
        hyper_parameters = create(input_config=self.model_template.hyper_parameters.data)

        labels = [self.dataset_generator.normal_label, self.dataset_generator.abnormal_label]
        label_schema = LabelSchemaEntity.from_labels(labels)

        task_environment = TaskEnvironment(
            model_template=self.model_template, model=None, hyper_parameters=hyper_parameters, label_schema=label_schema
        )

        return task_environment

    def create_task(self) -> AnomalyClassificationTask:
        """
        Create Anomaly Training Task

        Returns:
            AnomalyClassificationTask: anomaly training task.
        """
        path = self.model_template.entrypoints.base
        module_name, class_name = path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)(task_environment=self.task_environment)

    def train(self):
        """
        Train OTE Model
        """

        # Check if there is a previous train cycle, and if it's successful.
        if self.was_training_run_before and self.stored_exception_training:
            logger.warning(
                "In function run_ote_training_once: found that previous call of the function "
                "caused exception -- re-raising it"
            )
            raise self.stored_exception_training

        # Train the Model.
        if not self.was_training_run_before:
            try:
                self.base_task.train(
                    dataset=self.dataset, output_model=self.output_model, train_parameters=TrainParameters()
                )
                # Update task environment threshold based on the computed value during training
                hyper_parameters = self.task_environment.get_hyper_parameters()
                hyper_parameters.model.threhold = self.base_task.model.threshold.item()
                self.task_environment.set_hyper_parameters(hyper_parameters=hyper_parameters)

            except Exception as exception:
                self.stored_exception_training = exception
                raise exception

            self.was_training_run_before = True

        # Get the training performance of `output_model`.
        performance = self.output_model.performance
        if performance is None:
            raise ValueError("Model does not have a saved performance.")

        logger.debug("Training performance: %s, %3.2f", performance.score.name, performance.score.value)

    def validate(
        self,
        task: Union[AnomalyClassificationTask, OpenVINOAnomalyClassificationTask],
        subset=Subset.TESTING,
        optimize: bool = False,
    ) -> ResultSetEntity:
        """
        Run one epoch on the inference dataset using Base or OpenVINO Inferencer.

        Args:
            task (Union[AnomalyClassificationTask, OpenVINOAnomalyClassificationTask]): Base or OpenVINO inferencer
            subset (Subset, optional): Validation or Test split. Defaults to Subset.TESTING.
            optimize (bool, optional): Boolean to optimize the model via POT.

        Returns:
            ResultSetEntity: [description]
        """

        inference_dataset = self.dataset.get_subset(subset)
        inference_parameters = InferenceParameters(is_evaluation=True)

        predicted_inference_dataset = task.infer(
            dataset=inference_dataset.with_empty_annotations(), inference_parameters=inference_parameters
        )

        result_set = ResultSetEntity(
            model=self.output_model,
            ground_truth_dataset=inference_dataset,
            prediction_dataset=predicted_inference_dataset,
        )

        if optimize:
            if isinstance(task, AnomalyClassificationTask):
                raise ValueError("Base task cannot perform optimization")

            optimized_model = ModelEntity(
                inference_dataset,
                self.task_environment.get_model_configuration(),
                optimization_type=ModelOptimizationType.POT,
                optimization_methods=[OptimizationMethod.QUANTIZATION],
                optimization_objectives={},
                precision=[ModelPrecision.INT8],
                target_device=TargetDevice.CPU,
                performance_improvement={},
                model_size_reduction=1.0,
                model_status=ModelStatus.NOT_READY,
            )
            self.openvino_task.optimize(
                optimization_type=OptimizationType.POT,
                dataset=inference_dataset,
                output_model=optimized_model,
                optimization_parameters=OptimizationParameters(),
            )

        task.evaluate(output_resultset=result_set)
        return result_set

    def cancel_training(self):
        """
        Cancel Training
        """
        time.sleep(1)
        self.base_task.cancel_training()

    def export(self):
        """
        Export the OpenVINO Model
        """
        self.base_task.export(ExportType.OPENVINO, self.output_model)
        self.openvino_task = OpenVINOAnomalyClassificationTask(
            config=self.base_task.config, task_environment=self.task_environment
        )
