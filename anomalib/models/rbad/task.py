#  COSMONiO © All rights reserved.
#  This file is subject to the terms and conditions defined in file 'LICENSE.txt',
#  which is part of this source code package.

import copy
import os
import pickle
import sys
from pickle import UnpicklingError
from typing import Optional

import numpy as np
import torch
from sc_sdk.entities.analyse_parameters import AnalyseParameters
from sc_sdk.entities.datasets import Dataset, DatasetItem, Subset
from sc_sdk.entities.label import Label, ScoredLabel
from sc_sdk.entities.metrics import NullPerformance, Performance
from sc_sdk.entities.model import Model, NullModel
from sc_sdk.entities.resultset import ResultSetEntity
from sc_sdk.entities.shapes.box import Box
from sc_sdk.entities.task_environment import TaskEnvironment
from sc_sdk.entities.train_parameters import TrainParameters
from sc_sdk.logging import logger_factory
from sc_sdk.usecases.evaluation.metrics_helper import MetricsHelper
from sc_sdk.usecases.tasks.image_deep_learning_task import ImageDeepLearningTask
from sc_sdk.usecases.tasks.interfaces.configurable_parameters_interface import (
    IConfigurableParameters,
)
from sc_sdk.usecases.tasks.interfaces.unload_interface import IUnload
from sc_sdk.utils.shape_factory import ShapeFactory
from tasks.anomaly_detection import anomalib
from tasks.anomaly_detection.anomalib.features import FeatureExtractor
from tasks.anomaly_detection.anomalib.model import NormalityModel
from tasks.anomaly_detection.anomalib.region import RegionExtractor
from tasks.anomaly_detection.anomalib.utils.time_tracker import TimeTracker
from tasks.anomaly_detection.configurable_parameters import AnomalyDetectionParameters

logger = logger_factory.get_logger("Anomaly Detection")


class CannotLoadModelException(ValueError):
    pass


class AnomalyDetectionTask(ImageDeepLearningTask, IConfigurableParameters, IUnload):
    def __init__(self, task_environment: TaskEnvironment):
        self.task_environment = task_environment
        self.labels = task_environment.labels
        self.train_tracker: Optional[TimeTracker] = None
        self.anomalous_name = "anomalous"
        self.normal_name = "normal"
        self.anomalous_label: Optional[Label] = None
        self.normal_label: Optional[Label] = None
        self.set_labels()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        #
        # Model
        self.region_extractor: Optional[RegionExtractor] = None
        self.feature_extractor: Optional[FeatureExtractor] = None
        self.model: Optional[NormalityModel] = None
        self.best_model: Optional[Model] = None
        self.inference_model: Optional[NormalityModel] = None
        #
        # Params
        self.configurable_parameters: Optional[AnomalyDetectionParameters] = None
        self.max_training_points = None
        self.pca_components = None
        self.confidence_threshold = None
        self.box_likelihood = None
        self.box_max_overlap = None
        self.box_min_size = None

        self.load_task_configuration()
        self.initialize_models()
        self.max_normal_images = 1000

    @staticmethod
    def is_docker():
        path = "/proc/self/cgroup"
        return os.path.exists("/.dockerenv") or os.path.isfile(path) and any("docker" in line for line in open(path))

    def unload(self):
        if self.is_docker():
            logger.warning("Got unload request. Unloading models. Throwing Segmentation Fault on purpose")
            import ctypes

            ctypes.string_at(0)
        else:
            logger.warning("Got unload request, but not on Docker. Only clearing CUDA cache")
            del self.region_extractor
            torch.cuda.empty_cache()
            logger.warning(
                f"Done unloading. " f"Torch is still occupying {torch.cuda.memory_allocated()} bytes of GPU memory"
            )

    def set_labels(self):
        try:
            self.anomalous_label = [label for label in self.labels if label.name == self.anomalous_name][0]
        except IndexError:
            raise IndexError(f"Anomaly detection task requires label named '{self.anomalous_name}'")
        try:
            self.normal_label = [label for label in self.labels if label.name == self.normal_name][0]
        except IndexError:
            raise IndexError(f"Anomaly detection task requires label named '{self.normal_name}'")

    def load_task_configuration(self):
        """
        Initializes variables to be used by the task
        """
        self.configurable_parameters = self.get_configurable_parameters(self.task_environment)
        anomaly_parameters = self.configurable_parameters.anomaly_parameters
        region_parameters = self.configurable_parameters.region_parameters

        self.pca_components = anomaly_parameters.pca_components.value
        self.max_training_points = anomaly_parameters.max_training_points.value
        self.confidence_threshold = anomaly_parameters.confidence_threshold.value

        box_likelihood = region_parameters.box_likelihood.value
        box_max_overlap = region_parameters.max_overlap.value
        box_min_size = region_parameters.min_size.value

        # only perform time-consuming initialization if necessary
        if not (
            box_likelihood == self.box_likelihood
            and box_max_overlap == self.box_max_overlap
            and box_min_size == self.box_min_size
        ):
            self.box_likelihood = box_likelihood
            self.box_max_overlap = box_max_overlap
            self.box_min_size = box_min_size
            self.region_extractor = (
                RegionExtractor(
                    max_overlap=self.box_max_overlap,
                    min_size=self.box_min_size,
                    likelihood=self.box_likelihood,
                )
                .eval()
                .to(self.device)
            )
            self.feature_extractor = FeatureExtractor().eval().to(self.device)

    @staticmethod
    def get_configurable_parameters(task_environment: TaskEnvironment) -> AnomalyDetectionParameters:
        """
        :return: A configurable parameters object for this task
        """
        return task_environment.get_configurable_parameters(instance_of=AnomalyDetectionParameters)

    def update_configurable_parameters(self, task_environment: TaskEnvironment):
        self.task_environment = task_environment
        self.load_task_configuration()

    def initialize_models(self):
        """
        Called to create model(s)
        """
        self.load_model(self.task_environment)
        if self.model is None:
            self.model = NormalityModel()

    def load_model(self, task_environment: TaskEnvironment):
        """
        Called to load weights from database into model
        """
        self.task_environment = task_environment
        logger.info("Loading the model")
        if not isinstance(task_environment.model, NullModel):
            try:
                sys.modules["tasks.anomaly_detection.anomalib"] = anomalib  # Legacy model support
                self.model = pickle.loads(task_environment.model.data)
                self.inference_model = self.model.clone()
            except UnpicklingError:
                raise CannotLoadModelException("Could not load the saved model. The model file structure is invalid.")

    @staticmethod
    def density_to_probability(density: float) -> float:
        """
        :param density: density of a box
        :return: probability that box with {density} is a anomalous
        """
        return 1 - 1 / (1 + np.exp(-0.5 - 0.05 * density))  # https://www.desmos.com/calculator/bciohsusgu

    def analyse(self, dataset: Dataset, analyse_parameters: Optional[AnalyseParameters] = None) -> Dataset:

        tracker = TimeTracker(steps=len(dataset))

        for dataset_item in dataset:
            image = dataset_item.numpy
            height, width, _ = image.shape
            is_normal = True

            boxes = self.region_extractor([image])
            shapes = []
            probabilities = None
            if not boxes.shape[0] == 0:
                features = self.feature_extractor(image, boxes)
                densities = self.inference_model.evaluate(features, as_density=True, ln=True)
                probabilities = np.asarray([self.density_to_probability(density) for density in densities])
                anomalous_mask = probabilities > self.confidence_threshold
                filtered_boxes = boxes[anomalous_mask]
                filtered_probabilities = probabilities[anomalous_mask]

                for box, probability in zip(filtered_boxes, filtered_probabilities):
                    is_normal = False
                    shapes.append(
                        Box(
                            x1=box[0] / width,
                            y1=box[1] / height,
                            x2=box[2] / width,
                            y2=box[3] / height,
                            labels=[ScoredLabel(self.anomalous_label, probability=probability)],
                        )
                    )
            if is_normal:
                shapes.append(
                    Box(
                        x1=0.0,
                        y1=0.0,
                        x2=1.0,
                        y2=1.0,
                        labels=[
                            ScoredLabel(
                                self.normal_label,
                                probability=self.confidence_threshold
                                if probabilities is None
                                else np.average(1 - probabilities),
                            )
                        ],
                    )
                )
            roi_as_box = ShapeFactory.shape_as_box(
                dataset_item.roi, dataset_item.media.width, dataset_item.media.height
            )

            dataset_item.annotation.append_shapes(shapes=shapes, roi=roi_as_box)

            tracker.tick()

        return dataset

    def is_anomalous(self, item: DatasetItem) -> bool:
        return self.anomalous_label in item.get_roi_labels(self.task_environment.labels)

    def prepare_dataset(self, dataset):
        """
        Maintain own train subset.
        The manipulation is not done on input dataset directly,
        but rather on a buffer dataset to make sure the input dataset stays the same.
        Pick N normal images from the dataset and make them training data (N = self.max_normal_images)
        """
        training_dataset = Dataset()
        train_items = [item for item in dataset if not self.is_anomalous(item)]
        train_items = train_items[: self.max_normal_images]
        for item in train_items:
            buf_item = copy.deepcopy(item)
            buf_item.subset = Subset.TRAINING
            training_dataset.append(buf_item)

        return training_dataset

    def train(self, dataset: Dataset, train_parameters: Optional[TrainParameters] = None) -> Model:
        self.load_task_configuration()
        self.model.cancel_training = False

        training_dataset = self.prepare_dataset(dataset)
        self.train_tracker = TimeTracker(steps=len(training_dataset))
        for dataset_item in training_dataset:
            self.train_tracker.tick()

            # Forward-pass image to extract regions.
            image = dataset_item.numpy
            boxes = self.region_extractor([image])
            if boxes.shape[0] == 0:
                continue

            # Get the features based on the regions.
            features = self.feature_extractor(image, boxes)
            self.model.stage_features(features)

            # Cancel feature extraction if training is cancelled.
            if self.model.cancel_training:
                logger.info("Cancelling training.")
                break

        # Check the model if it is successful.
        improved = False
        if (
            hasattr(self.model, "feature_list")
            and isinstance(self.model.feature_list, list)
            and len(self.model.feature_list) > self.pca_components
            and not self.model.cancel_training
        ):
            improved = self.model.commit()

        # return the improved model or if there is no model.
        if improved or (isinstance(self.task_environment.model, NullModel) and not self.model.cancel_training):
            logger.info("Training finished, and it has an improved model")
            model_data = self.get_model_bytes()
            model = Model(
                project=self.task_environment.project,
                task_node=self.task_environment.task_node,
                configuration=self.task_environment.get_model_configuration(),
                data=model_data,
                performance=NullPerformance(),
                tags=None,
                train_dataset=dataset,
            )
            self.best_model = model
            self.inference_model = self.model.clone()
            self.task_environment.model = model
        elif self.model.cancel_training:
            logger.info("Training cancelled.")
        else:
            logger.info("Training finished. Model has not improved, so it is not saved.")

        self.train_tracker = None
        return self.task_environment.model

    def compute_performance(self, resultset: ResultSetEntity) -> Performance:
        metrics = MetricsHelper.compute_f_measure(resultset)
        return metrics.get_performance()

    def get_model_bytes(self) -> bytes:
        """
        Saves the current model
        :return:
        """
        return pickle.dumps(self.model, protocol=pickle.HIGHEST_PROTOCOL)

    def get_training_progress(self):
        return -1 if self.train_tracker is None else self.train_tracker.progress * 100

    def cancel_training(self):
        logger.info("Cancel Training Requested.")
        self.model.cancel_training = True
