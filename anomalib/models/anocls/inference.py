#  COSMONiO © All rights reserved.
#  This file is subject to the terms and conditions defined in file 'LICENSE.txt',
#  which is part of this source code package.

from typing import List, Union, Tuple

import numpy as np
import torch
from anocls.model import AnomalyClassificationModel
from anocls.transforms import get_anomaly_transform
from attrdict import AttrDict

from noussdk.entities.annotation import Annotation
from noussdk.entities.annotation import AnnotationKind
from noussdk.entities.id import ID
from noussdk.entities.label import Label
from noussdk.entities.label import ScoredLabel
from noussdk.entities.media_identifier import ImageIdentifier
from noussdk.entities.shapes.box import Box
from noussdk.usecases.exportable_code.inference import BaseInferencer


class AnomalyClassificationInferencer(BaseInferencer):

    def __init__(self, hparams: AttrDict, model: Union[str, AnomalyClassificationModel], class_labels: List[Label]):
        self.hparams = hparams
        self.normal_label = [label for label in class_labels if label.name == "normal"][0]
        self.anomalous_label = [label for label in class_labels if label.name == "anomalous"][0]

        if isinstance(model, AnomalyClassificationModel):
            self.model = model
        elif isinstance(model, str):
            self.model = AnomalyClassificationModel(hparams)
            self.load_model(model)
        else:
            raise ValueError("Parameter `model` could be either AnomalyClassificationModel or path to the model.")

    def load_model(self, model_path: str):
        with open(model_path, "rb") as f:
            model_bytes = f.read()
        self.model.load_model(model_bytes, self.hparams)
        self.model.update_inference_model()

    def pre_process(self, image: np.ndarray) -> Tuple:
        transform = get_anomaly_transform((self.hparams.image_size, self.hparams.image_size))
        image = transform(image).to(self.model.device)
        return image, None

    def forward(self, image: torch.Tensor) -> Tuple[list, list]:
        """
        Predicts anomaly scores for the provided dataset using the latest normality model.

        :return: a list of probabilities for the anomalous class and a list of corresponding indices in the nous dataset
        """

        # extract feature vector using pre-trained CNN
        feature_vector = self.model.feature_extractor(image.unsqueeze(0)).detach().cpu()
        # stage extracted feature vector
        probability = self.model.inference_model.predict(feature_vector).item()

        return probability

    def post_process(self, prediction, metadata) -> Annotation:
        if prediction >= self.hparams.confidence_threshold:
            label = ScoredLabel(self.anomalous_label, probability=prediction)
        else:
            label = ScoredLabel(self.normal_label, probability=1 - prediction)
        media_identifier = ImageIdentifier(image_id=ID("Test image"))
        shapes = [Box(x1=0.0, y1=0.0, x2=1.0, y2=1.0, labels=[label])]
        annotation = Annotation(kind=AnnotationKind.PREDICTION, media_identifier=media_identifier, shapes=shapes)
        return annotation
