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

import cv2

from openvino.model_zoo.model_api.models import SegmentationModel
from openvino.model_zoo.model_api.models.types import NumericalValue


class AnomalyClassification(SegmentationModel):
    __model__ = 'anomaly_classification'

    def __init__(self, model_adapter, configuration=None, preload=False):
        super().__init__(model_adapter, configuration, preload)

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters['resize_type'].update_default_value('crop')
        parameters.update({
            'threshold': NumericalValue(default_value=0.2, description="Threshold value to locate anomaly"),
        })

        return parameters

    def postprocess(self, outputs, meta):
        outputs = outputs[self.output_blob_name].squeeze()
        input_image_height = meta['original_shape'][0]
        input_image_width = meta['original_shape'][1]
        meta['threshold'] = self.threshold

        result = cv2.resize(outputs, (input_image_width, input_image_height))
        return result
