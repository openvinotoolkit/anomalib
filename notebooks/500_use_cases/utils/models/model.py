"""
 Copyright (C) 2020 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import logging
from openvino.runtime import PartialShape


class Model:
    def __init__(self, ie, model_path, input_transform=None):
        self.logger = logging.getLogger()
        self.logger.info('Reading model from IR...')
        self.net = ie.read_model(model_path)
        self.set_batch_size(1)
        self.input_transform = input_transform

    def preprocess(self, inputs):
        meta = {}
        return inputs, meta

    def postprocess(self, outputs, meta):
        return outputs

    def set_batch_size(self, batch):
        shapes = {}
        for input_layer in self.net.inputs:
            new_shape = list(input_layer.shape)
            new_shape[0] = 1
            shapes.update({input_layer: new_shape})
        self.net.reshape(shapes)
