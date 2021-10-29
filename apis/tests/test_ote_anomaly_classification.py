"""
Test Anomaly Classification Task
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
from threading import Thread

import numpy as np

from apis.ote.utils.config import get_anomalib_config
from apis.tests.helpers.config import get_config
from apis.tests.helpers.train import OTEAnomalyTrainer

logger = logging.getLogger(__name__)


class TestAnomalyClassification:
    """
    Anomaly Classification Task Tests.
    """

    _trainer: OTEAnomalyTrainer

    @staticmethod
    def test_ote_config():
        """
        Test generation of OTE config object from model template and conversion to Anomalib format. Also checks if
        default values are overwritten in Anomalib config.
        """
        template_file_path = "apis/ote/configs/template.yaml"
        train_batch_size = 16

        ote_config = get_config(template_file_path)

        # change parameter value in OTE config
        ote_config.dataset.train_batch_size = train_batch_size
        # convert OTE -> Anomalib
        anomalib_config = get_anomalib_config(ote_config)
        # check if default parameter was overwritten
        assert anomalib_config.dataset.train_batch_size == train_batch_size

    def test_cancel_training(self):
        """
        Training should stop when `cancel_training` is called
        """
        self._trainer = OTEAnomalyTrainer()
        thread = Thread(target=self._trainer.train)
        thread.start()
        self._trainer.cancel_training()
        assert self._trainer.base_task.model.results.performance == {}

    def test_ote_train_export_and_optimize(self):
        """
        E2E Train-Export Should Yield Similar Inference Results
        """
        # Train the model
        self._trainer = OTEAnomalyTrainer()
        self._trainer.train()
        base_results = self._trainer.validate(task=self._trainer.base_task)

        # Convert the model to OpenVINO
        self._trainer.export()
        openvino_results = self._trainer.validate(task=self._trainer.openvino_task)

        # Optimize the OpenVINO Model via POT
        optimized_openvino_results = self._trainer.validate(task=self._trainer.openvino_task, optimize=True)

        # Performance should be higher than a threshold.
        assert base_results.performance.score.value > 0.6

        # Performance should be almost the same
        assert np.allclose(base_results.performance.score.value, openvino_results.performance.score.value)
        assert np.allclose(openvino_results.performance.score.value, optimized_openvino_results.performance.score.value)
