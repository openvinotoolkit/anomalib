"""Callback to save metrics to CSV."""

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

from pathlib import Path

import numpy as np
import pandas as pd
from pytorch_lightning import Callback, Trainer

from anomalib.models.components import AnomalyModule


class SaveToCSVCallback(Callback):
    """Callback that saves the inference results of a model.

    The callback generates a csv file that saves the predicted label, the true label and the image name.
    """

    def __init__(self):
        """Callback to save metrics to CSV."""

    def on_test_epoch_end(self, trainer: Trainer, pl_module: AnomalyModule) -> None:
        """Save Results at the end of testing.

        Args:
            trainer (Trainer): Pytorch lightning trainer object (unused)
            pl_module (LightningModule): Lightning modules derived from BaseAnomalyLightning object.
        """
        results = pl_module.results
        data_frame = pd.DataFrame(
            {
                "name": results.filenames,
                "true_label": results.true_labels,
                "pred_label": results.pred_labels.astype(int),
                "wrong_prediction": np.logical_xor(results.true_labels, results.pred_labels).astype(int),
            }
        )

        if trainer.log_dir is not None:
            data_frame.to_csv(Path(trainer.log_dir) / "results.csv")
        else:
            raise ValueError("trainer.log_dir does not exist to save the results.")
