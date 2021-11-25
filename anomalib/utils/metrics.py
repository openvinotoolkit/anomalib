"""
Metrics
This module contains metric-related util functions.
"""

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


from typing import Tuple, Union

import numpy as np
from sklearn.metrics import precision_recall_curve
from torch import Tensor


def compute_threshold_and_f1_score(
    ground_truth: Union[Tensor, np.ndarray], predictions: Union[Tensor, np.ndarray]
) -> Tuple[float, float]:
    """
    Compute adaptive threshold, based on the f1 metric of the
    true labels and the predicted anomaly scores

    Args:
        ground_truth: Pixel-level or image-level ground truth labels.
        predictions: Anomaly scores predicted by the model.

    Examples:
        >>> import numpy as np
        >>> y_true = np.array([0, 0, 1, 1])
        >>> y_scores = np.array([0.1, 0.4, 0.35, 0.8])

        >>> compute_adaptive_threshold(y_true, y_scores)
        (0.35, 0.8)

    Returns:
        Threshold value based on the best f1 score.
        Value of the best f1 score.

    """

    precision, recall, thresholds = precision_recall_curve(ground_truth.flatten(), predictions.flatten())
    f1_score = (2 * precision * recall) / (precision + recall + 1e-10)
    threshold = thresholds[np.argmax(f1_score)]
    max_f1_score = np.max(f1_score)

    return threshold, max_f1_score
