"""Methods to help post-process raw model outputs."""

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

from .post_process import (
    add_anomalous_label,
    add_normal_label,
    anomaly_map_to_color_map,
    compute_mask,
    superimpose_anomaly_map,
)
from .visualizer import Visualizer

__all__ = [
    "add_anomalous_label",
    "add_normal_label",
    "anomaly_map_to_color_map",
    "superimpose_anomaly_map",
    "compute_mask",
    "Visualizer",
]
