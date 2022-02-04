"""Get callbacks related to sweep."""

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


from typing import List

from pytorch_lightning import Callback

from anomalib.utils.callbacks.timer import TimerCallback


def get_sweep_callbacks() -> List[Callback]:
    """Gets callbacks relevant to sweep.

    Args:
        config (Union[DictConfig, ListConfig]): Model config loaded from anomalib

    Returns:
        List[Callback]: List of callbacks
    """
    callbacks: List[Callback] = [TimerCallback()]

    return callbacks
