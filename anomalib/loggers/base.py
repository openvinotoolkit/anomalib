"""Base logger for image logging consistency across all loggers used in anomalib."""

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

from abc import abstractmethod
from typing import Any, Optional, Union

import numpy as np
from matplotlib.figure import Figure


class ImageLoggerBase:
    """Adds a common interface for logging the images."""

    @abstractmethod
    def add_image(self, image: Union[np.ndarray, Figure], name: Optional[str] = None, **kwargs: Any) -> None:
        """Interface to log images in the respective loggers."""
        raise NotImplementedError()
