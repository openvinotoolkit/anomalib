"""Test Config Getter."""

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

import pytest

from anomalib.config import get_configurable_parameters


class TestConfig:
    """Test Config Getter."""

    def test_get_configurable_parameters_return_correct_model_name(self):
        """Configurable parameter should return the correct model name."""
        model_name = "stfpm"
        configurable_parameters = get_configurable_parameters(model_name)
        assert configurable_parameters.model.name == model_name

    def test_get_configurable_parameter_fails_with_none_arguments(self):
        """Configurable parameter should raise an error with none arguments."""
        with pytest.raises(ValueError):
            get_configurable_parameters()
