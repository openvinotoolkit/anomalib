# TODO: Write our own implementation.
# TODO: https://jira.devtools.intel.com/browse/IAAALD-14

# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Abstract class for sampling methods.

Provides interface to sampling methods that allow same signature
for select_batch.  Each subclass implements select_batch_ with the desired
signature for readability.
"""

from __future__ import absolute_import, division, print_function

import abc

import numpy as np


class SamplingMethod:
    """Abstract class for sampling methods."""

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self, x, y, seed, **kwargs):
        self.embeddings = x
        self.targets = y
        self.seed = seed

    def flatten_x(self):
        """If shape of embedding vector is greater than 2, it is reshaped to 2"""
        shape = self.embeddings.shape
        flat_x = self.embeddings
        if len(shape) > 2:
            flat_x = np.reshape(self.embeddings, (shape[0], np.product(shape[1:])))
        return flat_x

    @abc.abstractmethod
    def select_batch_(self, model, already_selected, batch_size):
        """
        Abstract method for forming a batch which minimizes the maximum distance to a cluster center among all
        unlabeled datapoint
        """
        return

    def select_batch(self, **kwargs):
        """Intended to return points which minimize cluster centers"""
        return self.select_batch_(**kwargs)
