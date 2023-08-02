"""PyTorch model for the PatchCore model implementation."""

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
import logging
from typing import Optional, Union

import torch
from torch import Tensor
from anomalib.models.patchcore.torch_model import PatchcoreModel

try:
    import faiss

    # Importing this will add the bindings between faiss and torch
    from faiss.contrib import torch_utils

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

import numpy as np

log = logging.getLogger(__name__)


# This is just an experimental model it is not designed to work as it is hardly exportable


class FaissNN(object):
    def __init__(self, num_workers: int = 4) -> None:
        """FAISS Nearest neighbourhood search.
        Args:
            num_workers: Number of workers to use with FAISS for similarity search.
        """
        if not FAISS_AVAILABLE:
            raise ImportError("Faiss is not installed, if you are using a linux based machine install faiss-gpu==1.7.2")
        faiss.omp_set_num_threads(num_workers)
        self.search_index = None
        self.fit_on_gpu = False

    def _gpu_cloner_options(self):
        return faiss.GpuClonerOptions()

    def _index_to_gpu(self, index, device: torch.device):
        # For the non-gpu faiss python package, there is no GpuClonerOptions
        # so we can not make a default in the function header.
        return faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), device.index, index, self._gpu_cloner_options())

    def _index_to_cpu(self, index):
        if self.fit_on_gpu:
            return faiss.index_gpu_to_cpu(index)
        return index

    def _create_index(self, dimension, device=None):
        index = faiss.IndexFlatL2(dimension)
        if device is None or device.type == "cpu":
            return index
        else:
            self.fit_on_gpu = True
            return self._index_to_gpu(index, device)

    def fit(self, features: torch.Tensor) -> None:
        """
        Adds features to the FAISS search index.
        Args:
            features: Array of size NxD.
        """
        if self.search_index:
            self.reset_index()
        self.search_index = self._create_index(features.shape[-1], features.device)
        self._train(self.search_index, features)
        self.search_index.add(features)

    def _train(self, _index, _features):
        pass

    def run(
        self,
        n_nearest_neighbours,
        query_features: np.ndarray,
        index_features: np.ndarray = None,
    ) -> Union[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns distances and indices of nearest neighbour search.
        Args:
            query_features: Features to retrieve.
            index_features: [optional] Index features to search in.
        """
        if index_features is None:
            return self.search_index.search(query_features, n_nearest_neighbours)

        # Build a search index just for this search.
        search_index = self._create_index(index_features.shape[-1], index_features.device)
        self._train(search_index, index_features)
        search_index.add(index_features)
        return search_index.search(query_features, n_nearest_neighbours)

    def save(self, filename: str) -> None:
        faiss.write_index(self._index_to_cpu(self.search_index), filename)

    def load(self, filename: str, device: torch.device) -> None:
        self.search_index = self._index_to_gpu(faiss.read_index(filename), device)

    def reset_index(self):
        if self.search_index:
            self.search_index.reset()
            self.search_index = None


class FaissPatchcoreModel(PatchcoreModel):
    """Patchcore Module with FAISS implementation, this cannot be jitted so be careful.
    Moreover the FAISS index doesn't work with half precision.
    """

    def __init__(
        self,
        **kwargs,
    ) -> None:
        if not FAISS_AVAILABLE:
            raise ImportError("Faiss is not installed, if you are using a linux based machine install faiss-gpu==1.7.2")

        super().__init__(**kwargs)
        self.faiss_nn = FaissNN(num_workers=8)

    def fit_faiss(self):
        """Fit the faiss index with the current memory bank."""
        log.info("Fitting FAISS index")
        if self.memory_bank.dtype == torch.float16:
            raise ValueError("At the current time FAISS does not support half precision training")

        self.faiss_nn.fit(self.memory_bank)

    def subsample_embedding(self, embedding: torch.Tensor, sampling_ratio: float, mode: str = "anomalib") -> None:
        """Subsample the embedding to a given ratio and fit the faiss index."""
        super().subsample_embedding(embedding, sampling_ratio, mode)
        self.fit_faiss()

    def nearest_neighbors(self, embedding: Tensor, n_neighbors: torch.Tensor) -> Tensor:
        """Nearest Neighbours using faiss index.

        Args:
            embedding (Tensor): Features to compare the distance with the memory bank.
            n_neighbors (int): Number of neighbors to look at

        Returns:
            Tensor: Patch scores.
        """
        patch_scores, _ = self.faiss_nn.run(n_neighbors.item(), embedding.contiguous())
        patch_scores = torch.sqrt(patch_scores)
        return patch_scores
