# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""PyTorch model for the SuperADD model implementation.

This module implements a PatchCore-style anomaly detector built on a DINOv3
backbone. Multi-layer ViT token features are extracted over overlapping image
patches, a per-layer memory bank is built from normal training images via
distance-based coreset subsampling, and test images are scored by 1-nearest-
neighbor distance to the bank.

The model stores representative patch features from normal training images and
detects anomalies by comparing test image patches against this memory bank using
nearest neighbor search.

See Also:
    - :class:`anomalib.models.image.super_add.lightning_model.SuperADD`:
        Lightning implementation of the SuperADD model
"""

import gc
import itertools
import logging
import math
from collections import defaultdict

import timm
import torch
from torch import nn
from tqdm import tqdm

from anomalib.data import InferenceBatch
from anomalib.models.components import DynamicBufferMixin, GaussianBlur2d

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE = 1024

DINO_TARGET_LAYERS = {
    "small": [3, 5, 8, 10],
    "base": [3, 5, 8, 10],
    "large": [5, 11, 17, 23],
    "huge": [7, 15, 23, 31],
}


class DinoV3Backbone(nn.Module):
    """DINOv3 Vision Transformer feature extractor.

    Wraps a pretrained timm DINOv3 model and exposes the token features of a
    selected set of intermediate transformer blocks. The backbone is created in
    evaluation mode and run without classification head.

    Args:
        backbone (str): Name of the timm DINOv3 model to instantiate.
        layers (list[int]): Indices of the intermediate transformer blocks whose
            token features should be returned.

    Attributes:
        dino (nn.Module): The underlying pretrained timm model.
        layers (list[int]): Indices of the extracted intermediate blocks.
        model_patch_size (int): Patch size of the ViT (pixels per token).
    """

    def __init__(self, backbone: str, layers: list[int]) -> None:
        super().__init__()

        self.dino = timm.create_model(backbone, pretrained=True, num_classes=0).eval()
        self.layers = layers
        self.model_patch_size = self.dino.patch_embed.patch_size[0]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Extract intermediate-layer token features.

        Args:
            x (torch.Tensor): Input image batch of shape ``(b, c, h, w)``.

        Returns:
            list[torch.Tensor]: One tensor per requested layer, each in ``NLC``
                format (batch, tokens, channels).
        """
        with torch.inference_mode():
            return self.dino.forward_intermediates(
                x,
                indices=self.layers,
                norm=False,
                output_fmt="NLC",
                intermediates_only=True,
            )


class PatchedExecution(nn.Module):
    """Run a backbone on large images via overlapping patches.

    The input image is split into a grid of overlapping square patches, each
    patch is processed independently by the wrapped backbone, and the resulting
    per-patch token grids are stitched back into full-resolution token maps. The
    overlap regions are split between neighboring patches so that each output
    token is taken from the patch in which it is most centered, avoiding border
    artifacts.

    Args:
        model (nn.Module): Backbone applied to each patch. Expected to return a
            list of token tensors (one per layer) in ``NLC`` format.
        patch_size (int): Side length (in pixels) of each square patch.
        patch_overlap (int): Overlap (in pixels) between neighboring patches.
        model_patch_size (int): Patch size of the backbone (pixels per token).
            ``patch_size`` and ``patch_overlap`` must both be multiples of this.
    """

    def __init__(self, model: nn.Module, patch_size: int, patch_overlap: int, model_patch_size: int) -> None:
        super().__init__()
        assert patch_overlap > 0
        assert patch_size > 2 * patch_overlap
        assert patch_size % model_patch_size == 0
        assert patch_overlap % model_patch_size == 0

        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        self.model_patch_size = model_patch_size
        self.model = model

    def axis_patch_split(self, dim_size: int) -> tuple[list, list, list]:
        """Compute the patch layout along a single image axis.

        Determines how one axis of length ``dim_size`` is covered by overlapping
        patches and returns, in backbone-token units, the regions used to slice
        the input, to keep from each patch's prediction, and to write into the
        stitched output. The input ROIs are returned in pixels.

        Args:
            dim_size (int): Length of the image axis in pixels. Must be at least
                ``patch_size``.

        Returns:
            tuple[list, list, list]: ``(input_rois, prediction_rois,
                result_rois)``. ``input_rois`` are ``(start, end)`` pixel
                ranges used to crop input patches; ``prediction_rois`` are the
                ``(start, end)`` token ranges to keep from each patch output;
                ``result_rois`` are the ``(start, end)`` token ranges they map
                to in the stitched result.
        """
        assert dim_size >= self.patch_size

        dim_size_t = dim_size // self.model_patch_size
        overlap_t = self.patch_overlap // self.model_patch_size
        patch_size_t = self.patch_size // self.model_patch_size

        n_patches = math.ceil((dim_size_t - patch_size_t) / (patch_size_t - 2 * overlap_t)) + 1

        fac = dim_size_t - patch_size_t
        div = max(1, n_patches - 1)

        input_rois = []
        for i in range(n_patches):
            patch_min = (i * fac) // div
            patch_max = patch_min + patch_size_t
            input_rois.append((patch_min, patch_max))

        prediction_rois = []
        for i in range(n_patches):
            start = 0 if i == 0 else math.ceil((input_rois[i - 1][1] - input_rois[i][0]) / 2)
            end = (
                patch_size_t
                if i == n_patches - 1
                else patch_size_t - math.floor((input_rois[i][1] - input_rois[i + 1][0]) / 2)
            )
            prediction_rois.append((start, end))

        result_rois = [(0, prediction_rois[0][1] - prediction_rois[0][0])]
        for i in range(1, n_patches):
            start = result_rois[i - 1][1]
            end = start + prediction_rois[i][1] - prediction_rois[i][0]
            result_rois.append((start, end))

        input_rois = [(s * self.model_patch_size, e * self.model_patch_size) for (s, e) in input_rois]

        return input_rois, prediction_rois, result_rois

    def forward(self, x: torch.Tensor) -> list:
        """Run the backbone over overlapping patches and stitch the outputs.

        Splits the input into overlapping patches, runs the backbone on the full
        batch of patches, and reassembles the per-layer token grids into
        full-resolution token maps.

        Args:
            x (torch.Tensor): Input image batch of shape ``(b, c, h, w)``.

        Returns:
            list[torch.Tensor]: One stitched token map per backbone layer, each
                of shape ``(b, h // model_patch_size, w // model_patch_size,
                feature_count)``.
        """
        assert len(x.shape) == 4

        b, c, h, w = x.shape
        patch_w, patch_h = self.patch_size, self.patch_size

        input_rois_y, prediction_rois_y, result_rois_y = self.axis_patch_split(h)
        input_rois_x, prediction_rois_x, result_rois_x = self.axis_patch_split(w)
        x_overlapped = torch.zeros(
            (b, len(input_rois_y) * len(input_rois_x), c, patch_h, patch_w),
            device=x.device,
            dtype=x.dtype,
        )

        for i, ((y_start, y_end), (x_start, x_end)) in enumerate(itertools.product(input_rois_y, input_rois_x)):
            x_overlapped[:, i] = x[:, :, y_start:y_end, x_start:x_end]

        x_overlapped = x_overlapped.reshape(-1, c, patch_h, patch_w)
        with torch.no_grad():
            prediction = self.model(x_overlapped)
        tokens_y, tokens_x = patch_h // self.model_patch_size, patch_w // self.model_patch_size

        prediction = torch.stack(prediction)
        vector_count, _, _, feature_count = prediction.shape

        prediction = prediction.reshape(vector_count, b, -1, tokens_y, tokens_x, feature_count)

        result = torch.zeros(
            (vector_count, b, h // self.model_patch_size, w // self.model_patch_size, feature_count),
            device=prediction.device,
            dtype=x.dtype,
        )

        prediction_rois = itertools.product(prediction_rois_y, prediction_rois_x)
        result_rois = itertools.product(result_rois_y, result_rois_x)
        for i, ((pred_roi_y, pred_roi_x), (res_roi_y, res_roi_x)) in enumerate(
            zip(prediction_rois, result_rois, strict=True),
        ):
            p = prediction[:, :, i, pred_roi_y[0] : pred_roi_y[1], pred_roi_x[0] : pred_roi_x[1]]
            result[:, :, res_roi_y[0] : res_roi_y[1], res_roi_x[0] : res_roi_x[1]] = p

        return list(result)


class SuperADDModel(DynamicBufferMixin, nn.Module):
    """SuperADD PyTorch model for anomaly detection.

    This model implements the SuperADD algorithm, a PatchCore-style detector
    built on a pretrained DINOv3 backbone. Multi-layer ViT token features are
    extracted over overlapping image patches, a per-layer memory bank is built
    from normal training images via distance-based coreset subsampling, and test
    images are scored by nearest-neighbor distance to the bank.

    The model works in two phases:
    1. Training: Extract patch features from normal training images and store
       them, then build a coreset memory bank via :meth:`subsample_embedding`.
    2. Inference: Compare test image patches against the memory bank using
       nearest neighbor search to produce an anomaly map and score.

    Args:
        backbone (str): Name of the timm DINOv3 backbone used for feature
            extraction. Defaults to ``"vit_small_patch16_dinov3"``.
        patch_size (int): Side length (in pixels) of the overlapping patches the
            input image is split into. Defaults to ``448``.
        patch_overlap (int): Overlap (in pixels) between neighboring patches.
            Defaults to ``16``.
        layers (list[int] | None): List of encoder layer indices to extract features from.
            If None, DINO_TARGET_LAYERS of the respective model size.
        max_database_size (int): Target number of features retained per layer
            after coreset subsampling. Defaults to ``100000``.
        subsampling_iterations (int): Number of random-subset iterations used by
            the fast subsampling routine. Defaults to ``100``.
        gaussian_blur_sigma (float): Standard deviation of the Gaussian filter
            applied to the anomaly map before scoring. Suppresses single-token
            outliers that would otherwise dominate the image score and the
            adaptive threshold. Set to ``0`` to disable. Defaults to ``4.0``.
        score_quantile (float): Fraction of the highest anomaly map pixels
            averaged into the image-level score. Using a small quantile instead
            of the single maximum keeps the score independent of the input
            resolution (number of tokens). Defaults to ``1e-3``.

    Attributes:
        backbone (DinoV3Backbone): DINOv3 feature extractor.
        patch_exec (PatchedExecution): Overlapping-patch inference wrapper.
        layers (list[int]): Indices of the extracted backbone layers.
        memory_bank (torch.Tensor): Per-layer coreset of normal patch features.

    Note:
        The model requires no optimization/backpropagation as it uses a
        pretrained backbone and nearest neighbor search.

    See Also:
        - :class:`anomalib.models.image.super_add.lightning_model.SuperADD`:
            Lightning implementation of the SuperADD model
    """

    def __init__(
        self,
        backbone: str = "vit_huge_plus_patch16_dinov3",
        layers: list[int] | None = None,
        patch_size: int = 448,
        patch_overlap: int = 16,
        max_database_size: int = 100000,
        subsampling_iterations: int = 100,
        gaussian_blur_sigma: float = 4.0,
        score_quantile: float = 1e-3,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone

        if layers is None:
            for arch_name, target_layers in DINO_TARGET_LAYERS.items():
                if arch_name in backbone:
                    self.layers = target_layers
                    break
            if not hasattr(self, "layers"):
                msg = (
                    f"Could not infer target layers for backbone '{backbone}'. "
                    f"Known backbone keys: {sorted(DINO_TARGET_LAYERS.keys())}. "
                    "Please pass `layers` explicitly."
                )
                raise ValueError(msg)
        else:
            if not layers:
                msg = "`layers` must be a non-empty list of encoder layer indices."
                raise ValueError(msg)
            self.layers = layers

        self.backbone = DinoV3Backbone(backbone, self.layers)
        self.patch_exec = PatchedExecution(self.backbone, patch_size, patch_overlap, self.backbone.model_patch_size)
        self.max_database_size = max_database_size
        self.subsampling_iterations = subsampling_iterations
        self.score_quantile = score_quantile
        self.blur: GaussianBlur2d | None = None
        if gaussian_blur_sigma > 0:
            kernel_size = 2 * int(4.0 * gaussian_blur_sigma + 0.5) + 1
            self.blur = GaussianBlur2d(
                kernel_size=(kernel_size, kernel_size),
                sigma=(gaussian_blur_sigma, gaussian_blur_sigma),
                channels=1,
            )

        self.threshold = 0

        self.embedding_store: dict[int, list[torch.Tensor]] = defaultdict(list)
        self.memory_bank: torch.Tensor

        self.register_buffer("memory_bank", torch.empty(0))

    @staticmethod
    def __clear_cache() -> None:
        torch.cuda.empty_cache()
        gc.collect()

    def _compute_device(self) -> torch.device:
        return next(self.backbone.parameters()).device

    def subsample_embedding(self) -> None:
        """Build the memory bank by coreset-subsampling the stored embeddings.

        For each extracted layer, the patch features collected during training
        (kept on CPU) are concatenated, reduced to at most ``max_database_size``
        representative features via distance-based subsampling, and written into
        the corresponding row of ``memory_bank``. The per-layer embedding store
        is cleared afterward to free memory.
        """
        device = self._compute_device()
        for layer_idx, _ in enumerate(self.layers):
            # Embeddings live on CPU (offloaded during training); concatenate and
            # subsample there, then move the (small) coreset to the compute device.
            embeddings = torch.cat(self.embedding_store[layer_idx], dim=0)

            embeddings = self.subsampling_distance_based_fast(
                embeddings,
                self.max_database_size,
                iterations=self.subsampling_iterations,
                normalize=False,
                knn_neighbors=100,
            )
            embeddings = embeddings.to(device)
            if self.memory_bank.numel() == 0:
                self.memory_bank = torch.empty(
                    len(self.layers),
                    embeddings.shape[0],
                    embeddings.shape[1],
                    dtype=embeddings.dtype,
                    device=device,
                )

            self.memory_bank[layer_idx] = embeddings
            self.embedding_store[layer_idx] = []
            del embeddings
        del self.embedding_store
        self.__clear_cache()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor | InferenceBatch:
        """Extract features during training or compute anomaly scores at test time.

        During training, multi-layer patch features are extracted and appended to
        the per-layer embedding store (on CPU) for later coreset construction. At
        inference time, the per-layer features are compared against the memory
        bank using chunked nearest-neighbor search; the resulting per-layer
        distance maps are normalized, interpolated to the input resolution,
        averaged into a single anomaly map and smoothed with a Gaussian filter.
        The mean of the top-quantile pixels is used as the image-level score.

        Args:
            input_tensor (torch.Tensor): Input image batch of shape
                ``(b, c, h, w)``.

        Returns:
            torch.Tensor | InferenceBatch: During training, the last layer's
                patch embedding (only used to satisfy the Lightning training
                step). During inference, an :class:`InferenceBatch` containing
                the anomaly map and predicted scores.
        """
        input_tensor = input_tensor.type(self.memory_bank.dtype)
        if self.training:
            prediction = self.patch_exec(input_tensor)
            for layer_idx, (_, embedding) in enumerate(zip(self.layers, prediction, strict=False)):
                embedding_reshaped = embedding.reshape(-1, embedding.shape[-1])
                # Synchronous copy: subsample_embedding() reads embedding_store without a
                # device sync, so a non_blocking copy here can leave the last batch stale/NaN.
                self.embedding_store[layer_idx].append(embedding_reshaped.to("cpu"))

            return embedding_reshaped  # Return the embedding for the training step (not used for loss computation)

        input_shape = input_tensor.shape
        predicted_embeddings = self.patch_exec(input_tensor)

        layer_distances = []
        for layer_idx, (_, predicted_embedding) in enumerate(
            zip(self.layers, predicted_embeddings, strict=False),
        ):
            b, h, w, c = predicted_embedding.shape
            query = predicted_embedding.reshape(b, h * w, c)
            keys = self.memory_bank[layer_idx]
            # Chunk over the query (token) dimension so we never materialize the
            # full [b, h*w, N_bank] distance matrix at once.
            dists, _ = self.nearest_neighbors_chunked(query, keys, knn_neighbors=1)
            dists = dists.mean(dim=-1)
            dists = dists.reshape(b, h, w) / c  # normalize distance by dimsize to account for different dimsizes
            layer_distances.append(dists)
        layer_distances = torch.stack(layer_distances, dim=1)
        layer_distances = torch.nn.functional.interpolate(
            layer_distances,
            size=(input_shape[-2], input_shape[-1]),
            mode="bilinear",
            align_corners=False,
        )
        anomaly_map = torch.mean(layer_distances, dim=1, keepdim=True)
        if self.blur is not None:
            anomaly_map = self.blur(anomaly_map)
        anomaly_map = anomaly_map.squeeze(1)

        # Image score: mean of the top-quantile pixels. Robust to single-token
        # outliers and independent of the number of tokens, unlike a plain max.
        pixel_scores = anomaly_map.flatten(start_dim=1)
        top_k = max(1, int(pixel_scores.shape[1] * self.score_quantile))
        pred_score = pixel_scores.topk(top_k, dim=1).values.mean(dim=1)

        return InferenceBatch(pred_score=pred_score, anomaly_map=anomaly_map)

    def subsample_features(
        self,
        x: torch.Tensor,
        target_number_of_samples: int,
        knn_neighbors: int,
        normalize: bool,
    ) -> torch.Tensor:
        """Select a subset of features that are roughly evenly spaced.

        Computes the nearest-neighbor distances within ``x`` and probabilistically
        keeps each feature with a probability inversely proportional to how many
        neighbors fall within a growing distance threshold. The threshold is
        increased until at most ``target_number_of_samples`` features remain.

        Args:
            x (torch.Tensor): Candidate features of shape ``(n, d)``.
            target_number_of_samples (int): Maximum number of features to keep.
            knn_neighbors (int): Number of neighbors used for the density
                estimate.
            normalize (bool): Whether to L2-normalize before computing distances.

        Returns:
            torch.Tensor: Boolean keep-mask of shape ``(n,)`` on the CPU.
        """
        device = self._compute_device()
        x = x.to(device)
        knn_neighbors = min(knn_neighbors, len(x))  # subsets can be smaller than the neighbor count
        dists, _ = self.nearest_neighbors(x, x, knn_neighbors=knn_neighbors, normalize=normalize)
        # Start with a small distance and increase until we have fewer than the target number of samples
        target_distance_between_samples = dists.float().mean() / 10
        number_of_samples = target_number_of_samples + 1  # Initialize > target to enter the loop
        random_numbers = torch.rand(len(x), device=device)
        keep_mask = torch.zeros(len(x), dtype=torch.bool, device=device)

        while number_of_samples > target_number_of_samples:
            subsampling_factor = (dists < target_distance_between_samples).sum(dim=-1) + 1
            keep_mask = random_numbers < (1.0 / subsampling_factor.float())
            number_of_samples = int(keep_mask.sum())
            target_distance_between_samples *= 1.1  # Increase distance if still too many samples

        return keep_mask.cpu()

    def subsampling_distance_based_fast(
        self,
        features: torch.Tensor,
        target_number_of_samples: int,
        iterations: int = 100,
        normalize: bool = False,
        knn_neighbors: int = 100,
    ) -> torch.Tensor:
        """Distance-based coreset subsampling over random subsets.

        Repeatedly draws random subsets of the not-yet-kept features and applies
        :meth:`subsample_features` to each, accumulating a global keep-mask. This
        keeps the nearest-neighbor search tractable on large feature sets. If
        fewer than ``target_number_of_samples`` features are kept, additional
        features are randomly added back to reach the target.

        Args:
            features (torch.Tensor): Features to subsample, of shape ``(n, d)``.
            target_number_of_samples (int): Desired number of retained features.
            iterations (int): Number of random-subset iterations. Defaults to
                ``100``.
            normalize (bool): Whether to L2-normalize before computing distances.
                Defaults to ``False``.
            knn_neighbors (int): Number of neighbors used for the density
                estimate. Defaults to ``100``.

        Returns:
            torch.Tensor: The subsampled features of shape
                ``(target_number_of_samples, d)``.
        """
        # perform subsample iteratively on random subsets of the data to speed up the nearest neighbor search
        if iterations <= 0:
            msg = f"`iterations` must be a positive integer, got {iterations}."
            raise ValueError(msg)
        if target_number_of_samples <= 0:
            msg = f"`target_number_of_samples` must be positive, got {target_number_of_samples}."
            raise ValueError(msg)

        # If we already have fewer features than requested, return as-is.
        if len(features) <= target_number_of_samples:
            return features

        keep_mask_total = torch.zeros(len(features), dtype=torch.bool)
        # Guarantee at least 1 to avoid zero-sized subsets / targets on small inputs.
        size_of_subsets = max(1, len(features) // iterations)
        target_to_keep_subset = max(1, target_number_of_samples // iterations)
        number_of_samples = 0

        for _ in tqdm(range(iterations), desc="Subsampling embeddings"):
            candidate_indices = torch.where(~keep_mask_total)[0]
            if len(candidate_indices) == 0:
                break
            # Randomly select a subset of candidates for this iteration
            n = min(size_of_subsets, len(candidate_indices))
            perm = torch.randperm(len(candidate_indices))[:n]
            indices = candidate_indices[perm]

            keep_mask_subset = self.subsample_features(
                features[indices],
                target_to_keep_subset,
                knn_neighbors,
                normalize,
            )

            keep_mask_total[indices] = keep_mask_subset  # Update total keep mask with this subset's results
            number_of_samples = int(keep_mask_total.sum())

        difference = target_number_of_samples - number_of_samples

        # Randomly select some unkept samples to add back until we reach the target number of samples
        if difference > 0:
            indices_to_add_back = torch.where(~keep_mask_total)[0]
            shuffle = torch.randperm(len(indices_to_add_back))
            indices_to_add_back = indices_to_add_back[shuffle]
            keep_mask_total[indices_to_add_back[:difference]] = True

        return features[keep_mask_total]

    def nearest_neighbors(
        self,
        features_query: torch.Tensor,
        features_key: torch.Tensor,
        knn_neighbors: int,
        normalize: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Find the ``k`` nearest keys for each query by Euclidean distance.

        Computes the full pairwise distance matrix between queries and keys using
        :meth:`euclidean_dist` and returns the ``knn_neighbors`` smallest
        distances and their indices.

        Args:
            features_query (torch.Tensor): Query features of shape ``(..., n, d)``.
            features_key (torch.Tensor): Key features of shape ``(..., m, d)``.
            knn_neighbors (int): Number of nearest neighbors to return.
            normalize (bool): Whether to L2-normalize both inputs before
                computing distances. Defaults to ``False``.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: ``(distances, indices)``, each of
                shape ``(..., n, knn_neighbors)``.
        """
        if normalize:
            features_query = torch.nn.functional.normalize(features_query, dim=-1)
            features_key = torch.nn.functional.normalize(features_key, dim=-1)

        dists_full = self.euclidean_dist(features_query, features_key)

        topk_vals, topk_idx = torch.topk(
            dists_full,
            k=knn_neighbors,
            dim=-1,
            largest=False,  # Set largest=False to get smallest distances
            sorted=False,
        )
        return topk_vals, topk_idx

    @staticmethod
    def euclidean_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute pairwise Euclidean distances between two sets of vectors.

        Implements an efficient matrix computation of Euclidean distances between
        all pairs of vectors in ``x`` and ``y`` without using ``torch.cdist()``.

        Args:
            x (torch.Tensor): First tensor of shape ``(n, d)``.
            y (torch.Tensor): Second tensor of shape ``(m, d)``.

        Returns:
            torch.Tensor: Distance matrix of shape ``(n, m)`` where element
                ``(i,j)`` is the distance between row ``i`` of ``x`` and row
                ``j`` of ``y``.

        Example:
            >>> x = torch.randn(100, 512)
            >>> y = torch.randn(50, 512)
            >>> distances = SuperADDModel.euclidean_dist(x, y)
            >>> distances.shape
            torch.Size([100, 50])

        Note:
            This implementation avoids using ``torch.cdist()`` for better
            compatibility with ONNX export and OpenVINO conversion.
        """
        x_norm = x.pow(2).sum(dim=-1, keepdim=True)
        y_norm = y.pow(2).sum(dim=-1, keepdim=True)
        res = torch.matmul(x, y.transpose(-2, -1))
        res.mul_(-2)
        res.add_(x_norm)
        res.add_(y_norm.transpose(-2, -1))
        return res.clamp_min_(0).sqrt_()

    def nearest_neighbors_chunked(
        self,
        features_query: torch.Tensor,
        features_key: torch.Tensor,
        knn_neighbors: int,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        normalize: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Chunked nearest-neighbor search over the query (token) dimension.

        Produces the same result as :meth:`nearest_neighbors` but processes the
        queries in chunks so the full ``[b, n_query, n_key]`` distance matrix is
        never materialized at once. Each distance block is reduced to its top-k
        and discarded, keeping peak memory bounded by the key-bank size rather
        than the number of query tokens.

        Args:
            features_query (torch.Tensor): Query features of shape ``(b, n_query, c)``.
            features_key (torch.Tensor): Key features of shape ``(n_key, c)``.
            knn_neighbors (int): Number of nearest neighbors to return.
            chunk_size (int): Max number of query tokens processed per block.
            normalize (bool): Whether to L2-normalize before the distance computation.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: ``(distances, indices)`` each of
                shape ``(b, n_query, knn_neighbors)``.
        """
        n_query = features_query.shape[1]
        if n_query <= chunk_size:
            return self.nearest_neighbors(
                features_query,
                features_key,
                knn_neighbors=knn_neighbors,
                normalize=normalize,
            )

        vals: list[torch.Tensor] = []
        idxs: list[torch.Tensor] = []
        for start in range(0, n_query, chunk_size):
            end = min(start + chunk_size, n_query)
            chunk_vals, chunk_idx = self.nearest_neighbors(
                features_query[:, start:end],
                features_key,
                knn_neighbors=knn_neighbors,
                normalize=normalize,
            )
            vals.append(chunk_vals)
            idxs.append(chunk_idx)
        return torch.cat(vals, dim=1), torch.cat(idxs, dim=1)
