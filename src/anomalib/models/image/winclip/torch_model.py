"""PyTorch model implementation of WinCLIP for zero-/few-shot anomaly detection.

This module provides the core PyTorch model implementation of WinCLIP, which uses
CLIP embeddings and a sliding window approach to detect anomalies in images.

The model can operate in both zero-shot and few-shot modes:
- Zero-shot: No reference images needed
- Few-shot: Uses ``k`` reference normal images for better context

Example:
    >>> from anomalib.models.image.winclip.torch_model import WinClipModel
    >>> model = WinClipModel()  # doctest: +SKIP
    >>> # Zero-shot inference
    >>> prediction = model(image)  # doctest: +SKIP
    >>> # Few-shot with reference images
    >>> model = WinClipModel(reference_images=normal_images)  # doctest: +SKIP

Paper:
    WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation
    https://arxiv.org/abs/2303.14814

See Also:
    - :class:`WinClip`: Lightning model wrapper
    - :mod:`.prompting`: Prompt ensemble generation
    - :mod:`.utils`: Utility functions for scoring and aggregation
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from copy import copy

import open_clip
import torch
from open_clip.tokenizer import tokenize
from torch import nn
from torch.nn.modules.linear import Identity
from torchvision.transforms import Compose, ToPILImage

from anomalib.data import InferenceBatch
from anomalib.models.components import BufferListMixin, DynamicBufferMixin

from .prompting import create_prompt_ensemble
from .utils import class_scores, harmonic_aggregation, make_masks, visual_association_score

BACKBONE = "ViT-B-16-plus-240"
PRETRAINED = "laion400m_e31"
TEMPERATURE = 0.07  # temperature hyperparameter from the clip paper


class WinClipModel(DynamicBufferMixin, BufferListMixin, nn.Module):
    """PyTorch module that implements the WinClip model for image anomaly detection.

    The model uses CLIP embeddings and a sliding window approach to detect anomalies in
    images. It can operate in both zero-shot and few-shot modes.

    Args:
        class_name (str | None, optional): Name of the object class used in prompt
            ensemble. Defaults to ``None``.
        reference_images (torch.Tensor | None, optional): Reference images of shape
            ``(K, C, H, W)``. Defaults to ``None``.
        scales (tuple[int], optional): Scales of sliding windows for multi-scale
            detection. Defaults to ``(2, 3)``.
        apply_transform (bool, optional): Whether to apply default CLIP transform to
            inputs. Defaults to ``False``.

    Attributes:
        clip (CLIP): CLIP model for image and text encoding.
        grid_size (tuple[int]): Size of feature map grid.
        k_shot (int): Number of reference images for few-shot detection.
        scales (tuple[int]): Scales of sliding windows.
        masks (list[torch.Tensor] | None): Masks for sliding window locations.
        _text_embeddings (torch.Tensor | None): Text embeddings for prompt ensemble.
        _visual_embeddings (list[torch.Tensor] | None): Multi-scale reference embeddings.
        _patch_embeddings (torch.Tensor | None): Patch embeddings for reference images.

    Example:
        >>> from anomalib.models.image.winclip.torch_model import WinClipModel
        >>> # Zero-shot mode
        >>> model = WinClipModel(class_name="transistor")  # doctest: +SKIP
        >>> image = torch.rand(1, 3, 224, 224)  # doctest: +SKIP
        >>> prediction = model(image)  # doctest: +SKIP
        >>>
        >>> # Few-shot mode with reference images
        >>> ref_images = torch.rand(3, 3, 224, 224)  # doctest: +SKIP
        >>> model = WinClipModel(  # doctest: +SKIP
        ...     class_name="transistor",
        ...     reference_images=ref_images
        ... )
    """

    def __init__(
        self,
        class_name: str | None = None,
        reference_images: torch.Tensor | None = None,
        scales: tuple = (2, 3),
        apply_transform: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = BACKBONE
        self.pretrained = PRETRAINED
        self.temperature = TEMPERATURE
        self.class_name = class_name
        self.reference_images = reference_images
        self.scales = scales
        self.apply_transform = apply_transform
        self.k_shot = 0

        # initialize CLIP model
        self.clip, _, self._transform = open_clip.create_model_and_transforms(self.backbone, pretrained=self.pretrained)
        self.clip.visual.output_tokens = True
        self.grid_size = self.clip.visual.grid_size

        # register buffers
        self.register_buffer_list("masks", self._generate_masks(), persistent=False)  # no need to save masks
        self.register_buffer("_text_embeddings", torch.empty(0))
        self.register_buffer_list("_visual_embeddings", [torch.empty(0) for _ in self.scales])
        self.register_buffer("_patch_embeddings", torch.empty(0))

        # setup
        self.setup(class_name, reference_images)

    def setup(self, class_name: str | None = None, reference_images: torch.Tensor | None = None) -> None:
        """Setup WinCLIP model with class name and/or reference images.

        The setup stage collects text and visual embeddings used during inference:
        - Text embeddings for zero-shot inference if ``class_name`` provided
        - Visual embeddings for few-shot inference if ``reference_images`` provided
        The ``k_shot`` attribute is updated based on number of reference images.

        This method is called by constructor but can also be called manually to update
        embeddings after initialization.

        Args:
            class_name (str | None, optional): Name of object class for prompt ensemble.
                Defaults to ``None``.
            reference_images (torch.Tensor | None, optional): Reference images of shape
                ``(batch_size, C, H, W)``. Defaults to ``None``.

        Examples:
            >>> model = WinClipModel()  # doctest: +SKIP
            >>> model.setup("transistor")  # doctest: +SKIP
            >>> model.text_embeddings.shape  # doctest: +SKIP
            torch.Size([2, 640])

            >>> ref_images = torch.rand(2, 3, 240, 240)  # doctest: +SKIP
            >>> model = WinClipModel()  # doctest: +SKIP
            >>> model.setup("transistor", ref_images)  # doctest: +SKIP
            >>> model.k_shot  # doctest: +SKIP
            2
            >>> model.visual_embeddings[0].shape  # doctest: +SKIP
            torch.Size([2, 196, 640])

            >>> model = WinClipModel("transistor")  # doctest: +SKIP
            >>> model.k_shot  # doctest: +SKIP
            0
            >>> model.setup(reference_images=ref_images)  # doctest: +SKIP
            >>> model.k_shot  # doctest: +SKIP
            2

            >>> model = WinClipModel(  # doctest: +SKIP
            ...     class_name="transistor",
            ...     reference_images=ref_images
            ... )
            >>> model.text_embeddings.shape  # doctest: +SKIP
            torch.Size([2, 640])
            >>> model.visual_embeddings[0].shape  # doctest: +SKIP
            torch.Size([2, 196, 640])
        """
        # update class name and text embeddings
        self.class_name = class_name or self.class_name
        if self.class_name is not None:
            self._collect_text_embeddings(self.class_name)
        # update reference images, k_shot and visual embeddings
        self.reference_images = reference_images if reference_images is not None else self.reference_images
        if self.reference_images is not None:
            self.k_shot = self.reference_images.shape[0]  # update k_shot based on number of reference images
            self._collect_visual_embeddings(self.reference_images)

    def encode_image(self, batch: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        """Encode batch of images to get image, window and patch embeddings.

        The image and patch embeddings are obtained by passing images through the model.
        Window embeddings are obtained by masking feature map and passing through
        transformer. A forward hook retrieves intermediate feature map to share
        computation.

        Args:
            batch (torch.Tensor): Input images of shape ``(N, C, H, W)``.

        Returns:
            tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]: Tuple containing:
                - Image embeddings of shape ``(N, D)``
                - Window embeddings list, each of shape ``(N, W, D)``
                - Patch embeddings of shape ``(N, P, D)``
                where ``D`` is embedding dimension, ``W`` is number of windows,
                and ``P`` is number of patches.

        Examples:
            >>> model = WinClipModel()  # doctest: +SKIP
            >>> model.prepare_masks()  # doctest: +SKIP
            >>> batch = torch.rand((1, 3, 240, 240))  # doctest: +SKIP
            >>> outputs = model.encode_image(batch)  # doctest: +SKIP
            >>> image_embeddings, window_embeddings, patch_embeddings = outputs
            >>> image_embeddings.shape  # doctest: +SKIP
            torch.Size([1, 640])
            >>> [emb.shape for emb in window_embeddings]  # doctest: +SKIP
            [torch.Size([1, 196, 640]), torch.Size([1, 169, 640])]
            >>> patch_embeddings.shape  # doctest: +SKIP
            torch.Size([1, 225, 896])
        """
        # apply transform if needed
        if self.apply_transform:
            batch = torch.stack([self.transform(image) for image in batch])

        # register hook to retrieve intermediate feature map
        outputs = {}

        def get_feature_map(name: str) -> Callable:
            def hook(_model: Identity, inputs: tuple[torch.Tensor,], _outputs: torch.Tensor) -> None:
                del _model, _outputs
                outputs[name] = inputs[0].detach()

            return hook

        # register hook to get the intermediate tokens of the transformer
        self.clip.visual.patch_dropout.register_forward_hook(get_feature_map("feature_map"))

        # get image and patch embeddings
        image_embeddings, patch_embeddings = self.clip.encode_image(batch)

        # get window embeddings
        feature_map = outputs["feature_map"]
        window_embeddings = [self._get_window_embeddings(feature_map, masks) for masks in self.masks]

        return (
            image_embeddings,
            window_embeddings,
            patch_embeddings,
        )

    def _get_window_embeddings(self, feature_map: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """Compute embeddings for each window in feature map using given masks.

        Args:
            feature_map (torch.Tensor): Input features of shape
                ``(n_batches, n_patches, dimensionality)``.
            masks (torch.Tensor): Window location masks of shape
                ``(kernel_size, n_masks)``.

        Returns:
            torch.Tensor: Embeddings for each sliding window location.
        """
        batch_size = feature_map.shape[0]
        n_masks = masks.shape[1]

        # prepend zero index for class embeddings
        class_index = torch.zeros(1, n_masks, dtype=int).to(feature_map.device)
        masks = torch.cat((class_index, masks + 1)).T  # +1 to account for class index
        # apply masks to feature map
        masked = torch.cat([torch.index_select(feature_map, 1, mask) for mask in masks])

        # finish forward pass on masked features
        masked = self.clip.visual.patch_dropout(masked)
        masked = self.clip.visual.ln_pre(masked)

        masked = masked.permute(1, 0, 2)  # NLD -> LND
        masked = self.clip.visual.transformer(masked)
        masked = masked.permute(1, 0, 2)  # LND -> NLD

        masked = self.clip.visual.ln_post(masked)
        pooled, _ = self.clip.visual._global_pool(masked)  # noqa: SLF001

        if self.clip.visual.proj is not None:
            pooled = pooled @ self.clip.visual.proj

        return pooled.reshape((n_masks, batch_size, -1)).permute(1, 0, 2)

    @torch.no_grad
    def forward(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | InferenceBatch:
        """Forward pass to get image and pixel anomaly scores.

        Args:
            batch (torch.Tensor): Input images of shape ``(batch_size, C, H, W)``.

        Returns:
            tuple[torch.Tensor, torch.Tensor] | InferenceBatch: Either tuple containing:
                - Image scores of shape ``(batch_size,)``
                - Pixel scores of shape ``(batch_size, H, W)``
                Or ``InferenceBatch`` with same information.
        """
        image_embeddings, window_embeddings, patch_embeddings = self.encode_image(batch)

        # get zero-shot scores
        image_scores = class_scores(image_embeddings, self.text_embeddings, self.temperature, target_class=1)
        multi_scale_scores = self._compute_zero_shot_scores(image_scores, window_embeddings)

        # get few-shot scores
        if self.k_shot:
            few_shot_scores = self._compute_few_shot_scores(patch_embeddings, window_embeddings)
            multi_scale_scores = (multi_scale_scores + few_shot_scores) / 2
            image_scores = (image_scores + few_shot_scores.amax(dim=(-2, -1))) / 2

        # reshape to image dimensions
        pixel_scores = nn.functional.interpolate(
            multi_scale_scores.unsqueeze(1),
            size=batch.shape[-2:],
            mode="bilinear",
        )
        return InferenceBatch(pred_score=image_scores, anomaly_map=pixel_scores.squeeze(1))

    def _compute_zero_shot_scores(
        self,
        image_scores: torch.Tensor,
        window_embeddings: list[torch.Tensor],
    ) -> torch.Tensor:
        """Compute multi-scale anomaly score maps using text embeddings.

        Each window embedding is compared to text embeddings for similarity scores.
        Harmonic averaging aggregates window scores into score maps per scale.
        Score maps are combined into single multi-scale map by cross-scale
        aggregation.

        Args:
            image_scores (torch.Tensor): Full image scores of shape ``(batch_size)``.
            window_embeddings (list[torch.Tensor]): Window embeddings list, each of
                shape ``(batch_size, n_windows, n_features)``.

        Returns:
            torch.Tensor: Zero-shot scores of shape ``(batch_size, H, W)``.
        """
        # image scores are added to represent the full image scale
        multi_scale_scores = [image_scores.view(-1, 1, 1).repeat(1, self.grid_size[0], self.grid_size[1])]
        # add aggregated scores for each scale
        for window_embedding, mask in zip(window_embeddings, self.masks, strict=True):
            scores = class_scores(window_embedding, self.text_embeddings, self.temperature, target_class=1)
            multi_scale_scores.append(harmonic_aggregation(scores, self.grid_size, mask))
        # aggregate scores across scales
        return (len(self.scales) + 1) / (1 / torch.stack(multi_scale_scores)).sum(dim=0)

    def _compute_few_shot_scores(
        self,
        patch_embeddings: torch.Tensor,
        window_embeddings: list[torch.Tensor],
    ) -> torch.Tensor:
        """Compute multi-scale anomaly score maps using reference embeddings.

        Visual association scores are computed between extracted embeddings and
        reference embeddings at each scale. Window scores are aggregated into score
        maps per scale using harmonic averaging. Final maps obtained by averaging
        across scales.

        Args:
            patch_embeddings (torch.Tensor): Full-scale patch embeddings of shape
                ``(batch_size, n_patches, n_features)``.
            window_embeddings (list[torch.Tensor]): Window embeddings list, each of
                shape ``(batch_size, n_windows, n_features)``.

        Returns:
            torch.Tensor: Few-shot scores of shape ``(batch_size, H, W)``.
        """
        multi_scale_scores = [
            visual_association_score(patch_embeddings, self.patch_embeddings).reshape((-1, *self.grid_size)),
        ]
        for window_embedding, reference_embedding, mask in zip(
            window_embeddings,
            self.visual_embeddings,
            self.masks,
            strict=True,
        ):
            scores = visual_association_score(window_embedding, reference_embedding)
            multi_scale_scores.append(harmonic_aggregation(scores, self.grid_size, mask))

        return torch.stack(multi_scale_scores).mean(dim=0)

    @torch.no_grad
    def _collect_text_embeddings(self, class_name: str) -> None:
        """Collect text embeddings using compositional prompt ensemble.

        Creates ensemble of normal and anomalous prompts based on class name.
        Prompts are tokenized and encoded to get embeddings. Embeddings are averaged
        per class and stored for inference.

        Args:
            class_name (str): Object class name for prompt ensemble.
        """
        # get the device, this is to ensure that we move the text embeddings to the same device as the model
        device = next(self.parameters()).device
        # collect prompt ensemble
        normal_prompts, anomalous_prompts = create_prompt_ensemble(class_name)
        # tokenize prompts
        normal_tokens = tokenize(normal_prompts)
        anomalous_tokens = tokenize(anomalous_prompts)
        # encode tokens to obtain prompt embeddings
        normal_embeddings = self.clip.encode_text(normal_tokens.to(device))
        anomalous_embeddings = self.clip.encode_text(anomalous_tokens.to(device))
        # average prompt embeddings
        normal_embeddings = torch.mean(normal_embeddings, dim=0, keepdim=True)
        anomalous_embeddings = torch.mean(anomalous_embeddings, dim=0, keepdim=True)
        # concatenate and store
        text_embeddings = torch.cat((normal_embeddings, anomalous_embeddings))
        self._text_embeddings = text_embeddings

    @torch.no_grad
    def _collect_visual_embeddings(self, images: torch.Tensor) -> None:
        """Collect visual embeddings from normal reference images.

        Args:
            images (torch.Tensor): Reference images of shape ``(K, C, H, W)``.
        """
        _, self._visual_embeddings, self._patch_embeddings = self.encode_image(images)

    def _generate_masks(self) -> list[torch.Tensor]:
        """Prepare multi-scale sliding window masks.

        Creates masks for each scale that select patches from feature map. Each mask
        represents a sliding window location. Masks are stored for inference.

        Returns:
            list[torch.Tensor]: List of masks, each of shape
            ``(n_patches_per_mask, n_masks)``.
        """
        return [make_masks(self.grid_size, scale, 1) for scale in self.scales]

    @property
    def transform(self) -> Compose:
        """Get model's transform pipeline.

        Retrieves transforms from CLIP backbone and prepends ``ToPILImage`` transform
        since original transforms expect PIL images.

        Returns:
            Compose: Transform pipeline for preprocessing images.
        """
        transforms = copy(self._transform.transforms)
        transforms.insert(0, ToPILImage())
        return Compose(transforms)

    @property
    def text_embeddings(self) -> torch.Tensor:
        """Get model's text embeddings.

        Returns:
            torch.Tensor: Text embeddings used for zero-shot inference.

        Raises:
            RuntimeError: If text embeddings not collected via ``setup``.
        """
        if self._text_embeddings.numel() == 0:
            msg = "Text embeddings have not been collected. Pass a class name to the model using ``setup``."
            raise RuntimeError(msg)
        return self._text_embeddings

    @property
    def visual_embeddings(self) -> list[torch.Tensor]:
        """Get model's visual embeddings.

        Returns:
            list[torch.Tensor]: Visual embeddings used for few-shot inference.

        Raises:
            RuntimeError: If visual embeddings not collected via ``setup``.
        """
        if self._visual_embeddings[0].numel() == 0:
            msg = "Visual embeddings have not been collected. Pass some reference images to the model using ``setup``."
            raise RuntimeError(msg)
        return self._visual_embeddings

    @property
    def patch_embeddings(self) -> torch.Tensor:
        """Get model's patch embeddings.

        Returns:
            torch.Tensor: Patch embeddings used for few-shot inference.

        Raises:
            RuntimeError: If patch embeddings not collected via ``setup``.
        """
        if self._patch_embeddings.numel() == 0:
            msg = "Patch embeddings have not been collected. Pass some reference images to the model using ``setup``."
            raise RuntimeError(msg)
        return self._patch_embeddings
