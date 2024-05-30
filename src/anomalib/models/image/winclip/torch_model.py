"""PyTorch model for the WinCLIP implementation."""

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

from anomalib.models.components import BufferListMixin, DynamicBufferMixin

from .prompting import create_prompt_ensemble
from .utils import class_scores, harmonic_aggregation, make_masks, visual_association_score

BACKBONE = "ViT-B-16-plus-240"
PRETRAINED = "laion400m_e31"
TEMPERATURE = 0.07  # temperature hyperparameter from the clip paper


class WinClipModel(DynamicBufferMixin, BufferListMixin, nn.Module):
    """PyTorch module that implements the WinClip model for image anomaly detection.

    Args:
        class_name (str, optional): The name of the object class used in the prompt ensemble.
            Defaults to ``None``.
        reference_images (torch.Tensor, optional): Tensor of shape ``(K, C, H, W)`` containing the reference images.
            Defaults to ``None``.
        scales (tuple[int], optional): The scales of the sliding windows used for multi-scale anomaly detection.
            Defaults to ``(2, 3)``.
        apply_transform (bool, optional): Whether to apply the default CLIP transform to the input images.
            Defaults to ``False``.

    Attributes:
        clip (CLIP): The CLIP model used for image and text encoding.
        grid_size (tuple[int]): The size of the feature map grid.
        k_shot (int): The number of reference images used for few-shot anomaly detection.
        scales (tuple[int]): The scales of the sliding windows used for multi-scale anomaly detection.
        masks (list[torch.Tensor] | None): The masks representing the sliding window locations.
        _text_embeddings (torch.Tensor | None): The text embeddings for the compositional prompt ensemble.
        _visual_embeddings (list[torch.Tensor] | None): The multi-scale embeddings for the reference images.
        _patch_embeddings (torch.Tensor | None): The patch embeddings for the reference images.
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
        """Setup WinCLIP.

        WinCLIP's setup stage consists of collecting the text and visual embeddings used during inference. The
        following steps are performed, depending on the arguments passed to the model:
        - Collect text embeddings for zero-shot inference.
        - Collect reference images for few-shot inference.
        The k_shot attribute is updated based on the number of reference images.

        The setup method is called internally by the constructor. However, it can also be called manually to update the
        text and visual embeddings after the model has been initialized.

        Args:
            class_name (str): The name of the object class used in the prompt ensemble.
            reference_images (torch.Tensor): Tensor of shape ``(batch_size, C, H, W)`` containing the reference images.

        Examples:
            >>> model = WinClipModel()
            >>> model.setup("transistor")
            >>> model.text_embeddings.shape
            torch.Size([2, 640])

            >>> ref_images = torch.rand(2, 3, 240, 240)
            >>> model = WinClipModel()
            >>> model.setup("transistor", ref_images)
            >>> model.k_shot
            2
            >>> model.visual_embeddings[0].shape
            torch.Size([2, 196, 640])

            >>> model = WinClipModel("transistor")
            >>> model.k_shot
            0
            >>> model.setup(reference_images=ref_images)
            >>> model.k_shot
            2

            >>> model = WinClipModel(class_name="transistor", reference_images=ref_images)
            >>> model.text_embeddings.shape
            torch.Size([2, 640])
            >>> model.visual_embeddings[0].shape
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
        """Encode the batch of images to obtain image embeddings, window embeddings, and patch embeddings.

        The image embeddings and patch embeddings are obtained by passing the batch of images through the model. The
        window embeddings are obtained by masking the feature map and passing it through the transformer. A forward hook
        is used to retrieve the intermediate feature map and share computation between the image and window embeddings.

        Args:
            batch (torch.Tensor): Batch of input images of shape ``(N, C, H, W)``.

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]: A tuple containing the image embeddings,
            window embeddings, and patch embeddings respectively.

        Examples:
            >>> model = WinClipModel()
            >>> model.prepare_masks()
            >>> batch = torch.rand((1, 3, 240, 240))
            >>> image_embeddings, window_embeddings, patch_embeddings = model.encode_image(batch)
            >>> image_embeddings.shape
            torch.Size([1, 640])
            >>> [embedding.shape for embedding in window_embeddings]
            [torch.Size([1, 196, 640]), torch.Size([1, 169, 640])]
            >>> patch_embeddings.shape
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
        """Computes the embeddings for each window in the feature map using the given masks.

        Args:
            feature_map (torch.Tensor): The input feature map of shape ``(n_batches, n_patches, dimensionality)``.
            masks (torch.Tensor): Masks of shape ``(kernel_size, n_masks)`` representing the sliding window locations.

        Returns:
            torch.Tensor: The embeddings for each sliding window location.
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
    def forward(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward-pass through the model to obtain image and pixel scores.

        Args:
            batch (torch.Tensor): Batch of input images of shape ``(batch_size, C, H, W)``.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing the image scores and pixel scores.
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
        return image_scores, pixel_scores.squeeze(1)

    def _compute_zero_shot_scores(
        self,
        image_scores: torch.Tensor,
        window_embeddings: list[torch.Tensor],
    ) -> torch.Tensor:
        """Compute the multi-scale anomaly score maps based on the text embeddings.

        Each window embedding is compared to the text embeddings to obtain a similarity score for each window. Harmonic
        averaging is then used to aggregate the scores for each window into a single score map for each scale. Finally,
        the score maps are combined into a single multi-scale score map by aggregating across scales.

        Args:
            image_scores (torch.Tensor): Tensor of shape ``(batch_size)`` representing the full image scores.
            window_embeddings (list[torch.Tensor]): List of tensors of shape ``(batch_size, n_windows, n_features)``
                representing the embeddings for each sliding window location.

        Returns:
            torch.Tensor: Tensor of shape ``(batch_size, H, W)`` representing the 0-shot scores for each patch location.
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
        """Compute the multi-scale anomaly score maps based on the reference image embeddings.

        Visual association scores are computed between the extracted embeddings and the reference image embeddings for
        each scale. The window-level scores are additionally aggregated into a single score map for each scale using
        harmonic averaging. The final score maps are obtained by averaging across scales.

        Args:
            patch_embeddings (torch.Tensor): Full-scale patch embeddings of shape
                ``(batch_size, n_patches, n_features)``.
            window_embeddings (list[torch.Tensor]): List of tensors of shape ``(batch_size, n_windows, n_features)``
                representing the embeddings for each sliding window location.

        Returns:
            torch.Tensor: Tensor of shape ``(batch_size, H, W)`` representing the few-shot scores for each patch
                location.
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
        """Collect text embeddings for the object class using a compositional prompt ensemble.

        First, an ensemble of normal and anomalous prompts is created based on the name of the object class. The
        prompt ensembles are then tokenized and encoded to obtain prompt embeddings. The prompt embeddings are
        averaged to obtain a single text embedding for the object class. These final text embeddings are stored in
        the model to be used during inference.

        Args:
            class_name (str): The name of the object class used in the prompt ensemble.
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
        """Collect visual embeddings based on a set of normal reference images.

        Args:
            images (torch.Tensor): Tensor of shape ``(K, C, H, W)`` containing the reference images.
        """
        _, self._visual_embeddings, self._patch_embeddings = self.encode_image(images)

    def _generate_masks(self) -> list[torch.Tensor]:
        """Prepare a set of masks that operate as multi-scale sliding windows.

        For each of the scales, a set of masks is created that select patches from the feature map. Each mask represents
        a sliding window location in the pixel domain. The masks are stored in the model to be used during inference.

        Returns:
            list[torch.Tensor]: A list of tensors of shape ``(n_patches_per_mask, n_masks)`` representing the sliding
                window locations for each scale.
        """
        return [make_masks(self.grid_size, scale, 1) for scale in self.scales]

    @property
    def transform(self) -> Compose:
        """The transform used by the model.

        To obtain the transforms, we retrieve the transforms from the clip backbone. Since the original transforms are
        intended for PIL images, we prepend a ToPILImage transform to the list of transforms.
        """
        transforms = copy(self._transform.transforms)
        transforms.insert(0, ToPILImage())
        return Compose(transforms)

    @property
    def text_embeddings(self) -> torch.Tensor:
        """The text embeddings used by the model."""
        if self._text_embeddings.numel() == 0:
            msg = "Text embeddings have not been collected. Pass a class name to the model using ``setup``."
            raise RuntimeError(msg)
        return self._text_embeddings

    @property
    def visual_embeddings(self) -> list[torch.Tensor]:
        """The visual embeddings used by the model."""
        if self._visual_embeddings[0].numel() == 0:
            msg = "Visual embeddings have not been collected. Pass some reference images to the model using ``setup``."
            raise RuntimeError(msg)
        return self._visual_embeddings

    @property
    def patch_embeddings(self) -> torch.Tensor:
        """The patch embeddings used by the model."""
        if self._patch_embeddings.numel() == 0:
            msg = "Patch embeddings have not been collected. Pass some reference images to the model using ``setup``."
            raise RuntimeError(msg)
        return self._patch_embeddings
