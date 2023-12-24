"""PyTorch model for the WinCLIP implementation."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable

import open_clip
import torch
from open_clip.tokenizer import tokenize
from torch import nn
from torch.nn.modules.linear import Identity

from .prompting import create_prompt_ensemble
from .utils import harmonic_aggregation, make_masks, simmilarity_score, visual_association_score

BACKBONE = "ViT-B-16-plus-240"
TEMPERATURE = 0.07  # temperature hyperparameter from the clip paper


class WinClipModel(nn.Module):
    """PyTorch module that implements the WinClip model for image anomaly detection.

    Args:
        k_shot (int, optional): The number of reference images used for few-shot anomaly detection.
            Defaults to 0.
        scales (tuple[int], optional): The scales of the sliding windows used for multiscale anomaly detection.
            Defaults to (2, 3).

    Attributes:
        clip (CLIP): The CLIP model used for image and text encoding.
        grid_size (tuple[int]): The size of the feature map grid.
        k_shot (int): The number of reference images used for few-shot anomaly detection.
        scales (tuple[int]): The scales of the sliding windows used for multiscale anomaly detection.
        masks (list[torch.Tensor] | None): The masks representing the sliding window locations.
        text_embeddings (torch.Tensor | None): The text embeddings for the compositional prompt ensemble.
        visual_embeddings (list[torch.Tensor] | None): The multiscale embeddings for the reference images.
        patch_embeddings (torch.Tensor | None): The patch embeddings for the reference images.
    """

    def __init__(self, k_shot: int = 0, scales: tuple = (2, 3)) -> None:
        super().__init__()
        self.backbone = BACKBONE
        self.temperature = TEMPERATURE
        self.k_shot = k_shot
        self.scales = scales

        # initialize CLIP model
        self.clip = open_clip.create_model(self.backbone, pretrained="laion400m_e31")
        self.clip.visual.output_tokens = True
        self.grid_size = self.clip.visual.grid_size

        self.masks: list[torch.Tensor] | None = None
        self.text_embeddings: torch.Tensor | None = None
        self.visual_embeddings: list[torch.Tensor] | None = None
        self.patch_embeddings: torch.Tensor | None = None

    def encode_image(self, batch: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        """Encode the batch of images to obtain image embeddings, window embeddings, and patch embeddings.

        The image embeddings and patch embeddings are obtained by passing the batch of images through the model. The
        window embeddings are obtained by masking the feature map and passing it through the transformer. A forward hook
        is used to retrieve the intermediate feature map and share computation between the image and window embeddings.

        Args:
            batch (torch.Tensor): Batch of input images of shape (N, C, H, W).

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]: A tuple containing the image embeddings,
            window embeddings, and patch embeddings respectively.
        """
        assert isinstance(self.masks, list), "Masks have not been prepared. Call prepare_masks before inference."
        # register hook to retrieve intermediate feature map
        outputs = {}

        def get_feature_map(name: str) -> Callable:
            def hook(_model: Identity, inputs: tuple[torch.Tensor,], _outputs: torch.Tensor) -> None:
                del _model, _outputs
                outputs[name] = inputs[0].detach()

            return hook

        # register hook to get the intermediate tokens of the transformer
        self.clip.visual.patch_dropout.register_forward_hook(get_feature_map("feature_map"))

        # get patch embeddings
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
            feature_map (torch.Tensor): The input feature map of shape (n_batches, n_patches, dimensionality).
            masks (torch.Tensor): Masks of shape (kernel_size, n_masks) representing the sliding window locations.

        Returns:
            torch.Tensor: The embeddings for each sliding window location.
        """
        batch_size = feature_map.shape[0]
        n_masks = masks.shape[1]
        device = feature_map.device

        # prepend zero index for class embeddings
        class_index = torch.zeros(1, n_masks, dtype=int).to(device)
        masks = torch.cat((class_index, masks.to(device))).T
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

    def forward(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward-pass through the model to obtain image and pixel scores.

        Args:
            batch (torch.Tensor): Batch of input images of shape (batch_size, C, H, W).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing the image scores and pixel scores.
        """
        image_embeddings, window_embeddings, patch_embeddings = self.encode_image(batch)

        # get zero-shot scores
        image_scores = simmilarity_score(image_embeddings, self.text_embeddings, self.temperature)[..., -1]
        multiscale_scores = self._compute_zero_shot_scores(image_scores, window_embeddings)

        # get few-shot scores
        if self.k_shot:
            few_shot_scores = self._compute_few_shot_scores(patch_embeddings, window_embeddings)
            multiscale_scores = (multiscale_scores + few_shot_scores) / 2
            image_scores = (image_scores + few_shot_scores.amax(dim=(-2, -1))) / 2

        # reshape to image dimensions
        pixel_scores = nn.functional.interpolate(multiscale_scores.unsqueeze(1), size=batch.shape[-2:], mode="bilinear")
        return image_scores, pixel_scores.squeeze()

    def _compute_zero_shot_scores(
        self,
        image_scores: torch.Tensor,
        window_embeddings: list[torch.Tensor],
    ) -> torch.Tensor:
        """Compute the multiscale anomaly score maps based on the text embeddings.

        Each window embedding is compared to the text embeddings to obtain a similarity score for each window. Harmonic
        averaging is then used to aggregate the scores for each window into a single score map for each scale. Finally,
        the score maps are combined into a single multiscale score map by aggregating across scales.

        Args:
            image_scores (torch.Tensor): Tensor of shape (batch_size) representing the full image scores.
            window_embeddings (list[torch.Tensor]): List of tensors of shape (batch_size, n_windows, n_features)
                representing the embeddings for each sliding window location.

        Returns:
            torch.Tensor: Tensor of shape (batch_size, H, W) representing the 0-shot scores for each patch location.
        """
        assert isinstance(self.masks, list), "Masks have not been prepared. Call prepare_masks before inference."
        # image scores are added to represent the full image scale
        multiscale_scores = [image_scores.view(-1, 1, 1).repeat(1, self.grid_size[0], self.grid_size[1])]
        # add aggregated scores for each scale
        for window_embedding, mask in zip(window_embeddings, self.masks, strict=True):
            scores = simmilarity_score(window_embedding, self.text_embeddings, self.temperature)[..., -1]
            multiscale_scores.append(harmonic_aggregation(scores, self.grid_size, mask))
        # aggregate scores across scales
        return (len(self.scales) + 1) / (1 / torch.stack(multiscale_scores)).sum(dim=0)

    def _compute_few_shot_scores(
        self,
        patch_embeddings: torch.Tensor,
        window_embeddings: list[torch.Tensor],
    ) -> torch.Tensor:
        """Compute the multiscale anomaly score maps based on the reference image embeddings.

        Visual association scores are computed between the extracted embeddings and the reference image embeddings for
        each scale. The window-level scores are additionally aggregated into a single score map for each scale using
        harmonic averaging. The final score maps are obtained by averaging across scales.

        Args:
            patch_embeddings (torch.Tensor): Full-scale patch embeddings of shape (batch_size, n_patches, n_features).
            window_embeddings (list[torch.Tensor]): List of tensors of shape (batch_size, n_windows, n_features)
                representing the embeddings for each sliding window location.

        Returns:
            torch.Tensor: Tensor of shape (batch_size, H, W) representing the few-shot scores for each patch location.
        """
        assert isinstance(
            self.visual_embeddings,
            list,
        ), "Visual embeddings have not been prepared. Call collect_visual_embeddings before inference."
        assert isinstance(self.masks, list), "Masks have not been prepared. Call prepare_masks before inference."

        multiscale_scores = [
            visual_association_score(patch_embeddings, self.patch_embeddings).reshape((-1, *self.grid_size)),
        ]
        for window_embedding, reference_embedding, mask in zip(
            window_embeddings,
            self.visual_embeddings,
            self.masks,
            strict=True,
        ):
            scores = visual_association_score(window_embedding, reference_embedding)
            multiscale_scores.append(harmonic_aggregation(scores, self.grid_size, mask))

        return torch.stack(multiscale_scores).mean(dim=0)

    def collect_text_embeddings(self, class_name: str, device: torch.device | None = None) -> None:
        """Collect text embeddings for the object class using a compositional prompt ensemble.

        First, an ensemble of normal and anomalous prompts is created based on the name of the object class. The
        prompt ensembles are then tokenized and encoded to obtain prompt embeddings. The prompt embeddings are
        averaged to obtain a single text embedding for the object class. These final text embeddings are stored in
        the model to be used during inference.

        Args:
            class_name (str): The name of the object class used in the prompt ensemble.
            device (torch.device | None, optional): The device on which the embeddings should be stored.
                Defaults to None.
        """
        # collect prompt ensemble
        normal_prompts, anomalous_prompts = create_prompt_ensemble(class_name)
        # tokenize prompts
        normal_tokens = tokenize(normal_prompts)
        anomalous_tokens = tokenize(anomalous_prompts)
        # encode tokens to obtain prompt embeddings
        with torch.no_grad():
            normal_embeddings = self.clip.encode_text(normal_tokens)
            anomalous_embeddings = self.clip.encode_text(anomalous_tokens)
        # average prompt embeddings
        normal_embeddings = torch.mean(normal_embeddings, dim=0, keepdim=True)
        anomalous_embeddings = torch.mean(anomalous_embeddings, dim=0, keepdim=True)
        # concatenate and store
        text_embeddings = torch.cat((normal_embeddings, anomalous_embeddings))
        self.text_embeddings = text_embeddings.to(device)

    def collect_visual_embeddings(self, images: torch.Tensor, device: torch.device | None = None) -> None:
        """Collect visual embeddings based on a set of normal reference images.

        Args:
            images (torch.Tensor): Tensor of shape (batch_size, C, H, W) containing the reference images.
            device (torch.device | None, optional): The device on which the embeddings should be stored.
                Defaults to None.
        """
        with torch.no_grad():
            _, window_embeddings, patch_embeddings = self.encode_image(images)
        self.visual_embeddings = [window.to(device) for window in window_embeddings]
        self.patch_embeddings = patch_embeddings.to(device)

    def prepare_masks(self, device: torch.device | None = None) -> None:
        """Prepare a set of masks that operate as multiscale sliding windows.

        For each of the scales, a set of masks is created that select patches from the feature map. Each mask represents
        a sliding window location in the pixel domain. The masks are stored in the model to be used during inference.

        Args:
            device (torch.device | None, optional): The device on which the masks should be stored.
                Defaults to None.
        """
        self.masks = [make_masks(self.grid_size, scale, 1).to(device) for scale in self.scales]
