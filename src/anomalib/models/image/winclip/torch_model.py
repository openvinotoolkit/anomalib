# Original Code
# https://github.com/caoyunkang/WinClip.
# SPDX-License-Identifier: MIT
#
# Modified
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import open_clip
import torch
import torch.nn.functional as F
from open_clip.tokenizer import tokenize
from torch import nn

from .ad_prompts import *
from .prompting import create_prompt_ensemble
from .utils import harmonic_aggregation, make_masks, simmilarity_score, visual_association_score


class WinClipModel(nn.Module):
    def __init__(self, n_shot: int = 0, model_name="ViT-B-16-plus-240", scales=(2, 3)):
        super().__init__()
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained="laion400m_e31")
        self.grid_size = self.model.visual.grid_size
        self.n_shot = n_shot
        # self.model.visual.output_tokens = True
        self.scales = scales

        self.masks: list[torch.Tensor] | None = None
        self.text_embeddings: torch.Tensor | None = None

    def multiscale(self):
        pass

    def encode_text(self, text):
        return self.model.encode_text(text)

    def encode_image(self, image):
        self.model.visual.output_tokens = True
        # TODO: Investigate if this is actually needed
        self.model.visual.final_ln_after_pool = True

        # register hook to retrieve feature map
        feature_map = {}

        def get_feature_map(name):
            def hook(model, input, output):
                feature_map[name] = input[0].detach()

            return hook

        self.model.visual.patch_dropout.register_forward_hook(get_feature_map("patch_dropout"))

        # get patch embeddings
        image_embeddings, patch_embeddings = self.model.encode_image(image)

        # get window embeddings
        intermediate_tokens = feature_map["patch_dropout"]
        window_embeddings = [self._get_window_embeddings(intermediate_tokens, masks) for masks in self.masks]

        return (
            image_embeddings,
            window_embeddings,
            patch_embeddings,
        )

    def _get_window_embeddings(self, feature_map, masks):
        batch_size = feature_map.shape[0]
        n_masks = masks.shape[1]
        device = feature_map.device

        # prepend zero index for class embeddings
        class_index = torch.zeros(1, n_masks, dtype=int).to(device)
        masks = torch.cat((class_index, masks.to(device))).T
        # apply masks to feature map
        masked = torch.cat([torch.index_select(feature_map, 1, mask) for mask in masks])

        # finish forward pass on masked features
        masked = self.model.visual.patch_dropout(masked)
        masked = self.model.visual.ln_pre(masked)

        masked = masked.permute(1, 0, 2)  # NLD -> LND
        masked = self.model.visual.transformer(masked)
        masked = masked.permute(1, 0, 2)  # LND -> NLD

        pooled, _ = self.model.visual._global_pool(masked)
        pooled = self.model.visual.ln_post(pooled)

        if self.model.visual.proj is not None:
            pooled = pooled @ self.model.visual.proj

        return pooled.reshape((n_masks, batch_size, -1)).permute(1, 0, 2)

    def forward(self, image):
        image = torch.load("/home/djameln/WinCLIP-pytorch/image_wnclp.pt")
        image_embeddings, window_embeddings, patch_embeddings = self.encode_image(image)

        # get anomaly scores
        image_scores = simmilarity_score(image_embeddings, self.text_embeddings)[..., -1]

        # get 0-shot scores
        multiscale_scores = self._compute_zero_shot_scores(image_scores, window_embeddings)

        # get n-shot scores
        if self.n_shot:
            few_shot_scores = self._compute_few_shot_scores(patch_embeddings, window_embeddings)
            multiscale_scores = (multiscale_scores + few_shot_scores) / 2
            image_scores = (image_scores + few_shot_scores.amax(dim=(-2, -1))) / 2

        pixel_scores = F.interpolate(
            multiscale_scores.unsqueeze(1),
            size=image.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze()
        return image_scores, pixel_scores

    def _compute_zero_shot_scores(
        self, image_scores: torch.Tensor, window_embeddings: list[torch.Tensor]
    ) -> torch.Tensor:
        """Compute the text scores for each window based on the image scores and window embeddings.

        Args:
            image_scores (torch.Tensor): Tensor of shape (batch_size) representing the image scores.
            window_embeddings (list[torch.Tensor]): List of tensors representing the embeddings for each window.

        Returns:
            torch.Tensor: Tensor of shape (batch_size, num_windows) representing the text scores for each window.
        """
        multiscale_scores = [image_scores.view(-1, 1, 1).repeat(1, self.grid_size[0], self.grid_size[1])]
        for window_embedding, mask in zip(window_embeddings, self.masks):
            scores = simmilarity_score(window_embedding, self.text_embeddings)[..., -1]
            multiscale_scores.append(harmonic_aggregation(scores, self.grid_size, mask))

        return (len(self.scales) + 1) / (1 / torch.stack(multiscale_scores)).sum(dim=0)

    def _compute_few_shot_scores(self, patch_embeddings, window_embeddings):
        multiscale_scores = [
            visual_association_score(patch_embeddings, self.patch_embeddings).reshape((-1,) + self.grid_size),
        ]
        for window_embedding, reference_embedding, mask in zip(
            window_embeddings,
            self.visual_embeddings,
            self.masks,
        ):
            scores = visual_association_score(window_embedding, reference_embedding)
            multiscale_scores.append(harmonic_aggregation(scores, self.grid_size, mask))

        return torch.stack(multiscale_scores).mean(dim=0)

    def collect_text_embeddings(self, object: str, device: torch.device | None = None):
        # collect prompt ensemble
        normal_prompts, anomalous_prompts = create_prompt_ensemble(object)
        # tokenize
        normal_tokens = tokenize(normal_prompts)
        anomalous_tokens = tokenize(anomalous_prompts)
        # encode
        normal_embeddings = self.model.encode_text(normal_tokens)
        anomalous_embeddings = self.model.encode_text(anomalous_tokens)
        # average
        normal_embeddings = torch.mean(normal_embeddings, dim=0, keepdim=True)
        anomalous_embeddings = torch.mean(anomalous_embeddings, dim=0, keepdim=True)
        # normalize
        text_embeddings = torch.cat((normal_embeddings, anomalous_embeddings))
        # text_embeddings /= text_embeddings.norm(dim=1, keepdim=True)
        # move to device
        if device is not None:
            text_embeddings = text_embeddings.to(device)
        # store
        self.text_embeddings = text_embeddings.detach()

    def collect_image_embeddings(self, images: torch.Tensor, device: torch.device | None = None) -> None:
        # encode
        with torch.no_grad():
            _, window_embeddings, patch_embeddings = self.encode_image(images)
        self.visual_embeddings = [window.to(device) for window in window_embeddings]
        self.patch_embeddings = patch_embeddings.to(device)

    def create_masks(self, device: torch.device | None = None) -> None:
        """Create masks for each scale."""
        self.masks = [make_masks(self.grid_size, scale, 1).to(device) for scale in self.scales]
