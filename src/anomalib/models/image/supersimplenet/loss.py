"""Loss function for the SuperSimpleNet model implementation."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from functools import partial

import torch
from torch import nn
from torchvision.ops.focal_loss import sigmoid_focal_loss


class SSNLoss(nn.Module):
    """SuperSimpleNet loss function.

    Args:
        truncation_term (float): L1 loss truncation term preventing overfitting.
    """

    def __init__(self, truncation_term: float = 0.5) -> None:
        super().__init__()
        self.focal_loss = partial(sigmoid_focal_loss, alpha=-1, gamma=4.0, reduction="mean")
        self.th = truncation_term

    def trunc_l1_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate the truncated L1 loss between `pred` and `target`.

        Args:
            pred (torch.Tensor): predicted values.
            target (torch.Tensor): target GT values.

        Returns:
            torch.Tensor: L1 truncated loss value.
        """
        normal_scores = pred[target == 0]
        anomalous_scores = pred[target > 0]
        # push normal towards negative numbers
        true_loss = torch.clip(normal_scores + self.th, min=0)
        # push anomalous towards positive numbers
        fake_loss = torch.clip(-anomalous_scores + self.th, min=0)

        true_loss = true_loss.mean() if len(true_loss) else torch.tensor(0)
        fake_loss = fake_loss.mean() if len(fake_loss) else torch.tensor(0)

        return true_loss + fake_loss

    def forward(
        self,
        pred_map: torch.Tensor,
        pred_score: torch.Tensor,
        target_mask: torch.Tensor,
        target_label: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate loss based on predicted anomaly maps and scores.

        Total loss = Lseg and Lcls
        where
        Lseg = Lfocal(map) + Ltruncl1(map)
        Lcls = Lfocal(score)

        Args:
            pred_map: predicted anomaly maps.
            pred_score: predicted anomaly scores.
            target_mask: GT anomaly masks.
            target_label: GT anomaly labels.

        Returns:
            torch.Tensor: loss value.
        """
        map_focal = self.focal_loss(pred_map, target_mask)
        map_trunc_l1 = self.trunc_l1_loss(pred_map, target_mask)
        score_focal = self.focal_loss(pred_score, target_label)

        return map_focal + map_trunc_l1 + score_focal
