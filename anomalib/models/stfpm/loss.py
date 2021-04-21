from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class FeaturePyramidLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss(reduction="none")

    def __compute_layer_loss(self, teacher_feats: Tensor, student_feats: Tensor) -> Tensor:
        norm_teacher_features = F.normalize(teacher_feats)
        norm_student_features = F.normalize(student_feats)

        layer_loss = 0.5 * self.mse_loss(norm_teacher_features, norm_student_features)
        layer_loss = torch.mean(layer_loss)
        return layer_loss

    def forward(self, teacher_features: Dict[str, Tensor], student_features: Dict[str, Tensor]) -> Tensor:

        losses: List[Tensor] = []
        for layer in teacher_features.keys():
            loss = self.__compute_layer_loss(teacher_features[layer], student_features[layer])
            losses.append(loss)

        return sum(losses)
