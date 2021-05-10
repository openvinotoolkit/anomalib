#  COSMONiO © All rights reserved.
#  This file is subject to the terms and conditions defined in file 'LICENSE.txt',
#  which is part of this source code package.

import torch.nn as nn
from torchvision.models import resnet50, ResNet


def resnet50_feature_extractor() -> ResNet:
    """
    Creates resnet50 model with pretrained weights from torchvision model zoo, and removes the last FC layer to make it
    a feature extractor.
    """
    model = resnet50(pretrained=True)
    model.fc = nn.Identity()
    return model
