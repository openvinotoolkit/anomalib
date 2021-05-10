#  COSMONiO © All rights reserved.
#  This file is subject to the terms and conditions defined in file 'LICENSE.txt',
#  which is part of this source code package.

from typing import Tuple

from torchvision import transforms


def get_anomaly_transform(image_shape: Tuple[int, int]) -> transforms.Compose:
    """
    Create instance of pytorch transforms for anomaly classification.
    """
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(image_shape),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
        ])
    return transform
