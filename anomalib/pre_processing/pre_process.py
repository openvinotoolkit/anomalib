"""Pre Process.

This module contains `PreProcessor` class that applies preprocessing
to an input image before the forward-pass stage.
"""

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

from typing import Optional, Tuple, Union

import albumentations as A
from albumentations.pytorch import ToTensorV2


class PreProcessor:
    """Applies pre-processing and data augmentations to the input and returns the transformed output.

    Output could be either numpy ndarray or torch tensor.
    When `PreProcessor` class is used for training, the output would be `torch.Tensor`.
    For the inference it returns a numpy array.

    Args:
        config (Optional[Union[str, A.Compose]], optional): Transformation configurations.
            When it is ``None``, ``PreProcessor`` only applies resizing. When it is ``str``
            it loads the config via ``albumentations`` deserialisation methos . Defaults to None.
        image_size (Optional[Union[int, Tuple[int, int]]], optional): When there is no config,
        ``image_size`` resizes the image. Defaults to None.
        to_tensor (bool, optional): Boolean to check whether the augmented image is transformed
            into a tensor or not. Defaults to True.

    Examples:
        >>> import skimage
        >>> image = skimage.data.astronaut()

        >>> pre_processor = PreProcessor(image_size=256, to_tensor=False)
        >>> output = pre_processor(image=image)
        >>> output["image"].shape
        (256, 256, 3)

        >>> pre_processor = PreProcessor(image_size=256, to_tensor=True)
        >>> output = pre_processor(image=image)
        >>> output["image"].shape
        torch.Size([3, 256, 256])


        Transforms could be read from albumentations Compose object.
            >>> import albumentations as A
            >>> from albumentations.pytorch import ToTensorV2
            >>> config = A.Compose([A.Resize(512, 512), ToTensorV2()])
            >>> pre_processor = PreProcessor(config=config, to_tensor=False)
            >>> output = pre_processor(image=image)
            >>> output["image"].shape
            (512, 512, 3)
            >>> type(output["image"])
            numpy.ndarray

        Transforms could be deserialized from a yaml file.
            >>> transforms = A.Compose([A.Resize(1024, 1024), ToTensorV2()])
            >>> A.save(transforms, "/tmp/transforms.yaml", data_format="yaml")
            >>> pre_processor = PreProcessor(config="/tmp/transforms.yaml")
            >>> output = pre_processor(image=image)
            >>> output["image"].shape
            torch.Size([3, 1024, 1024])
    """

    def __init__(
        self,
        config: Optional[Union[str, A.Compose]] = None,
        image_size: Optional[Union[int, Tuple]] = None,
        to_tensor: bool = True,
    ) -> None:
        self.config = config
        self.image_size = image_size
        self.to_tensor = to_tensor

        self.transforms = self.get_transforms()

    def get_transforms(self) -> A.Compose:
        """Get transforms from config or image size.

        Returns:
            A.Compose: List of albumentation transformations to apply to the
                input image.
        """
        if self.config is None and self.image_size is None:
            raise ValueError(
                "Both config and image_size cannot be `None`. "
                "Provide either config file to de-serialize transforms "
                "or image_size to get the default transformations"
            )

        transforms: A.Compose

        if self.config is None and self.image_size is not None:
            if isinstance(self.image_size, int):
                height, width = self.image_size, self.image_size
            elif isinstance(self.image_size, tuple):
                height, width = self.image_size
            else:
                raise ValueError("``image_size`` could be either int or Tuple[int, int]")

            transforms = A.Compose(
                [
                    A.Resize(height=height, width=width, always_apply=True),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ]
            )

        if self.config is not None:
            if isinstance(self.config, str):
                transforms = A.load(filepath=self.config, data_format="yaml")
            elif isinstance(self.config, A.Compose):
                transforms = self.config
            else:
                raise ValueError("config could be either ``str`` or ``A.Compose``")

        if not self.to_tensor:
            if isinstance(transforms[-1], ToTensorV2):
                transforms = A.Compose(transforms[:-1])

        return transforms

    def __call__(self, *args, **kwargs):
        """Return transformed arguments."""
        return self.transforms(*args, **kwargs)
