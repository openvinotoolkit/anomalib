"""
Pre Process
This module contains `PreProcessor` class that applies preprocessing
to an input image before the forward-pass stage.
"""

from typing import Union

from omegaconf import DictConfig, ListConfig

from .transforms import get_transforms


class PreProcessor:
    """
    PreProcessor class applies the pre-processing and data augmentations
    to the input and returns the transformed output, which could be
    either numpy ndarray or torch tensor. When `PreProcessor` class is
    used for training, the output would be `torch.Tensor`. For the inference
    it returns a numpy array

    Args:
        config: Transformation configurations
        is_train: To check whether training mode is on. When on trianing mode,
            Transformations are applied with a probability
        to_tensor: Convert the final transformed output
            to `torch.Tensor`. For OpenVino inference, this is set to False
            to return `np.ndarray`. Defaults to True.

    Examples:
        >>> import skimage
        >>> image = skimage.data.astronaut()

        >>> pre_processor = PreProcessor(config=config.transform, to_tensor=True)
        >>> output = pre_processor(image=image)
        >>> output["image"].shape
        torch.Size([3, 128, 128])

        >>> pre_processor = PreProcessor(config=config.transform, to_tensor=False)
        >>> output = pre_processor(image=image)
        >>> output["image"].shape
        (128, 128, 3)
    """

    def __init__(self, config: Union[DictConfig, ListConfig], is_train: bool = True, to_tensor: bool = True) -> None:
        self.config = config
        self.is_train = is_train
        self.to_tensor = to_tensor

        if is_train is True and to_tensor is False:
            raise ValueError("to_tensor cannot be False in train mode!")

        self.transforms = get_transforms(config=self.config, is_train=self.is_train, to_tensor=self.to_tensor)

    def __call__(self, *args, force_apply=False, **data):
        return self.transforms(*args, force_apply=False, **data)

    def __repr__(self) -> str:
        return self.transforms.__repr__()
