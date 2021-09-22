"""
Transforms
The script contains image transform functions to be used during data pre-processing
Transforms are created following albumentation library to fit with rest of the pipeline
"""
from typing import Tuple

import cv2
import numpy as np
from albumentations.core.transforms_interface import ImageOnlyTransform


class RgbToGray(ImageOnlyTransform):
    """
    Transform for RGB to 1-channel gray image transform
    """

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        # pylint: disable=unused-argument
        """
        Function implements RgbToGray transformation
        Args:
            img: input image to be transformed

        Returns: Transformed image

        """
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


class BilateralFilter(ImageOnlyTransform):
    """
    Transform to apply image filter to reduce noise while retaining edges
    """

    def __init__(
        self,
        diameter: int = 9,
        sigma_color: int = 75,
        sigma_space: int = 75,
        always_apply: bool = False,
        probability: float = 1.0,
    ):
        """
        Initialize parameters for bilateral filtering
        Args:
            diameter: diameter of each pixel neighborhood
            sigma_color: value of sigma in color space. The greater the value,
                        the colors farther to each other will start to get mixed
            sigma_space: value of sigma in coordinate space. The greater the value,
                        the more further pixels within the sigma_color range will start to get mixed
            always_apply: flag to enable transformation irrespective of its probability
            probability: probability of applying this transformation
        """
        super().__init__(always_apply, probability)
        self.diameter = diameter
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        # pylint: disable=unused-argument
        """
        Function implements Bilateral Filtering transformation
        Args:
            img: input image to be transformed

        Returns: Transformed image
        """
        return cv2.bilateralFilter(img, self.diameter, self.sigma_color, self.sigma_space)

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        """
        List of parameters that are input for the transformation
        """
        return "diameter", "sigma_color", "sigma_space"
