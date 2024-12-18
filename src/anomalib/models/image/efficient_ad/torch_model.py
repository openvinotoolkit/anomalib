"""PyTorch implementation of the EfficientAd model architecture.

This module contains the PyTorch implementation of the student, teacher and
autoencoder networks used in EfficientAd for fast and accurate anomaly detection.

The model consists of:
    - A pre-trained EfficientNet teacher network
    - A lightweight student network
    - Knowledge distillation training
    - Anomaly detection via feature comparison

Example:
    >>> from anomalib.models.image.efficient_ad.torch_model import EfficientAdModel
    >>> model = EfficientAdModel()
    >>> input_tensor = torch.randn(32, 3, 256, 256)
    >>> output = model(input_tensor)
    >>> output["anomaly_map"].shape
    torch.Size([32, 256, 256])

Paper:
    "EfficientAd: Accurate Visual Anomaly Detection at
    Millisecond-Level Latencies"
    https://arxiv.org/pdf/2303.14535.pdf

See Also:
    :class:`anomalib.models.image.efficient_ad.lightning_model.EfficientAd`:
        Lightning implementation of the EfficientAd model.
"""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import math
from enum import Enum

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812
from torchvision import transforms

from anomalib.data import InferenceBatch

logger = logging.getLogger(__name__)


def imagenet_norm_batch(x: torch.Tensor) -> torch.Tensor:
    """Normalize batch of images using ImageNet mean and standard deviation.

    This function normalizes a batch of images using the standard ImageNet mean and
    standard deviation values. The normalization is done channel-wise.

    Args:
        x (torch.Tensor): Input batch tensor of shape ``(N, C, H, W)`` where
            ``N`` is batch size, ``C`` is number of channels (3 for RGB),
            ``H`` is height and ``W`` is width.

    Returns:
        torch.Tensor: Normalized batch tensor with same shape as input, where each
            channel is normalized using ImageNet statistics:
            - Red channel: mean=0.485, std=0.229
            - Green channel: mean=0.456, std=0.224
            - Blue channel: mean=0.406, std=0.225
    """
    mean = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None].to(x.device)
    std = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None].to(x.device)
    return (x - mean) / std


def reduce_tensor_elems(tensor: torch.Tensor, m: int = 2**24) -> torch.Tensor:
    """Reduce the number of elements in a tensor by random sampling.

    This function flattens an n-dimensional tensor and randomly samples at most ``m``
    elements from it. This is used to handle the limitation of ``torch.quantile``
    operation which supports a maximum of 2^24 elements.

    Reference:
        https://github.com/pytorch/pytorch/blob/b9f81a483a7879cd3709fd26bcec5f1ee33577e6/aten/src/ATen/native/Sorting.cpp#L291

    Args:
        tensor (torch.Tensor): Input tensor of any shape from which elements will be
            sampled.
        m (int, optional): Maximum number of elements to sample. If the flattened
            tensor has more elements than ``m``, random sampling is performed.
            Defaults to ``2**24``.

    Returns:
        torch.Tensor: A flattened tensor containing at most ``m`` elements randomly
            sampled from the input tensor.

    Example:
        >>> import torch
        >>> tensor = torch.randn(1000, 1000)  # 1M elements
        >>> reduced = reduce_tensor_elems(tensor, m=1000)
        >>> reduced.shape
        torch.Size([1000])
    """
    tensor = torch.flatten(tensor)
    if len(tensor) > m:
        # select a random subset with m elements.
        perm = torch.randperm(len(tensor), device=tensor.device)
        idx = perm[:m]
        tensor = tensor[idx]
    return tensor


class EfficientAdModelSize(str, Enum):
    """Supported EfficientAd model sizes.

    The EfficientAd model comes in two sizes:
        - ``M`` (medium): Uses a larger architecture with more parameters
        - ``S`` (small): Uses a smaller architecture with fewer parameters

    Example:
        >>> from anomalib.models.image.efficient_ad.torch_model import (
        ...     EfficientAdModelSize
        ... )
        >>> model_size = EfficientAdModelSize.S
        >>> model_size
        'small'
        >>> model_size = EfficientAdModelSize.M
        >>> model_size
        'medium'
    """

    M = "medium"
    S = "small"


class SmallPatchDescriptionNetwork(nn.Module):
    """Small variant of the Patch Description Network.

    This network processes input images through a series of convolutional and pooling
    layers to extract patch-level features. It uses a smaller architecture compared
    to the medium variant.

    Args:
        out_channels (int): Number of output channels in the final convolution layer.
        padding (bool, optional): Whether to use padding in convolutional layers.
            Defaults to ``False``.

    Example:
        >>> import torch
        >>> from anomalib.models.image.efficient_ad.torch_model import (
        ...     SmallPatchDescriptionNetwork
        ... )
        >>> model = SmallPatchDescriptionNetwork(out_channels=384)
        >>> input_tensor = torch.randn(32, 3, 64, 64)
        >>> output = model(input_tensor)
        >>> output.shape
        torch.Size([32, 384, 13, 13])

    Note:
        The network applies ImageNet normalization to the input before processing.
    """

    def __init__(self, out_channels: int, padding: bool = False) -> None:
        super().__init__()
        pad_mult = 1 if padding else 0
        self.conv1 = nn.Conv2d(3, 128, kernel_size=4, stride=1, padding=3 * pad_mult)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=3 * pad_mult)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1 * pad_mult)
        self.conv4 = nn.Conv2d(256, out_channels, kernel_size=4, stride=1, padding=0 * pad_mult)
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape ``(N, 3, H, W)``.

        Returns:
            torch.Tensor: Output feature maps of shape
                ``(N, out_channels, H', W')``, where ``H'`` and ``W'`` are
                determined by the network architecture and padding settings.
        """
        x = imagenet_norm_batch(x)
        x = F.relu(self.conv1(x))
        x = self.avgpool1(x)
        x = F.relu(self.conv2(x))
        x = self.avgpool2(x)
        x = F.relu(self.conv3(x))
        return self.conv4(x)


class MediumPatchDescriptionNetwork(nn.Module):
    """Medium-sized patch description network.

    This network processes input images through a series of convolutional and
    pooling layers to extract descriptive features from image patches.

    Args:
        out_channels (int): Number of output channels in the final convolution
            layer.
        padding (bool, optional): Whether to use padding in convolutional layers.
            Defaults to ``False``.

    Example:
        >>> import torch
        >>> from anomalib.models.image.efficient_ad.torch_model import (
        ...     MediumPatchDescriptionNetwork
        ... )
        >>> model = MediumPatchDescriptionNetwork(out_channels=384)
        >>> input_tensor = torch.randn(32, 3, 64, 64)
        >>> output = model(input_tensor)
        >>> output.shape
        torch.Size([32, 384, 13, 13])

    Note:
        The network applies ImageNet normalization to the input before
        processing.
    """

    def __init__(self, out_channels: int, padding: bool = False) -> None:
        super().__init__()
        pad_mult = 1 if padding else 0
        self.conv1 = nn.Conv2d(3, 256, kernel_size=4, stride=1, padding=3 * pad_mult)
        self.conv2 = nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=3 * pad_mult)
        self.conv3 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0 * pad_mult)
        self.conv4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1 * pad_mult)
        self.conv5 = nn.Conv2d(512, out_channels, kernel_size=4, stride=1, padding=0 * pad_mult)
        self.conv6 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0 * pad_mult)
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape ``(N, 3, H, W)``.

        Returns:
            torch.Tensor: Output feature maps of shape
                ``(N, out_channels, H', W')``, where ``H'`` and ``W'`` are
                determined by the network architecture and padding settings.
        """
        x = imagenet_norm_batch(x)
        x = F.relu(self.conv1(x))
        x = self.avgpool1(x)
        x = F.relu(self.conv2(x))
        x = self.avgpool2(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        return self.conv6(x)


class Encoder(nn.Module):
    """Encoder module for the autoencoder architecture.

    The encoder consists of 6 convolutional layers that progressively reduce the
    spatial dimensions while increasing the number of channels.

    Example:
        >>> import torch
        >>> from anomalib.models.image.efficient_ad.torch_model import Encoder
        >>> model = Encoder()
        >>> input_tensor = torch.randn(32, 3, 256, 256)
        >>> output = model(input_tensor)
        >>> output.shape
        torch.Size([32, 64, 1, 1])

    Note:
        The encoder uses ReLU activation after each convolutional layer except
        the last one.
    """

    def __init__(self) -> None:
        super().__init__()
        self.enconv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1)
        self.enconv2 = nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1)
        self.enconv3 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.enconv4 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.enconv5 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.enconv6 = nn.Conv2d(64, 64, kernel_size=8, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the encoder network.

        Args:
            x (torch.Tensor): Input tensor of shape ``(N, 3, H, W)``.

        Returns:
            torch.Tensor: Encoded features of shape ``(N, 64, H', W')``, where
                ``H'`` and ``W'`` are determined by the network architecture.
        """
        x = F.relu(self.enconv1(x))
        x = F.relu(self.enconv2(x))
        x = F.relu(self.enconv3(x))
        x = F.relu(self.enconv4(x))
        x = F.relu(self.enconv5(x))
        return self.enconv6(x)


class Decoder(nn.Module):
    """Decoder module for the autoencoder architecture.

    The decoder consists of 8 convolutional layers with upsampling that
    progressively increase spatial dimensions while maintaining or reducing
    channel dimensions.

    Args:
        out_channels (int): Number of output channels in final conv layer.
        padding (int): Whether to use padding in convolutional layers.

    Example:
        >>> import torch
        >>> from anomalib.models.image.efficient_ad.torch_model import Decoder
        >>> model = Decoder(out_channels=384, padding=True)
        >>> input_tensor = torch.randn(32, 64, 1, 1)
        >>> image_size = (256, 256)
        >>> output = model(input_tensor, image_size)
        >>> output.shape
        torch.Size([32, 384, 64, 64])

    Note:
        - Uses ReLU activation and dropout after most convolutional layers
        - Performs bilinear upsampling between conv layers to increase spatial
          dimensions
        - Final output size depends on ``padding`` parameter and input
          ``image_size``
    """

    def __init__(self, out_channels: int, padding: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.padding = padding
        # use ceil to match output shape of PDN
        self.deconv1 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv2 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv3 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv4 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv5 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv6 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv7 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.deconv8 = nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1)
        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout3 = nn.Dropout(p=0.2)
        self.dropout4 = nn.Dropout(p=0.2)
        self.dropout5 = nn.Dropout(p=0.2)
        self.dropout6 = nn.Dropout(p=0.2)

    def forward(self, x: torch.Tensor, image_size: tuple[int, int] | torch.Size) -> torch.Tensor:
        """Perform a forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape ``(N, 64, H, W)``.
            image_size (tuple[int, int] | torch.Size): Target output size
                ``(H, W)``.

        Returns:
            torch.Tensor: Decoded features of shape
                ``(N, out_channels, H', W')``, where ``H'`` and ``W'`` are
                determined by the network architecture and padding settings.
        """
        last_upsample = (
            math.ceil(image_size[0] / 4) if self.padding else math.ceil(image_size[0] / 4) - 8,
            math.ceil(image_size[1] / 4) if self.padding else math.ceil(image_size[1] / 4) - 8,
        )
        x = F.interpolate(x, size=(image_size[0] // 64 - 1, image_size[1] // 64 - 1), mode="bilinear")
        x = F.relu(self.deconv1(x))
        x = self.dropout1(x)
        x = F.interpolate(x, size=(image_size[0] // 32, image_size[1] // 32), mode="bilinear")
        x = F.relu(self.deconv2(x))
        x = self.dropout2(x)
        x = F.interpolate(x, size=(image_size[0] // 16 - 1, image_size[1] // 16 - 1), mode="bilinear")
        x = F.relu(self.deconv3(x))
        x = self.dropout3(x)
        x = F.interpolate(x, size=(image_size[0] // 8, image_size[1] // 8), mode="bilinear")
        x = F.relu(self.deconv4(x))
        x = self.dropout4(x)
        x = F.interpolate(x, size=(image_size[0] // 4 - 1, image_size[1] // 4 - 1), mode="bilinear")
        x = F.relu(self.deconv5(x))
        x = self.dropout5(x)
        x = F.interpolate(x, size=(image_size[0] // 2 - 1, image_size[1] // 2 - 1), mode="bilinear")
        x = F.relu(self.deconv6(x))
        x = self.dropout6(x)
        x = F.interpolate(x, size=last_upsample, mode="bilinear")
        x = F.relu(self.deconv7(x))
        return self.deconv8(x)


class AutoEncoder(nn.Module):
    """EfficientAd Autoencoder.

    The autoencoder consists of an encoder and decoder network. The encoder extracts features
    from the input image which are then reconstructed by the decoder.

    Args:
        out_channels (int): Number of convolution output channels in the decoder.
        padding (int): Whether to use padding in convolutional layers.
        *args: Variable length argument list passed to parent class.
        **kwargs: Arbitrary keyword arguments passed to parent class.

    Example:
        >>> from torch import randn
        >>> autoencoder = AutoEncoder(out_channels=384, padding=True)
        >>> input_tensor = randn(32, 3, 256, 256)
        >>> output = autoencoder(input_tensor, image_size=(256, 256))
        >>> output.shape
        torch.Size([32, 384, 256, 256])

    Notes:
        The input images are normalized using ImageNet statistics before being passed
        through the encoder.
    """

    def __init__(self, out_channels: int, padding: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.encoder = Encoder()
        self.decoder = Decoder(out_channels, padding)

    def forward(self, x: torch.Tensor, image_size: tuple[int, int] | torch.Size) -> torch.Tensor:
        """Forward pass through the autoencoder.

        Args:
            x (torch.Tensor): Input tensor of shape ``(N, C, H, W)``.
            image_size (tuple[int, int] | torch.Size): Target output size ``(H, W)``.

        Returns:
            torch.Tensor: Reconstructed features of shape ``(N, out_channels, H', W')``,
                where ``H'`` and ``W'`` are determined by the decoder architecture and
                padding settings.
        """
        x = imagenet_norm_batch(x)
        x = self.encoder(x)
        return self.decoder(x, image_size)


class EfficientAdModel(nn.Module):
    """EfficientAd model.

    The EfficientAd model consists of a teacher and student network for anomaly
    detection. The teacher network is pre-trained and frozen, while the student
    network is trained to match the teacher's outputs.

    Args:
        teacher_out_channels (int): Number of convolution output channels of the
            pre-trained teacher model.
        model_size (EfficientAdModelSize): Size of student and teacher model.
            Defaults to ``EfficientAdModelSize.S``.
        padding (bool): Whether to use padding in convolutional layers.
            Defaults to ``False``.
        pad_maps (bool): Whether to pad output anomaly maps when ``padding=False``
            to match size of padded case. Only relevant if ``padding=False``.
            Defaults to ``True``.

    Example:
        >>> from anomalib.models.image.efficient_ad.torch_model import (
        ...     EfficientAdModel,
        ...     EfficientAdModelSize
        ... )
        >>> model = EfficientAdModel(
        ...     teacher_out_channels=384,
        ...     model_size=EfficientAdModelSize.S
        ... )
        >>> input_tensor = torch.randn(32, 3, 256, 256)
        >>> output = model(input_tensor)
        >>> output.anomaly_map.shape
        torch.Size([32, 1, 256, 256])

    Notes:
        The model uses a student-teacher architecture where:
        - Teacher network is pre-trained and frozen
        - Student network learns to match teacher outputs
        - Autoencoder provides additional feature extraction
        - Anomaly scores are computed from student-teacher differences
    """

    def __init__(
        self,
        teacher_out_channels: int,
        model_size: EfficientAdModelSize = EfficientAdModelSize.S,
        padding: bool = False,
        pad_maps: bool = True,
    ) -> None:
        super().__init__()

        self.pad_maps = pad_maps
        self.teacher: MediumPatchDescriptionNetwork | SmallPatchDescriptionNetwork
        self.student: MediumPatchDescriptionNetwork | SmallPatchDescriptionNetwork

        if model_size == EfficientAdModelSize.M:
            self.teacher = MediumPatchDescriptionNetwork(out_channels=teacher_out_channels, padding=padding).eval()
            self.student = MediumPatchDescriptionNetwork(out_channels=teacher_out_channels * 2, padding=padding)

        elif model_size == EfficientAdModelSize.S:
            self.teacher = SmallPatchDescriptionNetwork(out_channels=teacher_out_channels, padding=padding).eval()
            self.student = SmallPatchDescriptionNetwork(out_channels=teacher_out_channels * 2, padding=padding)

        else:
            msg = f"Unknown model size {model_size}"
            raise ValueError(msg)

        self.ae: AutoEncoder = AutoEncoder(out_channels=teacher_out_channels, padding=padding)
        self.teacher_out_channels: int = teacher_out_channels

        self.mean_std: nn.ParameterDict = nn.ParameterDict(
            {
                "mean": torch.zeros((1, self.teacher_out_channels, 1, 1)),
                "std": torch.zeros((1, self.teacher_out_channels, 1, 1)),
            },
        )

        self.quantiles: nn.ParameterDict = nn.ParameterDict(
            {
                "qa_st": torch.tensor(0.0),
                "qb_st": torch.tensor(0.0),
                "qa_ae": torch.tensor(0.0),
                "qb_ae": torch.tensor(0.0),
            },
        )

    @staticmethod
    def is_set(p_dic: nn.ParameterDict) -> bool:
        """Check if any parameters in the dictionary are non-zero.

        Args:
            p_dic (nn.ParameterDict): Parameter dictionary to check.

        Returns:
            bool: ``True`` if any parameter is non-zero, ``False`` otherwise.
        """
        return any(value.sum() != 0 for _, value in p_dic.items())

    @staticmethod
    def choose_random_aug_image(image: torch.Tensor) -> torch.Tensor:
        """Apply random augmentation to input image.

        Randomly selects and applies one of: brightness, contrast or saturation
        adjustment with coefficient sampled from U(0.8, 1.2).

        Args:
            image (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Augmented image tensor.
        """
        transform_functions = [
            transforms.functional.adjust_brightness,
            transforms.functional.adjust_contrast,
            transforms.functional.adjust_saturation,
        ]
        # Sample an augmentation coefficient λ from the uniform distribution U(0.8, 1.2)
        coefficient = np.random.default_rng().uniform(0.8, 1.2)
        transform_function = np.random.default_rng().choice(transform_functions)
        return transform_function(image, coefficient)

    def forward(
        self,
        batch: torch.Tensor,
        batch_imagenet: torch.Tensor | None = None,
        normalize: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | InferenceBatch:
        """Forward pass through the model.

        Args:
            batch (torch.Tensor): Input batch of images.
            batch_imagenet (torch.Tensor | None): Optional batch of ImageNet
                images for training. Defaults to ``None``.
            normalize (bool): Whether to normalize anomaly maps.
                Defaults to ``True``.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor] | InferenceBatch:
                If training:
                    - Loss components (student-teacher, autoencoder,
                      student-autoencoder)
                If inference:
                    - Batch containing anomaly maps and scores
        """
        student_output, distance_st = self.compute_student_teacher_distance(batch)
        if self.training:
            return self.compute_losses(batch, batch_imagenet, distance_st)

        map_st, map_stae = self.compute_maps(batch, student_output, distance_st, normalize)
        anomaly_map = 0.5 * map_st + 0.5 * map_stae
        pred_score = torch.amax(anomaly_map, dim=(-2, -1))
        return InferenceBatch(pred_score=pred_score, anomaly_map=anomaly_map)

    def compute_student_teacher_distance(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the student-teacher distance vectors.

        Args:
            batch (torch.Tensor): Input batch of images.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - Student network output features
                - Squared distance between normalized teacher and student features
        """
        with torch.no_grad():
            teacher_output = self.teacher(batch)
            if self.is_set(self.mean_std):
                teacher_output = (teacher_output - self.mean_std["mean"]) / self.mean_std["std"]

        student_output = self.student(batch)
        distance_st = torch.pow(teacher_output - student_output[:, : self.teacher_out_channels, :, :], 2)
        return student_output, distance_st

    def compute_losses(
        self,
        batch: torch.Tensor,
        batch_imagenet: torch.Tensor,
        distance_st: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute training losses.

        Computes three loss components:
        - Student-teacher loss (hard examples + ImageNet penalty)
        - Autoencoder reconstruction loss
        - Student-autoencoder consistency loss

        Args:
            batch (torch.Tensor): Input batch of images.
            batch_imagenet (torch.Tensor): Batch of ImageNet images.
            distance_st (torch.Tensor): Student-teacher distances.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - Student-teacher loss
                - Autoencoder loss
                - Student-autoencoder loss
        """
        # Student loss
        distance_st = reduce_tensor_elems(distance_st)
        d_hard = torch.quantile(distance_st, 0.999)
        loss_hard = torch.mean(distance_st[distance_st >= d_hard])
        student_output_penalty = self.student(batch_imagenet)[:, : self.teacher_out_channels, :, :]
        loss_penalty = torch.mean(student_output_penalty**2)
        loss_st = loss_hard + loss_penalty

        # Autoencoder and Student AE Loss
        aug_img = self.choose_random_aug_image(batch)
        ae_output_aug = self.ae(aug_img, batch.shape[-2:])

        with torch.no_grad():
            teacher_output_aug = self.teacher(aug_img)
            if self.is_set(self.mean_std):
                teacher_output_aug = (teacher_output_aug - self.mean_std["mean"]) / self.mean_std["std"]

        student_output_ae_aug = self.student(aug_img)[:, self.teacher_out_channels :, :, :]

        distance_ae = torch.pow(teacher_output_aug - ae_output_aug, 2)
        distance_stae = torch.pow(ae_output_aug - student_output_ae_aug, 2)

        loss_ae = torch.mean(distance_ae)
        loss_stae = torch.mean(distance_stae)
        return (loss_st, loss_ae, loss_stae)

    def compute_maps(
        self,
        batch: torch.Tensor,
        student_output: torch.Tensor,
        distance_st: torch.Tensor,
        normalize: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute anomaly maps from model outputs.

        Args:
            batch (torch.Tensor): Input batch of images.
            student_output (torch.Tensor): Student network output features.
            distance_st (torch.Tensor): Student-teacher distances.
            normalize (bool): Whether to normalize maps with pre-computed
                quantiles. Defaults to ``True``.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - Student-teacher anomaly map
                - Student-autoencoder anomaly map
        """
        image_size = batch.shape[-2:]
        # Eval mode.
        with torch.no_grad():
            ae_output = self.ae(batch, image_size)

            map_st = torch.mean(distance_st, dim=1, keepdim=True)
            map_stae = torch.mean(
                (ae_output - student_output[:, self.teacher_out_channels :]) ** 2,
                dim=1,
                keepdim=True,
            )

        if self.pad_maps:
            map_st = F.pad(map_st, (4, 4, 4, 4))
            map_stae = F.pad(map_stae, (4, 4, 4, 4))
        map_st = F.interpolate(map_st, size=image_size, mode="bilinear")
        map_stae = F.interpolate(map_stae, size=image_size, mode="bilinear")

        if self.is_set(self.quantiles) and normalize:
            map_st = 0.1 * (map_st - self.quantiles["qa_st"]) / (self.quantiles["qb_st"] - self.quantiles["qa_st"])
            map_stae = 0.1 * (map_stae - self.quantiles["qa_ae"]) / (self.quantiles["qb_ae"] - self.quantiles["qa_ae"])
        return map_st, map_stae

    def get_maps(self, batch: torch.Tensor, normalize: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute anomaly maps for a batch of images.

        Convenience method that combines distance computation and map generation.

        Args:
            batch (torch.Tensor): Input batch of images.
            normalize (bool): Whether to normalize maps. Defaults to ``False``.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - Student-teacher anomaly map
                - Student-autoencoder anomaly map
        """
        student_output, distance_st = self.compute_student_teacher_distance(batch)
        return self.compute_maps(batch, student_output, distance_st, normalize)
