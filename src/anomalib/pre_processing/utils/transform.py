"""Utility functions for transforms.

This module provides utility functions for managing transforms in the pre-processing
pipeline. The utilities handle:
    - Getting and setting transforms for different pipeline stages
    - Converting between transform types
    - Managing transforms across dataloaders and datamodules

Example:
    >>> from anomalib.pre_processing.utils.transform import get_dataloaders_transforms
    >>> transforms = get_dataloaders_transforms(dataloaders)
    >>> print(transforms["train"])  # Get training stage transform
    Compose(
        Resize(size=(256, 256), ...),
        ToTensor()
    )

The module ensures consistent transform handling across the training, validation,
and testing stages of the anomaly detection pipeline.
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence

from torch.utils.data import DataLoader
from torchvision.transforms.v2 import CenterCrop, Compose, Resize, Transform

from anomalib.data import AnomalibDataModule
from anomalib.data.transforms import ExportableCenterCrop


def get_dataloaders_transforms(dataloaders: Sequence[DataLoader]) -> dict[str, Transform]:
    """Extract transforms from a sequence of dataloaders.

    This function retrieves the transforms associated with different stages (train,
    validation, test) from a sequence of dataloaders. It maps Lightning stage names
    to their corresponding transform stages.

    The stage mapping is:
        - ``fit`` -> ``train``
        - ``validate`` -> ``val``
        - ``test`` -> ``test``
        - ``predict`` -> ``test``

    Args:
        dataloaders: A sequence of PyTorch :class:`DataLoader` objects to extract
            transforms from. Each dataloader should have a ``dataset`` attribute
            with a ``transform`` property.

    Returns:
        A dictionary mapping stage names (``train``, ``val``, ``test``) to their
        corresponding :class:`torchvision.transforms.v2.Transform` objects.

    Example:
        >>> from torch.utils.data import DataLoader
        >>> from torchvision.transforms.v2 import Resize, ToTensor
        >>> # Create dataloaders with transforms
        >>> train_loader = DataLoader(dataset_with_transform)
        >>> val_loader = DataLoader(dataset_with_transform)
        >>> # Get transforms
        >>> transforms = get_dataloaders_transforms([train_loader, val_loader])
        >>> print(transforms["train"])  # Access training transform
        Compose(
            Resize(size=(256, 256)),
            ToTensor()
        )
    """
    transforms: dict[str, Transform] = {}
    stage_lookup = {
        "fit": "train",
        "validate": "val",
        "test": "test",
        "predict": "test",
    }

    for dataloader in dataloaders:
        if not hasattr(dataloader, "dataset") or not hasattr(dataloader.dataset, "transform"):
            continue

        for stage in stage_lookup:
            if hasattr(dataloader, f"{stage}_dataloader"):
                transforms[stage_lookup[stage]] = dataloader.dataset.transform

    return transforms


def set_dataloaders_transforms(dataloaders: Sequence[DataLoader], transforms: dict[str, Transform | None]) -> None:
    """Set transforms to dataloaders based on their stage.

    This function propagates transforms to dataloaders based on their stage mapping.
    The stage mapping follows the convention:

        - ``fit`` -> ``train``
        - ``validate`` -> ``val``
        - ``test`` -> ``test``
        - ``predict`` -> ``test``

    Args:
        dataloaders: A sequence of PyTorch :class:`DataLoader` objects to set
            transforms for. Each dataloader should have a ``dataset`` attribute.
        transforms: Dictionary mapping stage names (``train``, ``val``, ``test``)
            to their corresponding :class:`torchvision.transforms.v2.Transform`
            objects. The transforms can be ``None``.

    Example:
        >>> from torch.utils.data import DataLoader
        >>> from torchvision.transforms.v2 import Resize, ToTensor
        >>> # Create transforms
        >>> transforms = {
        ...     "train": Compose([Resize((256, 256)), ToTensor()]),
        ...     "val": Compose([Resize((256, 256)), ToTensor()])
        ... }
        >>> # Create dataloaders
        >>> train_loader = DataLoader(dataset_with_transform)
        >>> val_loader = DataLoader(dataset_with_transform)
        >>> # Set transforms
        >>> set_dataloaders_transforms([train_loader, val_loader], transforms)
    """
    stage_mapping = {
        "fit": "train",
        "validate": "val",
        "test": "test",
        "predict": "test",  # predict uses test transform
    }

    for loader in dataloaders:
        if not hasattr(loader, "dataset"):
            continue

        for stage in stage_mapping:
            if hasattr(loader, f"{stage}_dataloader"):
                transform = transforms.get(stage_mapping[stage])
                if transform is not None:
                    set_dataloader_transform([loader], transform)


def set_dataloader_transform(dataloader: DataLoader | Sequence[DataLoader], transform: Transform) -> None:
    """Set a transform for a dataloader or sequence of dataloaders.

    This function sets the transform for either a single dataloader or multiple dataloaders.
    The transform is set on the dataset object of each dataloader if it has a ``transform``
    attribute.

    Args:
        dataloader: A single :class:`torch.utils.data.DataLoader` or a sequence of
            dataloaders to set the transform for. Each dataloader should have a
            ``dataset`` attribute with a ``transform`` attribute.
        transform: The :class:`torchvision.transforms.v2.Transform` object to set as
            the transform.

    Raises:
        TypeError: If ``dataloader`` is neither a :class:`torch.utils.data.DataLoader`
            nor a sequence of dataloaders.

    Example:
        >>> from torch.utils.data import DataLoader
        >>> from torchvision.transforms.v2 import Resize
        >>> # Create transform and dataloader
        >>> transform = Resize(size=(256, 256))
        >>> dataloader = DataLoader(dataset_with_transform)
        >>> # Set transform
        >>> set_dataloader_transform(dataloader, transform)
    """
    if isinstance(dataloader, DataLoader):
        if hasattr(dataloader.dataset, "transform"):
            dataloader.dataset.transform = transform
    elif isinstance(dataloader, Sequence):
        for dl in dataloader:
            set_dataloader_transform(dl, transform)
    else:
        msg = f"Unsupported dataloader type: {type(dataloader)}"
        raise TypeError(msg)


def set_datamodule_stage_transform(datamodule: AnomalibDataModule, transform: Transform, stage: str) -> None:
    """Set a transform for a specific stage in a :class:`AnomalibDataModule`.

    This function sets the transform for a specific stage (train/val/test/predict) in an
    AnomalibDataModule by mapping the stage name to the corresponding dataset attribute
    and setting its transform.

    Args:
        datamodule: The :class:`AnomalibDataModule` instance to set the transform for.
            Must have dataset attributes corresponding to different stages.
        transform: The :class:`torchvision.transforms.v2.Transform` object to set as
            the transform for the specified stage.
        stage: The pipeline stage to set the transform for. Must be one of:
            ``'fit'``, ``'validate'``, ``'test'``, or ``'predict'``.

    Note:
        The ``stage`` parameter maps to dataset attributes as follows:

        - ``'fit'`` -> ``'train_data'``
        - ``'validate'`` -> ``'val_data'``
        - ``'test'`` -> ``'test_data'``
        - ``'predict'`` -> ``'test_data'``

    Example:
        >>> from torchvision.transforms.v2 import Resize
        >>> from anomalib.data import MVTec
        >>> # Create transform and datamodule
        >>> transform = Resize(size=(256, 256))
        >>> datamodule = MVTec()
        >>> # Set transform for training stage
        >>> set_datamodule_stage_transform(datamodule, transform, "fit")
    """
    stage_datasets = {
        "fit": "train_data",
        "validate": "val_data",
        "test": "test_data",
        "predict": "test_data",
    }

    dataset_attr = stage_datasets.get(stage)
    if dataset_attr and hasattr(datamodule, dataset_attr):
        dataset = getattr(datamodule, dataset_attr)
        if hasattr(dataset, "transform"):
            dataset.transform = transform


def get_exportable_transform(transform: Transform | None) -> Transform | None:
    """Get an exportable version of a transform.

    This function converts a torchvision transform into a format that is compatible with
    ONNX and OpenVINO export. It handles two main compatibility issues:

    1. Disables antialiasing in ``Resize`` transforms
    2. Converts ``CenterCrop`` to ``ExportableCenterCrop``

    Args:
        transform (Transform | None): The transform to convert. If ``None``, returns
            ``None``.

    Returns:
        Transform | None: The converted transform that is compatible with ONNX/OpenVINO
            export. Returns ``None`` if input transform is ``None``.

    Example:
        >>> from torchvision.transforms.v2 import Compose, Resize, CenterCrop
        >>> transform = Compose([
        ...     Resize((224, 224), antialias=True),
        ...     CenterCrop(200)
        ... ])
        >>> exportable = get_exportable_transform(transform)
        >>> # Now transform is compatible with ONNX/OpenVINO export

    Note:
        Some torchvision transforms are not directly supported by ONNX/OpenVINO. This
        function handles the most common cases, but additional transforms may need
        special handling.
    """
    if transform is None:
        return None
    transform = disable_antialiasing(transform)
    return convert_center_crop_transform(transform)


def disable_antialiasing(transform: Transform) -> Transform:
    """Disable antialiasing in Resize transforms.

    This function recursively disables antialiasing in any ``Resize`` transforms found
    within the provided transform or transform composition. This is necessary because
    antialiasing is not supported during ONNX export.

    Args:
        transform (Transform): Transform or composition of transforms to process.

    Returns:
        Transform: The processed transform with antialiasing disabled in any
            ``Resize`` transforms.

    Example:
        >>> from torchvision.transforms.v2 import Compose, Resize
        >>> transform = Compose([
        ...     Resize((224, 224), antialias=True),
        ...     Resize((256, 256), antialias=True)
        ... ])
        >>> transform = disable_antialiasing(transform)
        >>> # Now all Resize transforms have antialias=False

    Note:
        This function modifies the transforms in-place by setting their
        ``antialias`` attribute to ``False``. The original transform object is
        returned.
    """
    if isinstance(transform, Resize):
        transform.antialias = False
    if isinstance(transform, Compose):
        for tr in transform.transforms:
            disable_antialiasing(tr)
    return transform


def convert_center_crop_transform(transform: Transform) -> Transform:
    """Convert torchvision's CenterCrop to ExportableCenterCrop.

    This function recursively converts any ``CenterCrop`` transforms found within the
    provided transform or transform composition to ``ExportableCenterCrop``. This is
    necessary because torchvision's ``CenterCrop`` is not supported during ONNX
    export.

    Args:
        transform (Transform): Transform or composition of transforms to process.

    Returns:
        Transform: The processed transform with all ``CenterCrop`` transforms
            converted to ``ExportableCenterCrop``.

    Example:
        >>> from torchvision.transforms.v2 import Compose, CenterCrop
        >>> transform = Compose([
        ...     CenterCrop(224),
        ...     CenterCrop((256, 256))
        ... ])
        >>> transform = convert_center_crop_transform(transform)
        >>> # Now all CenterCrop transforms are converted to ExportableCenterCrop

    Note:
        This function creates new ``ExportableCenterCrop`` instances to replace the
        original ``CenterCrop`` transforms. The original transform object is
        returned with the replacements applied.
    """
    if isinstance(transform, CenterCrop):
        transform = ExportableCenterCrop(size=transform.size)
    if isinstance(transform, Compose):
        for index in range(len(transform.transforms)):
            tr = transform.transforms[index]
            transform.transforms[index] = convert_center_crop_transform(tr)
    return transform
