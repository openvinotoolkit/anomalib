"""Custom Folder Dataset.

This script creates a custom PyTorch Dataset from a folder.
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from pathlib import Path

from pandas import DataFrame
from torchvision.transforms.v2 import Transform

from anomalib import TaskType
from anomalib.data.datasets.base.image import AnomalibDataset
from anomalib.data.utils import DirType, LabelName
from anomalib.data.utils.path import _prepare_files_labels, validate_and_resolve_path


class FolderDataset(AnomalibDataset):
    """A dataset class for handling folder-based anomaly detection datasets.

    This class is designed to work with datasets organized in folder structures, where
    normal and abnormal images are stored in separate directories. It supports various
    anomaly detection tasks and can handle mask annotations for segmentation tasks.

    Args:
        name (str): Name of the dataset.
        task (TaskType): Type of the anomaly detection task (e.g., classification, detection, segmentation).
        transform (Transform | None): Transforms to be applied to the images. Defaults to ``None``.
        normal_dir (str | Path | Sequence[str | Path]): Path(s) to the directory(ies) containing normal images.
        root (str | Path | None): Root directory of the dataset. If provided, other paths will be
            resolved relative to this. Defaults to ``None``.
        abnormal_dir (str | Path | Sequence[str | Path] | None): Path(s) to the directory(ies)
            containing abnormal images. Defaults to ``None``.
        mask_dir (str | Path | Sequence[str | Path] | None): Path(s) to the directory(ies)
            containing mask annotations for segmentation tasks. Defaults to ``None``.
        extensions (tuple[str, ...] | None): File extensions to include when reading images.
            If None, all files will be considered. Defaults to ``None``.

    Examples:
        Creating a simple classification dataset:

        >>> from anomalib import TaskType
        >>> dataset = FolderDataset(
        ...     name="my_dataset",
        ...     task=TaskType.CLASSIFICATION,
        ...     normal_dir="path/to/normal",
        ...     abnormal_dir="path/to/abnormal"
        ... )
        >>> print(len(dataset))
        1000  # Assuming there are 1000 images in total
        >>> sample = dataset[0]
        >>> print(sample.keys())
        dict_keys(['image', 'image_path', 'label', 'label_index'])

        Creating a segmentation dataset with masks:

        >>> dataset = FolderDataset(
        ...     name="segmentation_dataset",
        ...     task=TaskType.SEGMENTATION,
        ...     normal_dir="path/to/normal",
        ...     abnormal_dir="path/to/abnormal",
        ...     mask_dir="path/to/masks"
        ... )
        >>> sample = dataset[0]
        >>> print(sample.keys())
        dict_keys(['image', 'image_path', 'label', 'label_index', 'mask', 'mask_path'])

        Using multiple directories for normal and abnormal images:

        >>> dataset = FolderDataset(
        ...     name="multi_dir_dataset",
        ...     task=TaskType.CLASSIFICATION,
        ...     normal_dir=["path/to/normal1", "path/to/normal2"],
        ...     abnormal_dir=["path/to/abnormal1", "path/to/abnormal2"]
        ... )

        Applying transforms to the dataset:

        >>> from torchvision.transforms import v2 as T
        >>> transform = T.Compose([
        ...     T.Resize((224, 224)),
        ...     T.ToTensor(),
        ...     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ... ])
        >>> dataset = FolderDataset(
        ...     name="transformed_dataset",
        ...     task=TaskType.CLASSIFICATION,
        ...     normal_dir="path/to/normal",
        ...     abnormal_dir="path/to/abnormal",
        ...     transform=transform
        ... )

        Using with a specific file extension:

        >>> dataset = FolderDataset(
        ...     name="jpg_only_dataset",
        ...     task=TaskType.CLASSIFICATION,
        ...     normal_dir="path/to/normal",
        ...     abnormal_dir="path/to/abnormal",
        ...     extensions=(".jpg",)
        ... )

    Note:
        - The class uses the `make_folder_dataset` function to create the initial
          DataFrame of samples.
        - For segmentation tasks, ensure that the mask_dir is provided and
          contains corresponding masks for abnormal images.
        - The transform, if provided, will be applied to both images and masks
          (for segmentation tasks).
        - The class supports both single directory and multiple directory inputs
          for normal, abnormal, and mask data.
    """

    def __init__(
        self,
        name: str,
        task: TaskType,
        transform: Transform | None = None,
        normal_dir: str | Path | Sequence[str | Path] | None = None,
        root: str | Path | None = None,
        abnormal_dir: str | Path | Sequence[str | Path] | None = None,
        mask_dir: str | Path | Sequence[str | Path] | None = None,
        extensions: tuple[str, ...] | None = None,
    ) -> None:
        super().__init__(task, transform)

        self._name = name
        self.root = root
        self.normal_dir = normal_dir
        self.abnormal_dir = abnormal_dir
        self.mask_dir = mask_dir
        self.extensions = extensions

        self.samples = make_folder_dataset(
            root=self.root,
            normal_dir=self.normal_dir,
            abnormal_dir=self.abnormal_dir,
            mask_dir=self.mask_dir,
            extensions=self.extensions,
        )

    @property
    def name(self) -> str:
        """Name of the dataset."""
        return self._name


def make_folder_dataset(
    root: str | Path | None = None,
    normal_dir: str | Path | Sequence[str | Path] | None = None,
    abnormal_dir: str | Path | Sequence[str | Path] | None = None,
    mask_dir: str | Path | Sequence[str | Path] | None = None,
    extensions: tuple[str, ...] | None = None,
) -> DataFrame:
    """Create a DataFrame containing image paths and labels for a folder-based dataset.

    This function processes normal, abnormal, and mask directories to create a structured
    DataFrame representation of the dataset. It supports various input formats for directory
    paths and handles path resolution and validation.

    Args:
        normal_dir (str | Path | Sequence[str | Path]): Path(s) to the directory(ies) containing normal images.
        root (str | Path | None, optional): Root directory of the dataset. If provided, other paths will be
            resolved relative to this. Defaults to ``None``.
        abnormal_dir (str | Path | Sequence[str | Path] | None, optional): Path(s) to the directory(ies)
            containing abnormal images. Defaults to ``None``.
        mask_dir (str | Path | Sequence[str | Path] | None, optional): Path(s) to the directory(ies)
            containing mask annotations. Defaults to ``None``.
        extensions (tuple[str, ...] | None, optional): File extensions to include when reading images.
            If None, all files will be considered. Defaults to ``None``.

    Returns:
        DataFrame: A pandas DataFrame containing the following columns:
            - image_path: str, path to the image file
            - label: str, either "normal" or "abnormal"
            - label_index: int, 0 for normal, 1 for abnormal
            - mask_path: str, path to the corresponding mask file (if applicable)

    Raises:
        FileNotFoundError: If any of the specified directories do not exist.
        ValueError: If no valid image files are found in the specified directories.

    Examples:
        Basic usage with a single normal and abnormal directory:

        >>> df = make_folder_dataset(
        ...     normal_dir='path/to/normal',
        ...     abnormal_dir='path/to/abnormal'
        ... )

        Using multiple directories for normal and abnormal images:

        >>> df = make_folder_dataset(
        ...     normal_dir=['path/to/normal1', 'path/to/normal2'],
        ...     abnormal_dir=['path/to/abnormal1', 'path/to/abnormal2']
        ... )

        Including mask annotations:

        >>> df = make_folder_dataset(
        ...     normal_dir='path/to/normal',
        ...     abnormal_dir='path/to/abnormal',
        ...     mask_dir='path/to/masks'
        ... )

        Specifying a root directory and using relative paths:

        >>> df = make_folder_dataset(
        ...     root='path/to/dataset',
        ...     normal_dir='normal',
        ...     abnormal_dir='abnormal',
        ...     mask_dir='masks'
        ... )

        Filtering images by file extension:

        >>> df = make_folder_dataset(
        ...     normal_dir='path/to/normal',
        ...     abnormal_dir='path/to/abnormal',
        ...     extensions=('.jpg', '.png')
        ... )

        Using pathlib.Path objects:

        >>> from pathlib import Path
        >>> df = make_folder_dataset(
        ...     normal_dir=Path('path/to/normal'),
        ...     abnormal_dir=Path('path/to/abnormal')
        ... )

        Handling a dataset with only normal images:

        >>> df = make_folder_dataset(
        ...     normal_dir='path/to/normal'
        ... )

    Note:
        - The function will recursively search for images in the specified directories.
        - If mask_dir is provided, it assumes a one-to-one correspondence between abnormal images and masks.
        - The resulting DataFrame is sorted by image_path for consistency.
    """

    def _resolve_path_and_convert_to_list(path: str | Path | Sequence[str | Path] | None) -> list[Path]:
        if isinstance(path, Sequence) and not isinstance(path, str):
            return [validate_and_resolve_path(dir_path, root) for dir_path in path]
        return [validate_and_resolve_path(path, root)] if path is not None else []

    if normal_dir is None:
        msg = "At least one normal directory must be provided."
        raise ValueError(msg)

    normal_dir = _resolve_path_and_convert_to_list(normal_dir)
    abnormal_dir = _resolve_path_and_convert_to_list(abnormal_dir)
    mask_dir = _resolve_path_and_convert_to_list(mask_dir)

    filenames = []
    labels = []
    dirs = {DirType.NORMAL: normal_dir}

    if abnormal_dir:
        dirs[DirType.ABNORMAL] = abnormal_dir

    if mask_dir:
        dirs[DirType.MASK] = mask_dir

    for dir_type, paths in dirs.items():
        for path in paths:
            filename, label = _prepare_files_labels(path, dir_type, extensions)
            filenames += filename
            labels += label

    samples = DataFrame({"image_path": filenames, "label": labels})
    samples = samples.sort_values(by="image_path", ignore_index=True)

    # Update label column to use LabelName instead of DirType
    samples.loc[samples.label == DirType.NORMAL, "label"] = LabelName.NORMAL
    samples.loc[samples.label == DirType.ABNORMAL, "label"] = LabelName.ABNORMAL

    # Create label index for normal (0) and abnormal (1) images.
    samples.loc[samples.label == LabelName.NORMAL, "label_index"] = LabelName.NORMAL
    samples.loc[samples.label == LabelName.ABNORMAL, "label_index"] = LabelName.ABNORMAL
    samples.label_index = samples.label_index.astype("Int64")

    # If a path to mask is provided, add it to the sample dataframe.
    if mask_dir:
        samples.loc[samples.label == LabelName.ABNORMAL, "mask_path"] = samples.loc[
            samples.label == DirType.MASK
        ].image_path.to_numpy()
        samples["mask_path"] = samples["mask_path"].fillna("")
        samples = samples.astype({"mask_path": "str"})

    # remove all the rows with mask image samples that have already been assigned
    samples = samples.loc[(samples.label == LabelName.NORMAL) | (samples.label == LabelName.ABNORMAL)]

    # Ensure the pathlib objects are converted to str.
    return samples.astype({"image_path": "str"})
