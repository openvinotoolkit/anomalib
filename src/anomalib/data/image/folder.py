"""Custom Folder Dataset.

This script creates a custom dataset from a folder.
"""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from pathlib import Path

from pandas import DataFrame
from torchvision.transforms.v2 import Transform

from anomalib import TaskType
from anomalib.data.base import AnomalibDataModule, AnomalibDataset
from anomalib.data.utils import DirType, LabelName
from anomalib.data.utils.path import _prepare_files_labels, validate_and_resolve_path
from anomalib.data.utils.split import SplitMode, TestSplitMode, ValSplitMode, resolve_split_mode


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


class Folder(AnomalibDataModule):
    """Folder DataModule for handling folder-based anomaly detection datasets.

    This DataModule is designed to work with datasets organized in folder structures, where
    normal and abnormal images are stored in separate directories. It supports various
    anomaly detection tasks and can handle mask annotations for segmentation tasks.

    The expected folder structure for a typical anomaly detection dataset is as follows:

    .. code-block:: text

        dataset_root/
        ├── normal/
        │   ├── normal_image_1.png
        │   ├── normal_image_2.png
        │   └── ...
        ├── abnormal/
        │   ├── abnormal_category_1/
        │   │   ├── abnormal_image_1.png
        │   │   ├── abnormal_image_2.png
        │   │   └── ...
        │   ├── abnormal_category_2/
        │   │   ├── abnormal_image_1.png
        │   │   ├── abnormal_image_2.png
        │   │   └── ...
        │   └── ...
        └── masks/  # Optional, for segmentation tasks
            ├── abnormal_category_1/
            │   ├── abnormal_image_1_mask.png
            │   ├── abnormal_image_2_mask.png
            │   └── ...
            ├── abnormal_category_2/
            │   ├── abnormal_image_1_mask.png
            │   ├── abnormal_image_2_mask.png
            │   └── ...
            └── ...

    Args:
        name (str): Name of the dataset.
        normal_dir (str | Path | Sequence[str | Path]): Path(s) to the directory(ies) containing normal images.
        abnormal_dir (str | Path | Sequence[str | Path] | None, optional): Path(s) to the directory(ies)
            containing abnormal images.
            Defaults to ``None``.
        mask_dir (str | Path | Sequence[str | Path] | None, optional): Path(s) to the directory(ies)
            containing mask annotations for segmentation tasks.
            Defaults to ``None``.
        root (str | Path | None, optional): Root directory of the dataset. If provided, other paths will be
            resolved relative to this. Defaults to ``None``.
        extensions (tuple[str] | None, optional): File extensions to include when reading images.
            If None, all files will be considered. Defaults to ``None``.
        train_batch_size (int, optional): Batch size for training.
            Defaults to ``32``.
        eval_batch_size (int, optional): Batch size for evaluation.
            Defaults to ``32``.
        num_workers (int, optional): Number of workers for data loading.
            Defaults to ``8``.
        task (TaskType | str, optional): Type of the anomaly detection task.
            Defaults to ``TaskType.SEGMENTATION``.
        image_size (tuple[int, int] | None, optional): Size to which images will be resized.
            Defaults to ``None``.
        transform (Transform | None, optional): Transforms to be applied to the images.
            Defaults to ``None``.
        train_transform (Transform | None, optional): Transforms to be applied only to training images.
            Defaults to ``None``.
        eval_transform (Transform | None, optional): Transforms to be applied only to evaluation images.
            Defaults to ``None``.
        test_split_mode (SplitMode | str, optional): Mode for creating the test split.
            Defaults to ``SplitMode.AUTO``.
        test_split_ratio (float | None, optional): Ratio of data to use for testing.
            Defaults to ``None``.
        val_split_mode (SplitMode | str, optional): Mode for creating the validation split.
            Defaults to ``SplitMode.AUTO``.
        val_split_ratio (float, optional): Ratio of training data to use for validation.
            Defaults to ``0.5``.
        seed (int | None, optional): Random seed for reproducibility.
            Defaults to ``None``.

    Attributes:
        SplitMode (Enum): Enumeration of available split modes:
            - SYNTHETIC: Generate synthetic data for splitting.
            - PREDEFINED: Use a pre-defined split from an existing source.
            - AUTO: Automatically determine the best splitting strategy.

    Examples:
        Basic usage with normal and abnormal directories:

        >>> from anomalib.data import Folder
        >>> datamodule = Folder(
        ...     name="my_dataset",
        ...     normal_dir="path/to/normal",
        ...     abnormal_dir="path/to/abnormal",
        ...     task="classification"
        ... )

        Using multiple directories for normal and abnormal images:

        >>> datamodule = Folder(
        ...     name="multi_dir_dataset",
        ...     normal_dir=["path/to/normal1", "path/to/normal2"],
        ...     abnormal_dir=["path/to/abnormal1", "path/to/abnormal2"],
        ...     task="classification"
        ... )

        Segmentation task with mask directory:

        >>> datamodule = Folder(
        ...     name="segmentation_dataset",
        ...     normal_dir="path/to/normal",
        ...     abnormal_dir="path/to/abnormal",
        ...     mask_dir="path/to/masks",
        ...     task="segmentation"
        ... )

        Customizing data splits using the new SplitMode:

        >>> from anomalib.data.utils import SplitMode
        >>> datamodule = Folder(
        ...     name="custom_split_dataset",
        ...     normal_dir="path/to/normal",
        ...     abnormal_dir="path/to/abnormal",
        ...     test_split_mode=SplitMode.AUTO,
        ...     val_split_mode=SplitMode.SYNTHETIC,
        ...     test_split_ratio=0.2,
        ...     val_split_ratio=0.5
        ... )

    Note:
        - For segmentation tasks, ensure that the mask_dir is provided and
          contains corresponding masks for abnormal images.
        - The class supports both single directory and multiple directory inputs
          for normal, abnormal, and mask data.
        - The old TestSplitMode and ValSplitMode enums are deprecated. Use the new
          SplitMode enum instead.
        - When using SplitMode.PREDEFINED for segmentation tasks, a mask_dir must be provided.

    Warning:
        The 'normal_test_dir' and 'normal_split_ratio' arguments are deprecated and will be
        removed in a future release. Use 'test_split_ratio' or 'val_split_ratio' with the
        appropriate SplitMode instead.
    """

    def __init__(
        self,
        name: str,
        normal_dir: str | Path | Sequence[str | Path],
        abnormal_dir: str | Path | Sequence[str | Path] | None = None,
        mask_dir: str | Path | Sequence[str | Path] | None = None,
        root: str | Path | None = None,
        extensions: tuple[str] | None = None,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        num_workers: int = 8,
        task: TaskType | str = TaskType.SEGMENTATION,
        image_size: tuple[int, int] | None = None,
        transform: Transform | None = None,
        train_transform: Transform | None = None,
        eval_transform: Transform | None = None,
        test_split_mode: SplitMode | TestSplitMode | str = SplitMode.AUTO,
        test_split_ratio: float | None = None,
        val_split_mode: SplitMode | ValSplitMode | str = SplitMode.AUTO,
        val_split_ratio: float = 0.5,
        seed: int | None = None,
        normal_test_dir: str | Path | Sequence[str | Path] | None = None,  # DEPRECATED
        normal_split_ratio: float | None = None,  # DEPRECATED
    ) -> None:
        self._name = name
        self.root = root
        self.normal_dir = normal_dir
        self.abnormal_dir = abnormal_dir
        self.mask_dir = mask_dir
        self.task = TaskType(task)
        self.extensions = extensions
        test_split_mode = resolve_split_mode(test_split_mode)
        val_split_mode = resolve_split_mode(val_split_mode)
        super().__init__(
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            test_split_mode=test_split_mode,
            test_split_ratio=test_split_ratio,
            val_split_mode=val_split_mode,
            val_split_ratio=val_split_ratio,
            image_size=image_size,
            transform=transform,
            train_transform=train_transform,
            eval_transform=eval_transform,
            seed=seed,
        )

        if normal_test_dir is not None:
            msg = (
                "The 'normal_test_dir' argument is deprecated and will be removed in a future release. "
                "If you have a dedicated train/val/test directories, please use CSV datamodule instead."
            )
            raise DeprecationWarning(msg)

        if normal_split_ratio is not None:
            msg = (
                "The 'normal_split_ratio' argument is deprecated and will be removed in a future release. "
                "Please use 'test_split_ratio' or 'val_split_ratio' instead."
            )
            raise DeprecationWarning(msg)

        if task == TaskType.SEGMENTATION and test_split_mode == TestSplitMode.FROM_DIR and mask_dir is None:
            msg = (
                f"Segmentation task requires mask directory if test_split_mode is {test_split_mode}. "
                "You could set test_split_mode to {TestSplitMode.NONE} or provide a mask directory."
            )
            raise ValueError(msg)

    def _setup(self, _stage: str | None = None) -> None:
        """Setup the Folder datamodule.

        By default, the Folder datamodule auto splits the dataset into train/val/test.
        The split will be handled by `post_setup` method in the base class.
        """
        self.train_data = FolderDataset(
            name=self.name,
            task=self.task,
            transform=self.train_transform,
            root=self.root,
            normal_dir=self.normal_dir,
            abnormal_dir=self.abnormal_dir,
            mask_dir=self.mask_dir,
            extensions=self.extensions,
        )

    @property
    def name(self) -> str:
        """Name of the datamodule.

        Folder datamodule overrides the name property to provide a custom name.
        """
        return self._name
