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
from anomalib.data.errors import MisMatchError
from anomalib.data.utils import (
    DirType,
    LabelName,
    Split,
    TestSplitMode,
    ValSplitMode,
)
from anomalib.data.utils.path import _prepare_files_labels, validate_and_resolve_path


def make_folder_dataset(
    normal_dir: str | Path | Sequence[str | Path],
    root: str | Path | None = None,
    abnormal_dir: str | Path | Sequence[str | Path] | None = None,
    normal_test_dir: str | Path | Sequence[str | Path] | None = None,
    mask_dir: str | Path | Sequence[str | Path] | None = None,
    split: str | Split | None = None,
    extensions: tuple[str, ...] | None = None,
) -> DataFrame:
    """Make Folder Dataset.

    Args:
        normal_dir (str | Path | Sequence): Path to the directory containing normal images.
        root (str | Path | None): Path to the root directory of the dataset.
            Defaults to ``None``.
        abnormal_dir (str | Path | Sequence | None, optional): Path to the directory containing abnormal images.
            Defaults to ``None``.
        normal_test_dir (str | Path | Sequence | None, optional): Path to the directory containing normal images for
            the test dataset. Normal test images will be a split of `normal_dir` if `None`.
            Defaults to ``None``.
        mask_dir (str | Path | Sequence | None, optional): Path to the directory containing the mask annotations.
            Defaults to ``None``.
        split (str | Split | None, optional): Dataset split (ie., Split.FULL, Split.TRAIN or Split.TEST).
            Defaults to ``None``.
        extensions (tuple[str, ...] | None, optional): Type of the image extensions to read from the directory.
            Defaults to ``None``.

    Returns:
        DataFrame: an output dataframe containing samples for the requested split (ie., train or test).

    Examples:
        Assume that we would like to use this ``make_folder_dataset`` to create a dataset from a folder.
        We could then create the dataset as follows,

        .. code-block:: python

            folder_df = make_folder_dataset(
                normal_dir=dataset_root / "good",
                abnormal_dir=dataset_root / "crack",
                split="train",
            )
            folder_df.head()

        .. code-block:: bash

                      image_path           label  label_index mask_path        split
            0  ./toy/good/00.jpg  DirType.NORMAL            0            Split.TRAIN
            1  ./toy/good/01.jpg  DirType.NORMAL            0            Split.TRAIN
            2  ./toy/good/02.jpg  DirType.NORMAL            0            Split.TRAIN
            3  ./toy/good/03.jpg  DirType.NORMAL            0            Split.TRAIN
            4  ./toy/good/04.jpg  DirType.NORMAL            0            Split.TRAIN
    """

    def _resolve_path_and_convert_to_list(path: str | Path | Sequence[str | Path] | None) -> list[Path]:
        """Convert path to list of paths.

        Args:
            path (str | Path | Sequence | None): Path to replace with Sequence[str | Path].

        Examples:
            >>> _resolve_path_and_convert_to_list("dir")
            [Path("path/to/dir")]
            >>> _resolve_path_and_convert_to_list(["dir1", "dir2"])
            [Path("path/to/dir1"), Path("path/to/dir2")]

        Returns:
            list[Path]: The result of path replaced by Sequence[str | Path].
        """
        if isinstance(path, Sequence) and not isinstance(path, str):
            return [validate_and_resolve_path(dir_path, root) for dir_path in path]
        return [validate_and_resolve_path(path, root)] if path is not None else []

    # All paths are changed to the List[Path] type and used.
    normal_dir = _resolve_path_and_convert_to_list(normal_dir)
    abnormal_dir = _resolve_path_and_convert_to_list(abnormal_dir)
    normal_test_dir = _resolve_path_and_convert_to_list(normal_test_dir)
    mask_dir = _resolve_path_and_convert_to_list(mask_dir)
    if len(normal_dir) == 0:
        msg = "A folder location must be provided in normal_dir."
        raise ValueError(msg)

    filenames = []
    labels = []
    dirs = {DirType.NORMAL: normal_dir}

    if abnormal_dir:
        dirs[DirType.ABNORMAL] = abnormal_dir

    if normal_test_dir:
        dirs[DirType.NORMAL_TEST] = normal_test_dir

    if mask_dir:
        dirs[DirType.MASK] = mask_dir

    for dir_type, paths in dirs.items():
        for path in paths:
            filename, label = _prepare_files_labels(path, dir_type, extensions)
            filenames += filename
            labels += label

    samples = DataFrame({"image_path": filenames, "label": labels})
    samples = samples.sort_values(by="image_path", ignore_index=True)

    # Create label index for normal (0) and abnormal (1) images.
    samples.loc[
        (samples.label == DirType.NORMAL) | (samples.label == DirType.NORMAL_TEST),
        "label_index",
    ] = LabelName.NORMAL
    samples.loc[(samples.label == DirType.ABNORMAL), "label_index"] = LabelName.ABNORMAL
    samples.label_index = samples.label_index.astype("Int64")

    # If a path to mask is provided, add it to the sample dataframe.

    if len(mask_dir) > 0 and len(abnormal_dir) > 0:
        samples.loc[samples.label == DirType.ABNORMAL, "mask_path"] = samples.loc[
            samples.label == DirType.MASK
        ].image_path.to_numpy()
        samples["mask_path"] = samples["mask_path"].fillna("")
        samples = samples.astype({"mask_path": "str"})

        # make sure all every rgb image has a corresponding mask image.
        if not (
            samples.loc[samples.label_index == LabelName.ABNORMAL]
            .apply(lambda x: Path(x.image_path).stem in Path(x.mask_path).stem, axis=1)
            .all()
        ):
            msg = """Mismatch between anomalous images and mask images. Make sure the mask files "
                     "folder follow the same naming convention as the anomalous images in the dataset "
                     "(e.g. image: '000.png', mask: '000.png')."""
            raise MisMatchError(msg)

    else:
        samples["mask_path"] = ""

    # remove all the rows with temporal image samples that have already been assigned
    samples = samples.loc[
        (samples.label == DirType.NORMAL) | (samples.label == DirType.ABNORMAL) | (samples.label == DirType.NORMAL_TEST)
    ]

    # Ensure the pathlib objects are converted to str.
    # This is because torch dataloader doesn't like pathlib.
    samples = samples.astype({"image_path": "str"})

    # Create train/test split.
    # By default, all the normal samples are assigned as train.
    #   and all the abnormal samples are test.
    samples.loc[(samples.label == DirType.NORMAL), "split"] = Split.TRAIN
    samples.loc[(samples.label == DirType.ABNORMAL) | (samples.label == DirType.NORMAL_TEST), "split"] = Split.TEST

    # Get the data frame for the split.
    if split:
        samples = samples[samples.split == split]
        samples = samples.reset_index(drop=True)

    return samples


class FolderDataset(AnomalibDataset):
    """Folder dataset.

    This class is used to create a dataset from a folder. The class utilizes the Torch Dataset class.

    Args:
        name (str): Name of the dataset. This is used to name the datamodule, especially when logging/saving.
        task (TaskType): Task type. (``classification``, ``detection`` or ``segmentation``).
        transform (Transform, optional): Transforms that should be applied to the input images.
            Defaults to ``None``.
        normal_dir (str | Path | Sequence): Path to the directory containing normal images.
        root (str | Path | None): Root folder of the dataset.
            Defaults to ``None``.
        abnormal_dir (str | Path | Sequence | None, optional): Path to the directory containing abnormal images.
            Defaults to ``None``.
        normal_test_dir (str | Path | Sequence | None, optional): Path to the directory containing
            normal images for the test dataset.
            Defaults to ``None``.
        mask_dir (str | Path | Sequence | None, optional): Path to the directory containing
            the mask annotations.
            Defaults to ``None``.
        split (str | Split | None): Fixed subset split that follows from folder structure on file system.
            Choose from [Split.FULL, Split.TRAIN, Split.TEST]
            Defaults to ``None``.
        extensions (tuple[str, ...] | None, optional): Type of the image extensions to read from the directory.
            Defaults to ``None``.

    Raises:
        ValueError: When task is set to classification and `mask_dir` is provided. When `mask_dir` is
            provided, `task` should be set to `segmentation`.

    Examples:
        Assume that we would like to use this ``FolderDataset`` to create a dataset from a folder for a classification
        task. We could first create the transforms,

        >>> from anomalib.data.utils import InputNormalizationMethod, get_transforms
        >>> transform = get_transforms(image_size=256, normalization=InputNormalizationMethod.NONE)

        We could then create the dataset as follows,

        .. code-block:: python

            folder_dataset_classification_train = FolderDataset(
                normal_dir=dataset_root / "good",
                abnormal_dir=dataset_root / "crack",
                split="train",
                transform=transform,
                task=TaskType.CLASSIFICATION,
            )

    """

    def __init__(
        self,
        name: str,
        task: TaskType,
        normal_dir: str | Path | Sequence[str | Path],
        transform: Transform | None = None,
        root: str | Path | None = None,
        abnormal_dir: str | Path | Sequence[str | Path] | None = None,
        normal_test_dir: str | Path | Sequence[str | Path] | None = None,
        mask_dir: str | Path | Sequence[str | Path] | None = None,
        split: str | Split | None = None,
        extensions: tuple[str, ...] | None = None,
    ) -> None:
        super().__init__(task, transform)

        self._name = name
        self.split = split
        self.root = root
        self.normal_dir = normal_dir
        self.abnormal_dir = abnormal_dir
        self.normal_test_dir = normal_test_dir
        self.mask_dir = mask_dir
        self.extensions = extensions

        self.samples = make_folder_dataset(
            root=self.root,
            normal_dir=self.normal_dir,
            abnormal_dir=self.abnormal_dir,
            normal_test_dir=self.normal_test_dir,
            mask_dir=self.mask_dir,
            split=self.split,
            extensions=self.extensions,
        )

    @property
    def name(self) -> str:
        """Name of the dataset.

        Folder dataset overrides the name property to provide a custom name.
        """
        return self._name


class Folder(AnomalibDataModule):
    """Folder DataModule.

    Args:
        name (str): Name of the dataset. This is used to name the datamodule, especially when logging/saving.
        normal_dir (str | Path | Sequence): Name of the directory containing normal images.
        root (str | Path | None): Path to the root folder containing normal and abnormal dirs.
            Defaults to ``None``.
        abnormal_dir (str | Path | None | Sequence): Name of the directory containing abnormal images.
            Defaults to ``None``.
        normal_test_dir (str | Path | Sequence | None, optional): Path to the directory containing
            normal images for the test dataset.
            Defaults to ``None``.
        mask_dir (str | Path | Sequence | None, optional): Path to the directory containing
            the mask annotations.
            Defaults to ``None``.
        normal_split_ratio (float, optional): Ratio to split normal training images and add to the
            test set in case test set doesn't contain any normal images.
            Defaults to 0.2.
        extensions (tuple[str, ...] | None, optional): Type of the image extensions to read from the
            directory.
            Defaults to ``None``.
        train_batch_size (int, optional): Training batch size.
            Defaults to ``32``.
        eval_batch_size (int, optional): Validation, test and predict batch size.
            Defaults to ``32``.
        num_workers (int, optional): Number of workers.
            Defaults to ``8``.
        task (TaskType, optional): Task type. Could be ``classification``, ``detection`` or ``segmentation``.
            Defaults to ``segmentation``.
        image_size (tuple[int, int], optional): Size to which input images should be resized.
            Defaults to ``None``.
        transform (Transform, optional): Transforms that should be applied to the input images.
            Defaults to ``None``.
        train_transform (Transform, optional): Transforms that should be applied to the input images during training.
            Defaults to ``None``.
        eval_transform (Transform, optional): Transforms that should be applied to the input images during evaluation.
            Defaults to ``None``.
        test_split_mode (TestSplitMode): Setting that determines how the testing subset is obtained.
            Defaults to ``TestSplitMode.FROM_DIR``.
        test_split_ratio (float): Fraction of images from the train set that will be reserved for testing.
            Defaults to ``0.2``.
        val_split_mode (ValSplitMode): Setting that determines how the validation subset is obtained.
            Defaults to ``ValSplitMode.FROM_TEST``.
        val_split_ratio (float): Fraction of train or test images that will be reserved for validation.
            Defaults to ``0.5``.
        seed (int | None, optional): Seed used during random subset splitting.
            Defaults to ``None``.

    Examples:
        The following code demonstrates how to use the ``Folder`` datamodule. Assume that the dataset is structured
        as follows:

        .. code-block:: bash

            $ tree sample_dataset
            sample_dataset
            ├── colour
            │   ├── 00.jpg
            │   ├── ...
            │   └── x.jpg
            ├── crack
            │   ├── 00.jpg
            │   ├── ...
            │   └── y.jpg
            ├── good
            │   ├── ...
            │   └── z.jpg
            ├── LICENSE
            └── mask
                ├── colour
                │   ├── ...
                │   └── x.jpg
                └── crack
                    ├── ...
                    └── y.jpg

        .. code-block:: python

            folder_datamodule = Folder(
                root=dataset_root,
                normal_dir="good",
                abnormal_dir="crack",
                task=TaskType.SEGMENTATION,
                mask_dir=dataset_root / "mask" / "crack",
                image_size=256,
                normalization=InputNormalizationMethod.NONE,
            )
            folder_datamodule.setup()

        To access the training images,

        .. code-block:: python

            >> i, data = next(enumerate(folder_datamodule.train_dataloader()))
            >> print(data.keys(), data["image"].shape)

        To access the test images,

        .. code-block:: python

            >> i, data = next(enumerate(folder_datamodule.test_dataloader()))
            >> print(data.keys(), data["image"].shape)
    """

    def __init__(
        self,
        name: str,
        normal_dir: str | Path | Sequence[str | Path],
        root: str | Path | None = None,
        abnormal_dir: str | Path | Sequence[str | Path] | None = None,
        normal_test_dir: str | Path | Sequence[str | Path] | None = None,
        mask_dir: str | Path | Sequence[str | Path] | None = None,
        normal_split_ratio: float = 0.2,
        extensions: tuple[str] | None = None,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        num_workers: int = 8,
        task: TaskType | str = TaskType.SEGMENTATION,
        image_size: tuple[int, int] | None = None,
        transform: Transform | None = None,
        train_transform: Transform | None = None,
        eval_transform: Transform | None = None,
        test_split_mode: TestSplitMode | str = TestSplitMode.FROM_DIR,
        test_split_ratio: float = 0.2,
        val_split_mode: ValSplitMode | str = ValSplitMode.FROM_TEST,
        val_split_ratio: float = 0.5,
        seed: int | None = None,
    ) -> None:
        self._name = name
        self.root = root
        self.normal_dir = normal_dir
        self.abnormal_dir = abnormal_dir
        self.normal_test_dir = normal_test_dir
        self.mask_dir = mask_dir
        self.task = TaskType(task)
        self.extensions = extensions
        test_split_mode = TestSplitMode(test_split_mode)
        val_split_mode = ValSplitMode(val_split_mode)
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

        if task == TaskType.SEGMENTATION and test_split_mode == TestSplitMode.FROM_DIR and mask_dir is None:
            msg = (
                f"Segmentation task requires mask directory if test_split_mode is {test_split_mode}. "
                "You could set test_split_mode to {TestSplitMode.NONE} or provide a mask directory."
            )
            raise ValueError(
                msg,
            )

        self.normal_split_ratio = normal_split_ratio

    def _setup(self, _stage: str | None = None) -> None:
        self.train_data = FolderDataset(
            name=self.name,
            task=self.task,
            transform=self.train_transform,
            split=Split.TRAIN,
            root=self.root,
            normal_dir=self.normal_dir,
            abnormal_dir=self.abnormal_dir,
            normal_test_dir=self.normal_test_dir,
            mask_dir=self.mask_dir,
            extensions=self.extensions,
        )

        self.test_data = FolderDataset(
            name=self.name,
            task=self.task,
            transform=self.eval_transform,
            split=Split.TEST,
            root=self.root,
            normal_dir=self.normal_dir,
            abnormal_dir=self.abnormal_dir,
            normal_test_dir=self.normal_test_dir,
            mask_dir=self.mask_dir,
            extensions=self.extensions,
        )

    @property
    def name(self) -> str:
        """Name of the datamodule.

        Folder datamodule overrides the name property to provide a custom name.
        """
        return self._name
