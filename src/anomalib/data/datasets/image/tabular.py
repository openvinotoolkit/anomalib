# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Custom Tabular Dataset.

This module provides a custom PyTorch Dataset implementation for loading
images using a selection of paths and labels defined in a table or tabular file.
It does not require a specific folder structure and allows subsampling and
relabeling without moving files. The dataset supports both classification and
segmentation tasks.

The table should contain columns for ``image_paths``, ``label_index``, ``split``,
and optionally ``masks_paths`` for segmentation tasks.

Example:
    >>> from anomalib.data.datasets import TabularDataset
    >>> samples = {
    ...     "image_path": ["images/image1.png", "images/image2.png", "images/image3.png", ... ],
    ...     "label_index": [LabelName.NORMAL, LabelName.NORMAL, LabelName.ABNORMAL,  ... ],
    ...     "split": [Split.TRAIN, Split.TRAIN, Split.TEST, ... ],
    ... }
    >>> dataset = TabularDataset(
    ...     name="custom",
    ...     samples=samples,
    ...     root="./datasets/custom",
    ... )
"""

from functools import partial
from pathlib import Path

from pandas import DataFrame
from torchvision.transforms.v2 import Transform

from anomalib.data.datasets.base.image import AnomalibDataset
from anomalib.data.errors import MisMatchError
from anomalib.data.utils import DirType, LabelName, Split
from anomalib.data.utils.path import resolve_path_under_root


class TabularDataset(AnomalibDataset):
    """Dataset class for loading images from paths and labels defined in a table.

    Args:
        name (str): Name of the dataset. Used for logging/saving.
        samples (dict | list | DataFrame): Pandas ``DataFrame`` or compatible ``list``
            or ``dict`` containing the dataset information.
        augmentations (Transform | None, optional): Augmentations to apply to the images.
            Defaults to ``None``.
        root (str | Path | None, optional): Root directory of the dataset.
            Defaults to ``None``.
        split (str | Split | None, optional): Dataset split to load.
            Choose from ``Split.FULL``, ``Split.TRAIN``, ``Split.TEST``.
            Defaults to ``None``.

    Examples:
        Create a classification dataset:

        >>> from anomalib.data.utils import InputNormalizationMethod, get_transforms
        >>> from anomalib.data.datasets import TabularDataset
        >>> transform = get_transforms(
        ...     image_size=256,
        ...     normalization=InputNormalizationMethod.NONE
        ... )
        >>> samples = {
        ...     "image_path": ["images/image1.png", "images/image2.png", "images/image3.png", ... ],
        ...     "label_index": [LabelName.NORMAL, LabelName.NORMAL, LabelName.ABNORMAL,  ... ],
        ...     "split": [Split.TRAIN, Split.TRAIN, Split.TEST, ... ],
        ... }
        >>> dataset = TabularDataset(
        ...     name="custom",
        ...     samples=samples,
        ...     root="./datasets/custom",
        ...     transform=transform
        ... )

        Create a segmentation dataset:

        >>> samples = {
        ...     "image_path": ["images/image1.png", "images/image2.png", "images/image3.png", ... ],
        ...     "label_index": [LabelName.NORMAL, LabelName.NORMAL, LabelName.ABNORMAL,  ... ],
        ...     "split": [Split.TRAIN, Split.TRAIN, Split.TEST, ... ],
        ...     "mask_path": ["masks/mask1.png", "masks/mask2.png", "masks/mask3.png", ... ],
        ... }
        >>> dataset = TabularDataset(
        ...     name="custom",
        ...     samples=samples,
        ...     root="./datasets/custom",
        ...     transform=transform
        ... )
    """

    def __init__(
        self,
        name: str,
        samples: dict | list | DataFrame,
        augmentations: Transform | None = None,
        root: str | Path | None = None,
        split: str | Split | None = None,
    ) -> None:
        super().__init__(augmentations=augmentations)

        self._name = name
        self.split = split
        self.root = root
        self.samples = make_tabular_dataset(
            samples=samples,
            root=self.root,
            split=self.split,
        )

    @property
    def name(self) -> str:
        """Get dataset name.

        Returns:
            str: Name of the dataset
        """
        return self._name


def make_tabular_dataset(
    samples: dict | list | DataFrame,
    root: str | Path | None = None,
    split: str | Split | None = None,
) -> DataFrame:
    """Create a dataset from a table of image paths and labels.

    Args:
        samples (dict | list | DataFrame): Pandas ``DataFrame`` or compatible
            ``list`` or ``dict`` containing the dataset information.
        root (str | Path | None, optional): Root directory of the dataset.
            Defaults to ``None``.
        split (str | Split | None, optional): Dataset split to load.
            Choose from ``Split.FULL``, ``Split.TRAIN``, ``Split.TEST``.
            Defaults to ``None``.

    Returns:
        DataFrame: Dataset samples with columns for image paths, labels, splits
            and mask paths (for segmentation).

    Examples:
        Create a classification dataset:
        >>> samples = {
        ...     "image_path": ["images/00.png", "images/01.png", "images/02.png", ... ],
        ...     "label_index": [LabelName.NORMAL, LabelName.NORMAL, LabelName.NORMAL,  ... ],
        ...     "split": [Split.TRAIN, Split.TRAIN, Split.TRAIN, ... ],
        ... }
        >>> tabular_df = make_tabular_dataset(
        ...     samples=samples,
        ...     root="./datasets/custom",
        ...     split=Split.TRAIN,
        ... )
        >>> tabular_df.head()
           image_path                         label            label_index    mask_path    split
        0  ./datasets/custom/images/00.png    DirType.NORMAL    0                           Split.TRAIN
        1  ./datasets/custom/images/01.png    DirType.NORMAL    0                           Split.TRAIN
        2  ./datasets/custom/images/02.png    DirType.NORMAL    0                           Split.TRAIN
        3  ./datasets/custom/images/03.png    DirType.NORMAL    0                           Split.TRAIN
        4  ./datasets/custom/images/04.png    DirType.NORMAL    0                           Split.TRAIN
    """
    ######################
    ### Pre-processing ###
    ######################

    # Convert to pandas DataFrame if dictionary or list is given
    if isinstance(samples, dict | list):
        samples = DataFrame(samples)
    if "image_path" not in samples.columns:
        msg = "The samples table must contain an 'image_path' column."
        raise ValueError(msg)
    samples = samples.sort_values(by="image_path", ignore_index=True)

    ###########################
    ### Add missing columns ###
    ###########################

    # Adding missing columns successively:
    # The user can provide one or more of columns 'label_index', 'label', and 'split'.
    # The missing columns will be inferred from the provided columns by predefined rules.

    if "label_index" in samples.columns:
        samples.label_index = samples.label_index.astype("Int64")

    columns_present = [col in samples.columns for col in ["label_index", "label", "split"]]

    # all columns missing
    if columns_present == [
        False,  # label_index
        False,  # label
        False,  # split
    ]:
        msg = "The samples table must contain at least one of 'label_index', 'label' or 'split' columns."
        raise ValueError(msg)

    # label_index missing (split can be present or missing, therefore only first two values are checked)
    if columns_present[:2] == [
        False,  # label_index
        True,  # label
    ]:
        label_to_label_index = {
            DirType.ABNORMAL: LabelName.ABNORMAL,
            DirType.NORMAL: LabelName.NORMAL,
            DirType.NORMAL_TEST: LabelName.NORMAL,
        }
        samples["label_index"] = samples["label"].map(label_to_label_index).astype("Int64")

    # label_index and label missing
    elif columns_present == [
        False,  # label_index
        False,  # label
        True,  # split
    ]:
        split_to_label_index = {
            Split.TRAIN: LabelName.NORMAL,
            Split.TEST: LabelName.ABNORMAL,
        }
        samples["label_index"] = samples["split"].map(split_to_label_index).astype("Int64")

    # label and split missing
    elif columns_present == [
        True,  # label_index
        False,  # label
        False,  # split
    ]:
        label_index_to_label = {
            LabelName.ABNORMAL: DirType.ABNORMAL,
            LabelName.NORMAL: DirType.NORMAL,
        }
        samples["label"] = samples["label_index"].map(label_index_to_label)

    # reevaluate columns_present in case a column was added in the previous control flow
    columns_present = [col in samples.columns for col in ["label_index", "label", "split"]]
    # label missing
    if columns_present == [
        True,  # label_index
        False,  # label
        True,  # split
    ]:
        samples["label"] = samples.apply(
            lambda x: (
                DirType.NORMAL
                if (x["label_index"] == LabelName.NORMAL) and (x["split"] == Split.TRAIN.value)
                else (
                    DirType.NORMAL_TEST
                    if x["label_index"] == LabelName.NORMAL and x["split"] == Split.TEST.value
                    else (DirType.ABNORMAL if x["label_index"] == LabelName.ABNORMAL else None)
                )
            ),
            axis=1,
        )
    # split missing
    elif columns_present == [
        True,  # label_index
        True,  # label
        False,  # split
    ]:
        label_to_split = {
            DirType.NORMAL: Split.TRAIN,
            DirType.ABNORMAL: Split.TEST,
            DirType.NORMAL_TEST: Split.TEST,
        }
        samples["split"] = samples["label"].map(label_to_split)

    # Add mask_path column if not exists
    if "mask_path" not in samples.columns:
        samples["mask_path"] = ""

    #######################
    ### Post-processing ###
    #######################

    # Add root to paths. When ``root`` is set, paths must resolve under it;
    # absolute paths outside ``root`` are rejected.
    samples["mask_path"] = samples["mask_path"].fillna("")
    if root is not None:
        resolve = partial(resolve_path_under_root, root, should_exist=False)
        samples["image_path"] = samples["image_path"].map(resolve)
        samples.loc[
            samples["mask_path"] != "",
            "mask_path",
        ] = samples.loc[samples["mask_path"] != "", "mask_path"].map(resolve)
    samples = samples.astype({"image_path": "str", "mask_path": "str", "label": "str"})

    # Check if anomalous samples are in training set
    if ((samples.label_index == LabelName.ABNORMAL) & (samples.split == Split.TRAIN.value)).any():
        msg = "Training set must not contain anomalous samples."
        raise MisMatchError(msg)

    # Check for None or NaN values
    if samples.isna().any().any():
        msg = "The samples table contains None or NaN values."
        raise ValueError(msg)

    # Infer the task type
    samples.attrs["task"] = "classification" if (samples["mask_path"] == "").all() else "segmentation"

    # Get the dataframe for the split.
    if split:
        split_value = split.value if isinstance(split, Split) else split
        samples = samples[samples.split == split_value]
        samples = samples.reset_index(drop=True)

    return samples
