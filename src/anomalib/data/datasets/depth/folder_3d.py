"""Custom Folder Dataset for 3D anomaly detection.

This module provides a custom dataset class that loads RGB-D data from a folder
structure. The dataset supports both classification and segmentation tasks.

The folder structure should contain RGB images and their corresponding depth maps.
The dataset can be configured with separate directories for:

- Normal training samples
- Normal test samples (optional)
- Abnormal test samples (optional)
- Mask annotations (optional, for segmentation)
- Depth maps for each image type

Example:
    >>> from pathlib import Path
    >>> from anomalib.data.datasets import Folder3DDataset
    >>> dataset = Folder3DDataset(
    ...     name="custom",
    ...     root="datasets/custom",
    ...     normal_dir="normal",
    ...     abnormal_dir="abnormal",
    ...     normal_depth_dir="normal_depth",
    ...     abnormal_depth_dir="abnormal_depth",
    ...     mask_dir="ground_truth"
    ... )
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from pandas import DataFrame, isna
from torchvision.transforms.v2 import Transform

from anomalib.data.datasets.base.depth import AnomalibDepthDataset
from anomalib.data.errors import MisMatchError
from anomalib.data.utils import DirType, LabelName, Split
from anomalib.data.utils.path import _prepare_files_labels, validate_and_resolve_path


class Folder3DDataset(AnomalibDepthDataset):
    """Dataset class for loading RGB-D data from a custom folder structure.

    Args:
        name (str): Name of the dataset
        normal_dir (str | Path): Path to directory containing normal images
        root (str | Path | None, optional): Root directory of the dataset.
            Defaults to ``None``.
        abnormal_dir (str | Path | None, optional): Path to directory containing
            abnormal images. Defaults to ``None``.
        normal_test_dir (str | Path | None, optional): Path to directory
            containing normal test images. If not provided, normal test images
            will be split from ``normal_dir``. Defaults to ``None``.
        mask_dir (str | Path | None, optional): Path to directory containing
            ground truth masks. Required for segmentation. Defaults to ``None``.
        normal_depth_dir (str | Path | None, optional): Path to directory
            containing depth maps for normal images. Defaults to ``None``.
        abnormal_depth_dir (str | Path | None, optional): Path to directory
            containing depth maps for abnormal images. Defaults to ``None``.
        normal_test_depth_dir (str | Path | None, optional): Path to directory
            containing depth maps for normal test images. Defaults to ``None``.
        augmentations (Transform, optional): Augmentations that should be applied to the input images.
            Defaults to ``None``.
        split (str | Split | None, optional): Dataset split to load.
            One of ``["train", "test", "full"]``. Defaults to ``None``.
        extensions (tuple[str, ...] | None, optional): Image file extensions to
            include. Defaults to ``None``.

    Example:
        >>> dataset = Folder3DDataset(
        ...     name="custom",
        ...     root="./datasets/custom",
        ...     normal_dir="train/good",
        ...     abnormal_dir="test/defect",
        ...     mask_dir="test/ground_truth",
        ...     normal_depth_dir="train/good_depth",
        ...     abnormal_depth_dir="test/defect_depth"
        ... )
    """

    def __init__(
        self,
        name: str,
        normal_dir: str | Path,
        root: str | Path | None = None,
        abnormal_dir: str | Path | None = None,
        normal_test_dir: str | Path | None = None,
        mask_dir: str | Path | None = None,
        normal_depth_dir: str | Path | None = None,
        abnormal_depth_dir: str | Path | None = None,
        normal_test_depth_dir: str | Path | None = None,
        augmentations: Transform | None = None,
        split: str | Split | None = None,
        extensions: tuple[str, ...] | None = None,
    ) -> None:
        super().__init__(augmentations=augmentations)

        self._name = name
        self.split = split
        self.root = root
        self.normal_dir = normal_dir
        self.abnormal_dir = abnormal_dir
        self.normal_test_dir = normal_test_dir
        self.mask_dir = mask_dir
        self.normal_depth_dir = normal_depth_dir
        self.abnormal_depth_dir = abnormal_depth_dir
        self.normal_test_depth_dir = normal_test_depth_dir
        self.extensions = extensions

        self.samples = make_folder3d_dataset(
            root=self.root,
            normal_dir=self.normal_dir,
            abnormal_dir=self.abnormal_dir,
            normal_test_dir=self.normal_test_dir,
            mask_dir=self.mask_dir,
            normal_depth_dir=self.normal_depth_dir,
            abnormal_depth_dir=self.abnormal_depth_dir,
            normal_test_depth_dir=self.normal_test_depth_dir,
            split=self.split,
            extensions=self.extensions,
        )

    @property
    def name(self) -> str:
        """Get dataset name.

        Returns:
            str: Name of the dataset
        """
        return self._name


def make_folder3d_dataset(
    normal_dir: str | Path,
    root: str | Path | None = None,
    abnormal_dir: str | Path | None = None,
    normal_test_dir: str | Path | None = None,
    mask_dir: str | Path | None = None,
    normal_depth_dir: str | Path | None = None,
    abnormal_depth_dir: str | Path | None = None,
    normal_test_depth_dir: str | Path | None = None,
    split: str | Split | None = None,
    extensions: tuple[str, ...] | None = None,
) -> DataFrame:
    """Create a dataset by collecting files from a folder structure.

    The function creates a DataFrame containing paths to RGB images, depth maps,
    and masks (if available) along with their corresponding labels.

    Args:
        normal_dir (str | Path): Directory containing normal images
        root (str | Path | None, optional): Root directory. Defaults to ``None``.
        abnormal_dir (str | Path | None, optional): Directory containing abnormal
            images. Defaults to ``None``.
        normal_test_dir (str | Path | None, optional): Directory containing
            normal test images. Defaults to ``None``.
        mask_dir (str | Path | None, optional): Directory containing ground truth
            masks. Defaults to ``None``.
        normal_depth_dir (str | Path | None, optional): Directory containing
            depth maps for normal images. Defaults to ``None``.
        abnormal_depth_dir (str | Path | None, optional): Directory containing
            depth maps for abnormal images. Defaults to ``None``.
        normal_test_depth_dir (str | Path | None, optional): Directory containing
            depth maps for normal test images. Defaults to ``None``.
        split (str | Split | None, optional): Dataset split to return.
            Defaults to ``None``.
        extensions (tuple[str, ...] | None, optional): Image file extensions to
            include. Defaults to ``None``.

    Returns:
        DataFrame: Dataset samples with columns for paths and labels

    Raises:
        ValueError: If ``normal_dir`` is not a directory
        FileNotFoundError: If depth maps or mask files are missing
        MisMatchError: If depth maps don't match their RGB images
    """
    normal_dir = validate_and_resolve_path(normal_dir, root)
    abnormal_dir = validate_and_resolve_path(abnormal_dir, root) if abnormal_dir else None
    normal_test_dir = validate_and_resolve_path(normal_test_dir, root) if normal_test_dir else None
    mask_dir = validate_and_resolve_path(mask_dir, root) if mask_dir else None
    normal_depth_dir = validate_and_resolve_path(normal_depth_dir, root) if normal_depth_dir else None
    abnormal_depth_dir = validate_and_resolve_path(abnormal_depth_dir, root) if abnormal_depth_dir else None
    normal_test_depth_dir = validate_and_resolve_path(normal_test_depth_dir, root) if normal_test_depth_dir else None

    if not normal_dir.is_dir():
        msg = "A folder location must be provided in normal_dir."
        raise ValueError(msg)

    dirs = {
        DirType.NORMAL: normal_dir,
        DirType.ABNORMAL: abnormal_dir,
        DirType.NORMAL_TEST: normal_test_dir,
        DirType.NORMAL_DEPTH: normal_depth_dir,
        DirType.ABNORMAL_DEPTH: abnormal_depth_dir,
        DirType.NORMAL_TEST_DEPTH: normal_test_depth_dir,
        DirType.MASK: mask_dir,
    }

    filenames: list[Path] = []
    labels: list[str] = []

    for dir_type, dir_path in dirs.items():
        if dir_path is not None:
            filename, label = _prepare_files_labels(dir_path, dir_type, extensions)
            filenames += filename
            labels += label

    samples = DataFrame({"image_path": filenames, "label": labels})
    samples = samples.sort_values(by="image_path", ignore_index=True)

    samples.loc[
        (samples.label == DirType.NORMAL) | (samples.label == DirType.NORMAL_TEST),
        "label_index",
    ] = LabelName.NORMAL
    samples.loc[(samples.label == DirType.ABNORMAL), "label_index"] = LabelName.ABNORMAL
    samples.label_index = samples.label_index.astype("Int64")

    # If a path to mask is provided, add it to the sample dataframe.
    if normal_depth_dir:
        samples.loc[samples.label == DirType.NORMAL, "depth_path"] = samples.loc[
            samples.label == DirType.NORMAL_DEPTH
        ].image_path.to_numpy()
        samples.loc[samples.label == DirType.ABNORMAL, "depth_path"] = samples.loc[
            samples.label == DirType.ABNORMAL_DEPTH
        ].image_path.to_numpy()

        if normal_test_dir:
            samples.loc[samples.label == DirType.NORMAL_TEST, "depth_path"] = samples.loc[
                samples.label == DirType.NORMAL_TEST_DEPTH
            ].image_path.to_numpy()

        # make sure every rgb image has a corresponding depth image and that the file exists
        mismatch = (
            samples.loc[samples.label_index == LabelName.ABNORMAL]
            .apply(lambda x: Path(x.image_path).stem in Path(x.depth_path).stem, axis=1)
            .all()
        )
        if not mismatch:
            msg = (
                "Mismatch between anomalous images and depth images. "
                "Make sure the mask files in 'xyz' folder follow the same naming "
                "convention as the anomalous images in the dataset"
                "(e.g. image: '000.png', depth: '000.tiff')."
            )
            raise MisMatchError(msg)

        missing_depth_files = samples.depth_path.apply(
            lambda x: Path(x).exists() if not isna(x) else True,
        ).all()
        if not missing_depth_files:
            msg = "Missing depth image files."
            raise FileNotFoundError(msg)

        samples = samples.astype({"depth_path": "str"})

    # If a path to mask is provided, add it to the sample dataframe.
    if mask_dir and abnormal_dir:
        samples.loc[samples.label == DirType.ABNORMAL, "mask_path"] = samples.loc[
            samples.label == DirType.MASK
        ].image_path.to_numpy()
        samples["mask_path"] = samples["mask_path"].fillna("")
        samples = samples.astype({"mask_path": "str"})

        # Make sure all the files exist
        if not samples.mask_path.apply(
            lambda x: Path(x).exists() if x != "" else True,
        ).all():
            msg = f"Missing mask files. mask_dir={mask_dir}"
            raise FileNotFoundError(msg)
    else:
        samples["mask_path"] = ""

    # Remove all the rows with temporal image samples that have already been assigned
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

    # infer the task type
    samples.attrs["task"] = "classification" if (samples["mask_path"] == "").all() else "segmentation"

    # Get the data frame for the split.
    if split:
        samples = samples[samples.split == split]
        samples = samples.reset_index(drop=True)

    return samples
