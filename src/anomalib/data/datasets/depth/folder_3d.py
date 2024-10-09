"""Custom Folder Dataset.

This script creates a custom dataset from a folder.
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from pandas import DataFrame, isna

from anomalib import TaskType
from anomalib.data.datasets.base.depth import AnomalibDepthDataset
from anomalib.data.errors import MisMatchError
from anomalib.data.utils import DirType, LabelName, Split
from anomalib.data.utils.path import _prepare_files_labels, validate_and_resolve_path


class Folder3DDataset(AnomalibDepthDataset):
    """Folder dataset.

    Args:
        name (str): Name of the dataset.
        task (TaskType): Task type. (``classification``, ``detection`` or ``segmentation``).
        normal_dir (str | Path): Path to the directory containing normal images.
        root (str | Path | None): Root folder of the dataset.
            Defaults to ``None``.
        abnormal_dir (str | Path | None, optional): Path to the directory containing abnormal images.
            Defaults to ``None``.
        normal_test_dir (str | Path | None, optional): Path to the directory containing
            normal images for the test dataset.
            Defaults to ``None``.
        mask_dir (str | Path | None, optional): Path to the directory containing
            the mask annotations.
            Defaults to ``None``.
        normal_depth_dir (str | Path | None, optional): Path to the directory containing
            normal depth images for the test dataset. Normal test depth images will be a split of `normal_dir`
            Defaults to ``None``.
        abnormal_depth_dir (str | Path | None, optional): Path to the directory containing abnormal depth images for
            the test dataset.
            Defaults to ``None``.
        normal_test_depth_dir (str | Path | None, optional): Path to the directory containing
            normal depth images for the test dataset. Normal test images will be a split of `normal_dir` if `None`.
            Defaults to ``None``.
        split (str | Split | None): Fixed subset split that follows from folder structure on file system.
            Choose from [Split.FULL, Split.TRAIN, Split.TEST]
            Defaults to ``None``.
        extensions (tuple[str, ...] | None, optional): Type of the image extensions to read from the directory.
            Defaults to ``None``.

    Raises:
        ValueError: When task is set to classification and `mask_dir` is provided. When `mask_dir` is
            provided, `task` should be set to `segmentation`.
    """

    def __init__(
        self,
        name: str,
        task: TaskType,
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
    ) -> None:
        super().__init__(task)

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
        """Name of the dataset.

        Folder3D dataset overrides the name property to provide a custom name.
        """
        return self._name


def make_folder3d_dataset(  # noqa: C901
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
    """Make Folder Dataset.

    Args:
        normal_dir (str | Path): Path to the directory containing normal images.
        root (str | Path | None): Path to the root directory of the dataset.
            Defaults to ``None``.
        abnormal_dir (str | Path | None, optional): Path to the directory containing abnormal images.
            Defaults to ``None``.
        normal_test_dir (str | Path | None, optional): Path to the directory containing normal images for the test
        dataset. Normal test images will be a split of `normal_dir` if `None`.
            Defaults to ``None``.
        mask_dir (str | Path | None, optional): Path to the directory containing the mask annotations.
            Defaults to ``None``.
        normal_depth_dir (str | Path | None, optional): Path to the directory containing
            normal depth images for the test dataset. Normal test depth images will be a split of `normal_dir`
            Defaults to ``None``.
        abnormal_depth_dir (str | Path | None, optional): Path to the directory containing abnormal depth images for
            the test dataset.
            Defaults to ``None``.
        normal_test_depth_dir (str | Path | None, optional): Path to the directory containing normal depth images for
            the test dataset. Normal test images will be a split of `normal_dir` if `None`.
            Defaults to ``None``.
        split (str | Split | None, optional): Dataset split (ie., Split.FULL, Split.TRAIN or Split.TEST).
            Defaults to ``None``.
        extensions (tuple[str, ...] | None, optional): Type of the image extensions to read from the directory.
            Defaults to ``None``.

    Returns:
        DataFrame: an output dataframe containing samples for the requested split (ie., train or test)
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

    filenames = []
    labels = []
    dirs = {DirType.NORMAL: normal_dir}

    if abnormal_dir:
        dirs[DirType.ABNORMAL] = abnormal_dir

    if normal_test_dir:
        dirs[DirType.NORMAL_TEST] = normal_test_dir

    if normal_depth_dir:
        dirs[DirType.NORMAL_DEPTH] = normal_depth_dir

    if abnormal_depth_dir:
        dirs[DirType.ABNORMAL_DEPTH] = abnormal_depth_dir

    if normal_test_depth_dir:
        dirs[DirType.NORMAL_TEST_DEPTH] = normal_test_depth_dir

    if mask_dir:
        dirs[DirType.MASK] = mask_dir

    for dir_type, path in dirs.items():
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
            msg = """Mismatch between anomalous images and depth images. Make sure the mask files
            in 'xyz' folder follow the same naming convention as the anomalous images in the dataset
            (e.g. image: '000.png', depth: '000.tiff')."""
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

        # make sure all the files exist
        if not samples.mask_path.apply(
            lambda x: Path(x).exists() if x != "" else True,
        ).all():
            msg = f"Missing mask files. mask_dir={mask_dir}"
            raise FileNotFoundError(msg)
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
