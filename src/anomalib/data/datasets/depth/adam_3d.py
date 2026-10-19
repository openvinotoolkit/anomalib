# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""3D-ADAM Datamodule.

This module provides PyTorch Dataset for
the 3D-ADAM dataset. If the dataset is not available locally, it will be
downloaded and extracted automatically.

License:
    3D-ADAM dataset is released under the Creative Commons
    Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0)
    https://creativecommons.org/licenses/by-nc-sa/4.0/

Reference: https://arxiv.org/abs/2507.07838

"""

from collections.abc import Sequence
from pathlib import Path

from pandas import DataFrame
from torchvision.transforms.v2 import Transform

from anomalib.data.datasets.base.depth import AnomalibDepthDataset
from anomalib.data.errors import MisMatchError
from anomalib.data.utils import LabelName, Split, validate_path
from anomalib.utils.path import get_datasets_dir

IMG_EXTENSIONS = [".png", ".PNG", ".tiff"]
CATEGORIES = (
    "1m1",
    "1m2",
    "1m3",
    "2m1",
    "2m2h",
    "2m2m",
    "3m1",
    "3m2",
    "3m2c",
    "4m1",
    "4m2",
    "4m2c",
    "gripper_closed",
    "gripper_open",
    "helicalgear1",
    "helicalgear2",
    "rackgear",
    "spiralgear",
    "spurgear",
    "tapa2m1",
    "tapa3m1",
    "tapa4m1",
    "tapatbb",
)


class ADAM3DDataset(AnomalibDepthDataset):
    """3D ADAM dataset class.

    Args:
        root (Path | str | None): Path to the root of the dataset.
            Defaults to ``None``.
        category (str): Category name, e.g. ``"1m1"``.
            Defaults to ``"1m1"``.
        augmentations (Transform, optional): Augmentations that should be applied to the input images.
            Defaults to ``None``.
        split (str | Split | None): Dataset split - usually ``Split.TRAIN`` or
            ``Split.TEST``. Defaults to ``None``.

    Example:
        >>> from pathlib import Path
        >>> dataset = ADAM3DDataset(
        ...     root=Path("./datasets/ADAM3D"),
        ...     category="1m1",
        ...     split="train"
        ... )
    """

    def __init__(
        self,
        root: Path | str | None = None,
        category: str = "1m1",
        augmentations: Transform | None = None,
        split: str | Split | None = None,
    ) -> None:
        super().__init__(augmentations=augmentations)

        root = root if root is not None else get_datasets_dir() / "ADAM3D"

        self.root_category = Path(root) / Path(category)
        self.split = split
        self.samples = make_adam_3d_dataset(
            self.root_category,
            split=self.split,
            extensions=IMG_EXTENSIONS,
        )


def make_adam_3d_dataset(
    root: str | Path,
    split: str | Split | None = None,
    extensions: Sequence[str] | None = None,
) -> DataFrame:
    """Create 3D-ADAM samples by parsing the data directory structure.

    The files are expected to follow this structure::

        path/to/dataset/split/category/image_filename.png
        path/to/dataset/ground_truth/category/mask_filename.png

    The function creates a DataFrame with the following format::

        +---+---------------+-------+---------+---------------+--------------------+
        |   | path          | split | label   | image_path    | mask_path         |
        +---+---------------+-------+---------+---------------+--------------------+
        | 0 | datasets/name | test  | defect  | filename.png  | defect/mask.png   |
        +---+---------------+-------+---------+---------------+--------------------+

    Args:
        root (Path | str): Path to the dataset root directory.
        split (str | Split | None, optional): Dataset split (e.g., ``"train"`` or
            ``"test"``). Defaults to ``None``.
        extensions (Sequence[str] | None, optional): List of valid file extensions.
            Defaults to ``None``.

    Returns:
        DataFrame: DataFrame containing the dataset samples.

    Example:
        >>> from pathlib import Path
        >>> root = Path("./datasets/ADAM3D/1m1")
        >>> samples = make_adam_3d_dataset(root, split="train")
        >>> samples.head()
           path     split label image_path                  mask_path
        0  ADAM3D  train good  train/good/rgb/001_C.png     ground_truth/001_C.png
        1  ADAM3D  train good  train/good/rgb/015_D.png     ground_truth/015_D.png

    Raises:
        RuntimeError: If no images are found in the root directory.
        MisMatchError: If there is a mismatch between images and their
            corresponding mask/depth files.
    """
    if extensions is None:
        extensions = IMG_EXTENSIONS

    root = validate_path(root)
    samples_list = [(str(root), *f.parts[-4:]) for f in root.glob(r"**/*") if f.suffix in extensions]
    if not samples_list:
        msg = f"Found 0 images in {root}"
        raise RuntimeError(msg)

    samples = DataFrame(
        samples_list,
        columns=["path", "split", "label", "type", "file_name"],
    )

    # Modify image_path column by converting to absolute path
    samples.loc[(samples.type == "rgb"), "image_path"] = (
        samples.path + "/" + samples.split + "/" + samples.label + "/" + "rgb/" + samples.file_name
    )
    samples.loc[(samples.type == "rgb"), "depth_path"] = (
        samples.path
        + "/"
        + samples.split
        + "/"
        + samples.label
        + "/"
        + "xyz/"
        + samples.file_name.str.split(".").str[0]
        + ".tiff"
    )

    # Create label index for normal (0) and anomalous (1) images.
    samples.loc[(samples.label == "good"), "label_index"] = LabelName.NORMAL
    samples.loc[(samples.label != "good"), "label_index"] = LabelName.ABNORMAL
    samples.label_index = samples.label_index.astype(int)

    # separate masks from samples
    mask_samples = samples.loc[((samples.split == "test") & (samples.type == "rgb"))].sort_values(
        by="image_path",
        ignore_index=True,
    )
    samples = samples.sort_values(by="image_path", ignore_index=True)

    # assign mask paths to all test images
    samples.loc[((samples.split == "test") & (samples.type == "rgb")), "mask_path"] = (
        mask_samples.path + "/" + samples.split + "/" + samples.label + "/" + "ground_truth/" + samples.file_name
    )
    samples = samples.dropna(subset=["image_path"])
    samples = samples.astype({"image_path": "str", "mask_path": "str", "depth_path": "str"})

    # assert that the right mask files are associated with the right test images
    mismatch_masks = (
        samples.loc[samples.label_index == LabelName.ABNORMAL]
        .apply(lambda x: Path(x.image_path).stem in Path(x.mask_path).stem, axis=1)
        .all()
    )
    if not mismatch_masks:
        msg = (
            "Mismatch between anomalous images and ground truth masks. Ensure mask "
            "files in 'ground_truth' folder follow the same naming convention as "
            "the anomalous images (e.g. image: '000.png', mask: '000.png')."
        )
        raise MisMatchError(msg)

    mismatch_depth = (
        samples.loc[samples.label_index == LabelName.ABNORMAL]
        .apply(lambda x: Path(x.image_path).stem in Path(x.depth_path).stem, axis=1)
        .all()
    )
    if not mismatch_depth:
        msg = (
            "Mismatch between anomalous images and depth images. Ensure depth "
            "files in 'xyz' folder follow the same naming convention as the "
            "anomalous images (e.g. image: '000.png', depth: '000.tiff')."
        )
        raise MisMatchError(msg)

    # infer the task type
    samples.attrs["task"] = "classification" if (samples["mask_path"] == "").all() else "segmentation"

    if split:
        split_value = split.value if isinstance(split, Split) else split
        samples = samples[samples.split == split_value].reset_index(drop=True)

    return samples
