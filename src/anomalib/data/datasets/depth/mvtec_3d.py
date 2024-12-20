"""MVTec 3D-AD Datamodule.

This module provides PyTorch Dataset, Dataloader and PyTorch Lightning DataModule for
the MVTec 3D-AD dataset. If the dataset is not available locally, it will be
downloaded and extracted automatically.

License:
    MVTec 3D-AD dataset is released under the Creative Commons
    Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0)
    https://creativecommons.org/licenses/by-nc-sa/4.0/

Reference:
    Paul Bergmann, Xin Jin, David Sattlegger, Carsten Steger: The MVTec 3D-AD
    Dataset for Unsupervised 3D Anomaly Detection and Localization. In: Proceedings
    of the 17th International Joint Conference on Computer Vision, Imaging and
    Computer Graphics Theory and Applications - Volume 5: VISAPP, 202-213, 2022
    DOI: 10.5220/0010865000003124
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from pathlib import Path

from pandas import DataFrame
from torchvision.transforms.v2 import Transform

from anomalib.data.datasets.base.depth import AnomalibDepthDataset
from anomalib.data.errors import MisMatchError
from anomalib.data.utils import LabelName, Split, validate_path

IMG_EXTENSIONS = [".png", ".PNG", ".tiff"]
CATEGORIES = (
    "bagel",
    "cable_gland",
    "carrot",
    "cookie",
    "dowel",
    "foam",
    "peach",
    "potato",
    "rope",
    "tire",
)


class MVTec3DDataset(AnomalibDepthDataset):
    """MVTec 3D dataset class.

    Args:
        root (Path | str): Path to the root of the dataset.
            Defaults to ``"./datasets/MVTec3D"``.
        category (str): Category name, e.g. ``"bagel"``.
            Defaults to ``"bagel"``.
        augmentations (Transform, optional): Augmentations that should be applied to the input images.
            Defaults to ``None``.
        split (str | Split | None): Dataset split - usually ``Split.TRAIN`` or
            ``Split.TEST``. Defaults to ``None``.

    Example:
        >>> from pathlib import Path
        >>> dataset = MVTec3DDataset(
        ...     root=Path("./datasets/MVTec3D"),
        ...     category="bagel",
        ...     split="train"
        ... )
    """

    def __init__(
        self,
        root: Path | str = "./datasets/MVTec3D",
        category: str = "bagel",
        augmentations: Transform | None = None,
        split: str | Split | None = None,
    ) -> None:
        super().__init__(augmentations=augmentations)

        self.root_category = Path(root) / Path(category)
        self.split = split
        self.samples = make_mvtec_3d_dataset(
            self.root_category,
            split=self.split,
            extensions=IMG_EXTENSIONS,
        )


def make_mvtec_3d_dataset(
    root: str | Path,
    split: str | Split | None = None,
    extensions: Sequence[str] | None = None,
) -> DataFrame:
    """Create MVTec 3D-AD samples by parsing the data directory structure.

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
        >>> root = Path("./datasets/MVTec3D/bagel")
        >>> samples = make_mvtec_3d_dataset(root, split="train")
        >>> samples.head()
           path     split label image_path                  mask_path
        0  MVTec3D  train good  train/good/rgb/105.png     gt/105.png
        1  MVTec3D  train good  train/good/rgb/017.png     gt/017.png

    Raises:
        RuntimeError: If no images are found in the root directory.
        MisMatchError: If there is a mismatch between images and their
            corresponding mask/depth files.
    """
    if extensions is None:
        extensions = IMG_EXTENSIONS

    root = validate_path(root)
    samples_list = [(str(root),) + f.parts[-4:] for f in root.glob(r"**/*") if f.suffix in extensions]
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
        mask_samples.path + "/" + samples.split + "/" + samples.label + "/" + "gt/" + samples.file_name
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
        samples = samples[samples.split == split].reset_index(drop=True)

    return samples
