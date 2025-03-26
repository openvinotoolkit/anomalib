"""VAD Dataset.

This module provides PyTorch Dataset implementation for the VAD dataset. The
dataset will be downloaded and extracted automatically if not found locally.

The dataset contains one category of industrial objects with both normal and
anomalous samples. It includes RGB images only (no pixel-level masks).

License:
    VAD dataset is released under the Creative Commons
    Attribution-NonCommercial-ShareAlike 4.0 International License
    (CC BY-NC-SA 4.0).
    https://creativecommons.org/licenses/by-nc-sa/4.0/

Reference:
    Aimira Baitieva, David Hurych, Victor Besnier, Olivier Bernard:
    Supervised Anomaly Detection for Complex Industrial Images; in:
    The IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024,
    pp. 17754-17762, DOI: 10.1109/CVPR52733.2024.01681.
"""

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from pathlib import Path

from pandas import DataFrame
from torchvision.transforms.v2 import Transform

from anomalib.data.datasets.base import AnomalibDataset
from anomalib.data.utils import LabelName, Split, validate_path

IMG_EXTENSIONS = (".png", ".PNG")
CATEGORIES = ("vad",)


class VADDataset(AnomalibDataset):
    """VAD dataset class.

    Dataset class for loading and processing VAD dataset images. Supports
    only classification task.

    Args:
        root (Path | str): Path to root directory containing the dataset.
            Defaults to ``"./datasets/VAD"``.
        category (str): Category name, must be one of ``CATEGORIES``.
            Defaults to ``"vad"``.
        augmentations (Transform, optional): Augmentations that should be applied to the input images.
            Defaults to ``None``.
        split (str | Split | None, optional): Dataset split - usually
            ``Split.TRAIN`` or ``Split.TEST``. Defaults to ``None``.

    Example:
        >>> from pathlib import Path
        >>> from anomalib.data.datasets import VADDataset
        >>> dataset = VADDataset(
        ...     root=Path("./datasets/VAD"),
        ...     category="vad",
        ...     split="train"
        ... )

        For classification tasks, each sample contains:

        >>> sample = dataset[0]
        >>> list(sample.keys())
        ['image_path', 'label', 'image']

        Images are PyTorch tensors with shape ``(C, H, W)``, masks have shape
        ``(H, W)``:

        >>> sample["image"].shape, sample["mask"].shape
        (torch.Size([3, 256, 256]), torch.Size([256, 256]))
    """

    def __init__(
        self,
        root: Path | str = "./datasets/VAD",
        category: str = "vad",
        augmentations: Transform | None = None,
        split: str | Split | None = None,
    ) -> None:
        super().__init__(augmentations=augmentations)

        self.root_category = Path(root) / Path(category)
        self.category = category
        self.split = split
        self.samples = make_vad_dataset(
            self.root_category,
            split=self.split,
            extensions=IMG_EXTENSIONS,
        )


def make_vad_dataset(
    root: str | Path,
    split: str | Split | None = None,
    extensions: Sequence[str] | None = None,
) -> DataFrame:
    """Create VAD AD samples by parsing the data directory structure.

    The files are expected to follow the structure:
        ``path/to/dataset/split/category/image_filename.png``

    Args:
        root (Path | str): Path to dataset root directory
        split (str | Split | None, optional): Dataset split (train or test)
            Defaults to ``None``.
        extensions (Sequence[str] | None, optional): Valid file extensions
            Defaults to ``None``.

    Returns:
        DataFrame: Dataset samples with columns:
            - path: Base path to dataset
            - split: Dataset split (train/test)
            - label: Class label
            - image_path: Path to image file
            - mask_path: Empty path for compatibility, no masks are available for this dataset
            - label_index: Numeric label (0=normal, 1=abnormal)

    Example:
        >>> root = Path("./datasets/VAD/vad")
        >>> samples = make_vad_dataset(root, split="train")
        >>> samples.head()
           path                split label image_path           mask_path label_index
        0  datasets/VAD/vad train good  [...]/good/2041.png                         0
        1  datasets/VAD/vad train good  [...]/good/2565.png                         0

    Raises:
        RuntimeError: If no valid images are found
        MisMatchError: If anomalous images and masks don't match
    """
    if extensions is None:
        extensions = IMG_EXTENSIONS

    root = validate_path(root)
    samples_list = [(str(root),) + f.parts[-3:] for f in root.glob(r"**/*") if f.suffix in extensions]
    if not samples_list:
        msg = f"Found 0 images in {root}"
        raise RuntimeError(msg)

    samples = DataFrame(samples_list, columns=["path", "split", "label", "image_path"])

    # Modify image_path column by converting to absolute path
    samples["image_path"] = samples.path + "/" + samples.split + "/" + samples.label + "/" + samples.image_path

    # Create label index for normal (0) and anomalous (1) images.
    samples.loc[(samples.label == "good"), "label_index"] = LabelName.NORMAL
    samples.loc[(samples.label != "good"), "label_index"] = LabelName.ABNORMAL
    samples.label_index = samples.label_index.astype(int)

    # assign mask paths to empty string
    samples["mask_path"] = ""

    # infer the task type
    samples.attrs["task"] = "classification"

    if split:
        samples = samples[samples.split == split].reset_index(drop=True)

    return samples
