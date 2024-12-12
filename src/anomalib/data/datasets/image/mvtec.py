"""MVTec AD Dataset.

Description:
    This script contains PyTorch Dataset for the MVTec AD dataset.
    If the dataset is not on the file system, the script downloads and extracts
    the dataset and create PyTorch data objects.

License:
    MVTec AD dataset is released under the Creative Commons
    Attribution-NonCommercial-ShareAlike 4.0 International License
    (CC BY-NC-SA 4.0)(https://creativecommons.org/licenses/by-nc-sa/4.0/).

References:
    - Paul Bergmann, Kilian Batzner, Michael Fauser, David Sattlegger, Carsten Steger:
      The MVTec Anomaly Detection Dataset: A Comprehensive Real-World Dataset for
      Unsupervised Anomaly Detection; in: International Journal of Computer Vision
      129(4):1038-1059, 2021, DOI: 10.1007/s11263-020-01400-4.

    - Paul Bergmann, Michael Fauser, David Sattlegger, Carsten Steger: MVTec AD —
      A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection;
      in: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR),
      9584-9592, 2019, DOI: 10.1109/CVPR.2019.00982.
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from pathlib import Path

from pandas import DataFrame
from torchvision.transforms.v2 import Transform

from anomalib.data.datasets.base import AnomalibDataset
from anomalib.data.errors import MisMatchError
from anomalib.data.utils import LabelName, Split, validate_path

IMG_EXTENSIONS = (".png", ".PNG")
CATEGORIES = (
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
)


class MVTecDataset(AnomalibDataset):
    """MVTec dataset class.

    Args:
        root (Path | str): Path to the root of the dataset.
            Defaults to ``./datasets/MVTec``.
        category (str): Sub-category of the dataset, e.g. 'bottle'
            Defaults to ``bottle``.
        transform (Transform, optional): Transforms that should be applied to the input images.
            Defaults to ``None``.
        split (str | Split | None): Split of the dataset, usually Split.TRAIN or Split.TEST
            Defaults to ``None``.

    Examples:
        .. code-block:: python

            from anomalib.data.image.mvtec import MVTecDataset
            from anomalib.data.utils.transforms import get_transforms

            transform = get_transforms(image_size=256)
            dataset = MVTecDataset(
                task="classification",
                transform=transform,
                root='./datasets/MVTec',
                category='zipper',
            )
            dataset.setup()
            print(dataset[0].keys())
            # Output: dict_keys(['image_path', 'label', 'image'])

        When the task is segmentation, the dataset will also contain the mask:

        .. code-block:: python

            dataset.task = "segmentation"
            dataset.setup()
            print(dataset[0].keys())
            # Output: dict_keys(['image_path', 'label', 'image', 'mask_path', 'mask'])

        The image is a torch tensor of shape (C, H, W) and the mask is a torch tensor of shape (H, W).

        .. code-block:: python

            print(dataset[0]["image"].shape, dataset[0]["mask"].shape)
            # Output: (torch.Size([3, 256, 256]), torch.Size([256, 256]))

    """

    def __init__(
        self,
        root: Path | str = "./datasets/MVTec",
        category: str = "bottle",
        transform: Transform | None = None,
        split: str | Split | None = None,
    ) -> None:
        super().__init__(transform=transform)

        self.root_category = Path(root) / Path(category)
        self.category = category
        self.split = split
        self.samples = make_mvtec_dataset(self.root_category, split=self.split, extensions=IMG_EXTENSIONS)


def make_mvtec_dataset(
    root: str | Path,
    split: str | Split | None = None,
    extensions: Sequence[str] | None = None,
) -> DataFrame:
    """Create MVTec AD samples by parsing the MVTec AD data file structure.

    The files are expected to follow the structure:
        path/to/dataset/split/category/image_filename.png
        path/to/dataset/ground_truth/category/mask_filename.png

    This function creates a dataframe to store the parsed information based on the following format:

    +---+---------------+-------+---------+---------------+---------------------------------------+-------------+
    |   | path          | split | label   | image_path    | mask_path                             | label_index |
    +===+===============+=======+=========+===============+=======================================+=============+
    | 0 | datasets/name | test  | defect  | filename.png  | ground_truth/defect/filename_mask.png | 1           |
    +---+---------------+-------+---------+---------------+---------------------------------------+-------------+

    Args:
        root (Path): Path to dataset
        split (str | Split | None, optional): Dataset split (ie., either train or test).
            Defaults to ``None``.
        extensions (Sequence[str] | None, optional): List of file extensions to be included in the dataset.
            Defaults to ``None``.

    Examples:
        The following example shows how to get training samples from MVTec AD bottle category:

        >>> root = Path('./MVTec')
        >>> category = 'bottle'
        >>> path = root / category
        >>> path
        PosixPath('MVTec/bottle')

        >>> samples = make_mvtec_dataset(path, split='train', split_ratio=0.1, seed=0)
        >>> samples.head()
           path         split label image_path                           mask_path                   label_index
        0  MVTec/bottle train good MVTec/bottle/train/good/105.png MVTec/bottle/ground_truth/good/105_mask.png 0
        1  MVTec/bottle train good MVTec/bottle/train/good/017.png MVTec/bottle/ground_truth/good/017_mask.png 0
        2  MVTec/bottle train good MVTec/bottle/train/good/137.png MVTec/bottle/ground_truth/good/137_mask.png 0
        3  MVTec/bottle train good MVTec/bottle/train/good/152.png MVTec/bottle/ground_truth/good/152_mask.png 0
        4  MVTec/bottle train good MVTec/bottle/train/good/109.png MVTec/bottle/ground_truth/good/109_mask.png 0

    Returns:
        DataFrame: an output dataframe containing the samples of the dataset.
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

    # separate masks from samples
    mask_samples = samples.loc[samples.split == "ground_truth"].sort_values(by="image_path", ignore_index=True)
    samples = samples[samples.split != "ground_truth"].sort_values(by="image_path", ignore_index=True)

    # assign mask paths to anomalous test images
    samples["mask_path"] = ""
    samples.loc[
        (samples.split == "test") & (samples.label_index == LabelName.ABNORMAL),
        "mask_path",
    ] = mask_samples.image_path.to_numpy()

    # assert that the right mask files are associated with the right test images
    abnormal_samples = samples.loc[samples.label_index == LabelName.ABNORMAL]
    if (
        len(abnormal_samples)
        and not abnormal_samples.apply(lambda x: Path(x.image_path).stem in Path(x.mask_path).stem, axis=1).all()
    ):
        msg = """Mismatch between anomalous images and ground truth masks. Make sure t
        he mask files in 'ground_truth' folder follow the same naming convention as the
        anomalous images in the dataset (e.g. image: '000.png', mask: '000.png' or '000_mask.png')."""
        raise MisMatchError(msg)

    # infer the task type
    samples.attrs["task"] = "classification" if (samples["mask_path"] == "").all() else "segmentation"

    if split:
        samples = samples[samples.split == split].reset_index(drop=True)

    return samples
