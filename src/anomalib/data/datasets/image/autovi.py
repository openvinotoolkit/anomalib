# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""AutoVI Dataset.

This module provides a PyTorch Dataset implementation for the AutoVI
(Automotive Visual Inspection) dataset. The dataset is distributed as 6
separate zip files on Zenodo (https://zenodo.org/records/10459003), one zip
per class. Downloading is handled by the :class:`AutoVI` datamodule; this
module only parses an already-extracted on-disk layout.

The dataset follows the folder layout inside every zip:

    <category>/
    ├── train/
    │   └── good/
    │       └── *.png
    ├── test/
    │   ├── good/
    │   │   └── *.png
    │   └── <defect_type>/
    │       └── <image_number>.png
    └── ground_truth/
        └── <defect_type>/
            └── <image_number>/
                └── 0000.png   (pixel-level segmentation mask)

The ground-truth mask for a given anomalous test image is stored in a
subdirectory named after the image stem. For example:

    test/unclipped/0089.png  →  ground_truth/unclipped/0089/0000.png

Categories (one zip per category on Zenodo):
    engine_wiring, pipe_clip, pipe_staple,
    tank_screw, underbody_pipes, underbody_screw

Data License:
    Copyright © 2023-2024 Renault Group.
    Released under Creative Commons Attribution-NonCommercial-ShareAlike 4.0
    International License (CC BY-NC-SA 4.0).
    https://creativecommons.org/licenses/by-nc-sa/4.0/

Reference:
    Carvalho, P., Lafou, M., Durupt, A., Leblanc, A., & Grandvalet, Y. (2024).
    The Automotive Visual Inspection Dataset (AutoVI): A Genuine Industrial
    Production Dataset for Unsupervised Anomaly Detection [Dataset].
    https://doi.org/10.5281/zenodo.10459003
"""

from collections.abc import Sequence
from pathlib import Path

from pandas import DataFrame, Series
from torchvision.transforms.v2 import Transform

from anomalib.data.datasets.base import AnomalibDataset
from anomalib.data.errors import MisMatchError
from anomalib.data.utils import LabelName, Split, validate_path
from anomalib.utils.path import get_datasets_dir

IMG_EXTENSIONS = (".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG")

#: The 6 classes distributed as individual zip files on Zenodo.
CATEGORIES = (
    "engine_wiring",
    "pipe_clip",
    "pipe_staple",
    "tank_screw",
    "underbody_pipes",
    "underbody_screw",
)

#: Zenodo download URLs for each category zip.
DOWNLOAD_URLS: dict[str, str] = {
    cat: f"https://zenodo.org/records/10459003/files/{cat}.zip?download=1" for cat in CATEGORIES
}

DOWNLOAD_HASHES = {
    "engine_wiring": "252590d3249f7fbdf83a7e9b735ef0df175adc218460247e9a63bef1c03d420c",
    "pipe_clip": "955bb17b3a471e23979f46f998a4723acd00cf3afc4d5edf0578dfb6ea80d6c3",
    "pipe_staple": "fb9287f2cc86d660310e9886fdebbe2bd17269e853e29d06d708c6a996df1b18",
    "tank_screw": "48d7193164b36de03cc10c9f7b1b64ea98a0ce7aa57867c8f1d96341c497b4b0",
    "underbody_pipes": "fc1e53336d46fb2317d71e95c011bac012b609f2898fdb02780c920c19a113c7",
    "underbody_screw": "3e9bf6a43033a22c7c9f927a43c392a3b037d22580a27a48d379d4504d9cc6cf",
}


class AutoVIDataset(AnomalibDataset):
    """AutoVI Dataset class.

    Dataset class for loading and processing AutoVI images. Supports both
    classification and segmentation tasks.  The folder layout inside each
    category is identical to MVTec AD, so ``make_autovi_dataset`` reuses the
    same parsing logic.

    Args:
        root (Path | str | None): Path to the root directory containing
            the dataset (i.e. the parent of the per-category folders).
            Defaults to ``None`` (uses anomalib's default datasets dir).
        category (str): Category name.  Must be one of :data:`CATEGORIES`.
            Defaults to ``"engine_wiring"``.
        augmentations (Transform | None): Augmentations applied to input
            images.  Defaults to ``None``.
        split (str | Split | None): Dataset split — typically
            ``Split.TRAIN`` or ``Split.TEST``.  Defaults to ``None``.

    Example:
        >>> from pathlib import Path
        >>> from anomalib.data.datasets.image import AutoVIDataset
        >>> dataset = AutoVIDataset(
        ...     root=Path("./datasets/AutoVI"),
        ...     category="pipe_clip",
        ...     split="train",
        ... )

        Each sample from a classification task contains:

        >>> sample = dataset[0]
        >>> list(sample.keys())
        ['image_path', 'label', 'image']

        For segmentation tasks, samples also include mask information:

        >>> dataset.task = "segmentation"
        >>> sample = dataset[0]
        >>> list(sample.keys())
        ['image_path', 'label', 'image', 'mask_path', 'mask']
    """

    def __init__(
        self,
        root: Path | str | None = None,
        category: str = "engine_wiring",
        augmentations: Transform | None = None,
        split: str | Split | None = None,
    ) -> None:
        super().__init__(augmentations=augmentations)

        if category not in CATEGORIES:
            msg = f"Invalid category '{category}'. Choose from: {CATEGORIES}"
            raise ValueError(msg)

        root = root if root is not None else get_datasets_dir() / "AutoVI"
        self.root_category = Path(root) / category
        self.category = category
        self.split = split

        self.samples = make_autovi_dataset(
            self.root_category,
            split=self.split,
            extensions=IMG_EXTENSIONS,
        )


def make_autovi_dataset(
    root: str | Path,
    split: str | Split | None = None,
    extensions: Sequence[str] | None = None,
) -> DataFrame:
    """Create AutoVI samples by parsing the per-category directory structure.

    The files are expected to follow the MVTec-compatible structure::

        <root>/train/good/<image>.png
        <root>/test/<defect_or_good>/<image>.png
        <root>/ground_truth/<defect>/<image_stem>/0000.png

    Args:
        root (Path | str): Path to the **category** root directory
            (e.g. ``./datasets/AutoVI/pipe_clip``).
        split (str | Split | None): Dataset split (``"train"`` or ``"test"``).
            Defaults to ``None`` (returns all splits).
        extensions (Sequence[str] | None): Valid file extensions.
            Defaults to :data:`IMG_EXTENSIONS`.

    Returns:
        DataFrame: Dataset samples with columns:
            - ``path``        — base path to the category directory
            - ``split``       — dataset split (``train`` / ``test``)
            - ``label``       — class label folder name (``good`` / defect)
            - ``image_path``  — absolute path to the image file
            - ``mask_path``   — absolute path to the mask (anomalous test only)
            - ``label_index`` — 0 = normal, 1 = abnormal

    Raises:
        RuntimeError: If no valid images are found under ``root``.
        MisMatchError: If anomalous images and masks do not match.
    """
    if extensions is None:
        extensions = IMG_EXTENSIONS

    root = validate_path(root)

    # Only collect files that are exactly 3 levels deep under a known split dir
    # i.e. train/<label>/<file> or test/<label>/<file>
    # This excludes stray root-level files and the 4-level ground_truth masks
    valid_splits = {"train", "test"}
    samples_list = [
        (str(root), *f.parts[-3:])
        for f in root.glob("**/*")
        if f.suffix in extensions and f.parts[-3] in valid_splits  # parts[-3] = split dir name
    ]
    if not samples_list:
        msg = f"Found 0 images in {root}"
        raise RuntimeError(msg)

    samples = DataFrame(
        samples_list,
        columns=["path", "split", "label", "image_path"],
    )

    # Rebuild absolute image_path
    samples["image_path"] = (
        samples["path"] + "/" + samples["split"] + "/" + samples["label"] + "/" + samples["image_path"]
    )

    # Label index: 0 = normal ("good"), 1 = abnormal
    import numpy as np

    samples["label_index"] = np.where(
        samples["label"] == "good",
        int(LabelName.NORMAL),
        int(LabelName.ABNORMAL),
    ).astype(int)

    # Build mask lookup keyed by (defect_type, image_stem) to avoid stem
    # collisions across defect folders. Layout: ground_truth/<defect>/<stem>/0000.png
    mask_lookup: dict[tuple[str, str], str] = {
        (f.parent.parent.name, f.parent.name): str(f)
        for f in root.glob("ground_truth/*/*/0000.*")
        if f.suffix in extensions and f.stem == "0000"
    }

    def lookup_mask(row: Series) -> str:
        if row["split"] == "test" and row["label_index"] == LabelName.ABNORMAL:
            stem = Path(row["image_path"]).stem
            return mask_lookup.get((row["label"], stem), "")
        return ""

    samples["mask_path"] = samples.apply(lookup_mask, axis=1)

    # Verify all anomalous test images got a mask
    abnormal_test = samples[(samples["split"] == "test") & (samples["label_index"] == LabelName.ABNORMAL)]
    missing = abnormal_test[abnormal_test["mask_path"] == ""]
    if not missing.empty:
        msg = (
            f"{len(missing)} anomalous test image(s) have no matching ground-truth mask. "
            f"First missing: {missing.iloc[0]['image_path']}"
        )
        raise MisMatchError(msg)

    # Infer task type
    samples.attrs["task"] = "classification" if (samples["mask_path"] == "").all() else "segmentation"

    if split:
        samples = samples[samples["split"] == split].reset_index(drop=True)

    return samples
