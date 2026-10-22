# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""LIBAD Data Module.

This module provides a PyTorch Lightning DataModule for the LIBAD dataset.
The dataset can be downloaded automatically with authenticated, approved access to
Hugging Face (https://huggingface.co/datasets/Evenrose/LIBAD), or extracted manually.

Example:
    Create a LIBAD datamodule::

        >>> from anomalib.data import LIBAD
        >>> datamodule = LIBAD(
        ...     root="./datasets/LIBAD",
        ...     category="1_wrinkling",
        ...     modality="A"
        ... )

Notes:
    For manual downloads, extract the dataset into the expected
    directory structure is::

        datasets/
        └── LIBAD/
            ├── 1_wrinkling/
            │   ├── normal/
            │   │   ├── A01A.tiff
            │   │   ├── A01B.tiff
            │   │   ├── A01X.tiff
            │   │   └── A01L.tiff
            │   └── anomaly/
            │       ├── ...

License:
    LIBAD dataset is released under the CC-BY-4.0 License.

Reference:
    Wenbo Sui and Daniel Lichau and Harold Phelippeau and Zhao Liu. (2026).
    LIBAD: A Multimodal Anomaly Detection Benchmark for Li-Ion Battery Electrode Manufacturing.
    https://arxiv.org/abs/2608.07958
"""

import logging
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

from lightning_utilities.core.imports import module_available
from torchvision.transforms.v2 import Transform

if TYPE_CHECKING or module_available("huggingface_hub"):
    from huggingface_hub import get_token, hf_hub_download
    from huggingface_hub.utils import (
        EntryNotFoundError,
        GatedRepoError,
        HfHubHTTPError,
        LocalEntryNotFoundError,
        RepositoryNotFoundError,
        RevisionNotFoundError,
    )

    # Errors that indicate the dataset cannot be downloaded automatically
    HF_DOWNLOAD_ERRORS = (
        GatedRepoError,
        HfHubHTTPError,
        RepositoryNotFoundError,
        RevisionNotFoundError,
        EntryNotFoundError,
        LocalEntryNotFoundError,
        OSError,
    )
else:
    get_token = None
    hf_hub_download = None
    HF_DOWNLOAD_ERRORS = (Exception,)

from anomalib.data.datamodules.base.image import AnomalibDataModule
from anomalib.data.datasets.image.libad import CATEGORIES, LIBADDataset
from anomalib.data.utils import TestSplitMode, ValSplitMode, split_by_label
from anomalib.data.utils.download import extract
from anomalib.utils.path import resolve_dataset_root

logger = logging.getLogger(__name__)


class LIBAD(AnomalibDataModule):
    """LIBAD Lightning Data Module.

    Args:
        root (Path | str | None): Path to the root of the dataset.
            Defaults to ``"./datasets/LIBAD"``.
        category (str): Category of the LIBAD dataset (e.g. ``"1_wrinkling"``).
            Defaults to ``"1_wrinkling"``.
        modality (str): Modality suffix to filter images by.
            Options: 'A' (side A), 'B' (side B), 'L' (X-ray Low), 'X' (X-ray High).
            Defaults to ``"A"``.
        train_batch_size (int, optional): Training batch size.
            Defaults to ``32``.
        eval_batch_size (int, optional): Test batch size.
            Defaults to ``32``.
        num_workers (int, optional): Number of workers.
            Defaults to ``8``.
        train_augmentations (Transform | None): Augmentations to apply to the training images
            Defaults to ``None``.
        val_augmentations (Transform | None): Augmentations to apply to the validation images.
            Defaults to ``None``.
        test_augmentations (Transform | None): Augmentations to apply to the test images.
            Defaults to ``None``.
        augmentations (Transform | None): General augmentations to apply if stage-specific
            augmentations are not provided.
        test_split_mode (TestSplitMode): Setting that determines how the testing
            subset is obtained.
            Defaults to ``TestSplitMode.FROM_DIR``.
        test_split_ratio (float): Fraction of images from the train set that will
            be reserved for testing.
            Defaults to ``0.2``.
        val_split_mode (ValSplitMode): Setting that determines how the validation
            subset is obtained.
            Defaults to ``ValSplitMode.SAME_AS_TEST``.
        val_split_ratio (float): Fraction of train or test images that will be
            reserved for validation.
            Defaults to ``0.5``.
        seed (int | None, optional): Seed which may be set to a fixed value for
            reproducibility.
            Defaults to ``None``.

    Example:
        To create the LIBAD datamodule, instantiate the class and call
        ``setup``::

            >>> from anomalib.data import LIBAD
            >>> datamodule = LIBAD(
            ...     root="./datasets/LIBAD",
            ...     category="1_wrinkling",
            ...     train_batch_size=32,
            ...     eval_batch_size=32,
            ...     num_workers=8,
            ... )
            >>> datamodule.setup()
    """

    def __init__(
        self,
        root: Path | str | None = "./datasets/LIBAD",
        category: str = "1_wrinkling",
        modality: str = "A",
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        num_workers: int = 8,
        train_augmentations: Transform | None = None,
        val_augmentations: Transform | None = None,
        test_augmentations: Transform | None = None,
        augmentations: Transform | None = None,
        test_split_mode: TestSplitMode | str = TestSplitMode.FROM_DIR,
        test_split_ratio: float = 0.2,
        val_split_mode: ValSplitMode | str = ValSplitMode.SAME_AS_TEST,
        val_split_ratio: float = 0.5,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            train_augmentations=train_augmentations,
            val_augmentations=val_augmentations,
            test_augmentations=test_augmentations,
            augmentations=augmentations,
            test_split_mode=test_split_mode,
            test_split_ratio=test_split_ratio,
            val_split_mode=val_split_mode,
            val_split_ratio=val_split_ratio,
            seed=seed,
        )

        root = resolve_dataset_root(root, "LIBAD")
        self.root = Path(root)
        self.category = category
        self.modality = modality

    def _setup(self, _stage: str | None = None) -> None:
        dataset = LIBADDataset(
            split=None,
            root=self.root,
            category=self.category,
            modality=self.modality,
        )

        self.train_data, self.test_data = split_by_label(dataset)

    def prepare_data(self) -> None:
        """Check if the dataset is available, downloading it if possible.

        This method checks if the specified dataset is available in the file
        system. If it is not, and the ``huggingface_hub`` package is
        installed, it attempts to automatically download and extract the
        dataset from Hugging Face using the token from the
        ``HF_TOKEN`` environment variable or a cached ``hf auth login``
        session. Automatic download only succeeds if the user has already
        requested and been granted access to the dataset on Hugging Face.
        """
        if self.category not in CATEGORIES:
            msg = f"Unknown category {self.category}. Must be one of {CATEGORIES}"
            raise ValueError(msg)

        if (self.root / self.category).is_dir():
            logger.info("Found the dataset.")
            return

        if not module_available("huggingface_hub"):
            logger.error(
                "Dataset not found and huggingface_hub is not installed. Please install it or download manually.",
            )
            msg = "LIBAD dataset must be downloaded manually."
            raise FileNotFoundError(msg)

        if not get_token():
            logger.info(
                "No Hugging Face token found (HF_TOKEN env var or cached "
                "``hf auth login`` session). Skipping automatic download.",
            )
            msg = "LIBAD dataset must be downloaded manually."
            raise FileNotFoundError(msg)

        logger.info(
            "LIBAD dataset not found at %s. Attempting to download it from Hugging Face.",
            self.root,
        )

        self.root.mkdir(parents=True, exist_ok=True)
        try:
            with TemporaryDirectory(dir=self.root) as scratch_dir:
                logger.info("Downloading LIBAD.zip from Hugging Face.")
                downloaded_path = Path(
                    hf_hub_download(
                        repo_id="Evenrose/LIBAD",
                        repo_type="dataset",
                        revision="1f4c5f4dbef3b6bb3c122a2340dff5b19014bf74",
                        filename="LIBAD.zip",
                        local_dir=scratch_dir,
                    ),
                )
                extract(downloaded_path, Path(scratch_dir))

                extracted_dir = Path(scratch_dir) / "LIBAD"
                if extracted_dir.is_dir():
                    for item in extracted_dir.iterdir():
                        shutil.move(str(item), str(self.root / item.name))
                else:
                    for item in Path(scratch_dir).iterdir():
                        if item.is_dir() and item.name != "LIBAD.zip":
                            shutil.move(str(item), str(self.root / item.name))

        except HF_DOWNLOAD_ERRORS as exc:
            msg = "Failed to download LIBAD dataset from Hugging Face."
            raise FileNotFoundError(msg) from exc
